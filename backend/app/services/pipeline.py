import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ml.autoencoder import denoise
from app.ml.cnn_classifier import get_classifier
from app.models import Detection, Job, JobStatus, Star
from app.services.bls import run_bls
from app.services.fitter import fit_transit_parameters
from app.services.plotter import generate_plots
from app.services.preprocess import preprocess_light_curve

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    detection_id: int
    tic_id: int
    sector: int
    label: str
    confidence: float
    period_days: float
    duration_hours: float
    depth_ppt: float
    parameter_errors: dict
    plot_paths: dict


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, t0: float, bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    phase = (((time - t0 + 0.5 * period) % period) - 0.5 * period) / period
    order = np.argsort(phase)
    phase = phase[order]
    flux = flux[order]
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    folded = np.array([
        np.nanmedian(flux[(phase >= edges[i]) & (phase < edges[i + 1])]) if np.any((phase >= edges[i]) & (phase < edges[i + 1])) else np.nan
        for i in range(bins)
    ])
    folded = np.nan_to_num(folded, nan=float(np.nanmedian(flux)))
    return centers.astype(np.float32), folded.astype(np.float32)


async def update_job(db: AsyncSession, job_id: str, status: JobStatus, *, error_msg: str | None = None, result_id: int | None = None) -> None:
    job = await db.get(Job, job_id)
    if job is None:
        return
    job.status = status.value
    job.error_msg = error_msg
    if result_id is not None:
        job.result_id = result_id
    job.updated_at = datetime.utcnow()
    await db.commit()


async def create_job(db: AsyncSession, job_id: str, tic_id: int, sector: int) -> Job:
    job = Job(job_id=job_id, tic_id=tic_id, sector=sector, status=JobStatus.pending.value)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def run_full_pipeline(tic_id: int, sector: int, db: AsyncSession, job_id: str | None = None) -> PipelineResult:
    stages = {}
    try:
        stages["preprocess"] = preprocess_light_curve(tic_id, sector)
        time = stages["preprocess"]["time"]
        flux_norm = stages["preprocess"]["flux_norm"]

        stages["denoise"] = denoise(flux_norm)
        flux_denoised = stages["denoise"]

        stages["bls"] = run_bls(time, flux_denoised)
        bls = stages["bls"]

        phase_time, phase_flux = phase_fold(time, flux_denoised, bls["period"], bls["t0"], bins=128)
        scalar_features = np.array([
            bls["depth_ppt"],
            bls["duration_hours"] / 24 / bls["period"],
            float(bls["secondary_eclipse_flag"]),
            bls["odd_even_diff"],
        ], dtype=np.float32)
        stages["classify"] = get_classifier().predict(phase_flux, scalar_features)
        pred = stages["classify"]

        stages["fit"] = fit_transit_parameters(phase_time, phase_flux, bls["period"])
        fit = stages["fit"]
        model_flux = fit.pop("model_flux")

        stages["plot"] = generate_plots(tic_id, time, stages["preprocess"]["flux_raw"], flux_denoised, phase_time, phase_flux, model_flux, bls["power_spectrum"])
        plot_paths = stages["plot"]

        processed_path = settings.processed_dir / f"{tic_id}_s{sector}_processed.npz"
        np.savez_compressed(processed_path, time=time, flux_norm=flux_norm, flux_denoised=flux_denoised)
        star = await db.get(Star, tic_id)
        if star is None:
            star = Star(tic_id=tic_id)
            db.add(star)
        star.sector = sector
        star.processed_path = str(processed_path)
        star.processed_at = datetime.utcnow()

        detection = Detection(
            tic_id=tic_id,
            sector=sector,
            label=pred["label"],
            confidence=pred["confidence"],
            period_days=fit["period_days"],
            duration_hours=fit["duration_hours"],
            depth_ppt=fit["depth_ppt"],
            parameter_errors={k: v for k, v in fit.items() if k.endswith("err") or k == "chi_squared" or k == "t0"},
            plot_paths=plot_paths,
        )
        db.add(detection)
        await db.commit()
        await db.refresh(detection)
        if job_id:
            await update_job(db, job_id, JobStatus.done, result_id=detection.id)
        return PipelineResult(detection.id, tic_id, sector, pred["label"], pred["confidence"], fit["period_days"], fit["duration_hours"], fit["depth_ppt"], detection.parameter_errors, plot_paths)
    except Exception as exc:
        failed_stage = next((name for name in ["preprocess", "denoise", "bls", "classify", "fit", "plot"] if name not in stages), "database")
        logger.exception("Pipeline failed at stage %s for TIC %s sector %s: %s", failed_stage, tic_id, sector, exc)
        if job_id:
            await update_job(db, job_id, JobStatus.failed, error_msg=f"{failed_stage}: {exc}")
        raise


async def latest_detection(db: AsyncSession, tic_id: int) -> Detection | None:
    rows = await db.execute(select(Detection).where(Detection.tic_id == tic_id).order_by(Detection.created_at.desc()))
    return rows.scalars().first()
