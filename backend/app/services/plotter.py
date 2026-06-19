from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import settings


def _style(ax):
    ax.set_facecolor("#0a0f1e")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(alpha=0.18, color="white")


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="#0a0f1e")
    plt.close(fig)
    return str(path)


def generate_plots(tic_id, time, flux_raw, flux_denoised, phase_time, phase_flux, model_flux, period_confidence_curve) -> dict:
    out = settings.results_dir
    time = np.asarray(time)
    flux_raw = np.asarray(flux_raw)
    flux_denoised = np.asarray(flux_denoised)
    raw_aligned = flux_raw[-len(time):] if len(flux_raw) != len(time) else flux_raw
    residual = np.abs(raw_aligned - np.nanmedian(raw_aligned))
    outliers = residual > 4 * (np.nanstd(raw_aligned) or 1)
    dip = flux_denoised < np.nanpercentile(flux_denoised, 5)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120, facecolor="#0a0f1e")
    ax.scatter(time, raw_aligned, s=4, c="white", alpha=0.5)
    ax.scatter(time[outliers], raw_aligned[outliers], s=8, c="#ef4444", alpha=0.8)
    ax.scatter(time[dip], raw_aligned[dip], s=8, c="#ff6b35", alpha=0.8)
    ax.set_title(f"TIC {tic_id} raw light curve")
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Flux")
    _style(ax)
    raw_path = _save(fig, out / f"{tic_id}_raw.png")

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120, facecolor="#0a0f1e")
    ax.plot(time, raw_aligned, color="#9ca3af", lw=0.7, alpha=0.55, label="raw")
    ax.plot(time, flux_denoised, color="#00d4d4", lw=1.0, label="denoised")
    ax.legend(facecolor="#0a0f1e", edgecolor="white", labelcolor="white")
    ax.set_title(f"TIC {tic_id} denoised signal")
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Normalized flux")
    _style(ax)
    denoised_path = _save(fig, out / f"{tic_id}_denoised.png")

    curve = period_confidence_curve or {"period": [], "power": []}
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120, facecolor="#0a0f1e")
    ax.plot(curve.get("period", []), curve.get("power", []), color="#a855f7", lw=1.0)
    powers = np.asarray(curve.get("power", []), dtype=float)
    if powers.size:
        ax.axhline(float(np.nanpercentile(powers, 95)), color="#ff6b35", ls="--", lw=1.0)
    ax.set_title(f"TIC {tic_id} transit probability")
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("BLS power")
    _style(ax)
    prob_path = _save(fig, out / f"{tic_id}_prob.png")

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120, facecolor="#0a0f1e")
    ax.scatter(phase_time, phase_flux, s=8, color="#00d4d4", alpha=0.7)
    ax.plot(phase_time, model_flux, color="#fbbf24", lw=1.2)
    ax.set_title(f"TIC {tic_id} phase-folded transit")
    ax.set_xlabel("Phase time [days]")
    ax.set_ylabel("Normalized flux")
    _style(ax)
    phase_path = _save(fig, out / f"{tic_id}_phase.png")

    return {"raw": raw_path, "denoised": denoised_path, "prob": prob_path, "phase": phase_path}
