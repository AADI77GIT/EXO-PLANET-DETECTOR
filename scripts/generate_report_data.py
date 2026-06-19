import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.ml.cnn_classifier import TransitClassifier
from app.models import Detection, Star


async def build_summary(tic_id: int) -> dict:
    async with AsyncSessionLocal() as db:
        rows = await db.execute(select(Detection).where(Detection.tic_id == tic_id).order_by(Detection.created_at.desc()))
        detection = rows.scalars().first()
        if detection is None:
            raise SystemExit(f"No detection found for TIC {tic_id}")
        star = await db.get(Star, tic_id)
    metadata_path = settings.model_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    errors = detection.parameter_errors or {}
    return {
        "tic_id": tic_id,
        "sector": detection.sector if star is None else star.sector,
        "label": detection.label,
        "confidence": detection.confidence,
        "period_days": {"value": detection.period_days, "err": errors.get("period_err")},
        "duration_hours": {"value": detection.duration_hours, "err": errors.get("duration_err")},
        "depth_ppt": {"value": detection.depth_ppt, "err": errors.get("depth_err")},
        "detrending_method": f"{settings.detrending_method}, window_length={settings.wotan_window_length_days} days",
        "classifier_architecture": TransitClassifier().__class__.__name__ + ": CNN curve branch + scalar feature branch",
        "bls_period_grid_range": [settings.bls_min_period_days, settings.bls_max_period_days, settings.bls_period_samples],
        "augmentation_methods": metadata.get("augmentation_methods", ["Gaussian noise sigma=0.001", "random time flip", "random flux scale 0.98-1.02"]),
        "n_training_samples": metadata.get("n_training_samples", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tic_id", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = asyncio.run(build_summary(args.tic_id))
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
