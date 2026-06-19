from pathlib import Path

from fastapi import HTTPException, UploadFile
import numpy as np
from astropy.stats import sigma_clip
from wotan import flatten

from app.config import settings

VALID_FITS_EXTENSIONS = {".fits", ".fit"}


def validate_fits_upload(upload: UploadFile) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in VALID_FITS_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Uploaded file must be .fits or .fit")
    if upload.size is not None and upload.size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded FITS exceeds 50MB limit")


def _clean_arrays(time: np.ndarray, flux_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(time) & np.isfinite(flux_raw)
    time = time[finite]
    flux_raw = flux_raw[finite]
    if time.size < 128:
        raise HTTPException(status_code=500, detail="Light curve has too few valid cadences")
    if np.all(~np.isfinite(flux_raw)) or np.nanstd(flux_raw) == 0:
        raise HTTPException(status_code=500, detail="Light curve flux is invalid or all-NaN")

    # Transit dips are downward and often shallow; use a loose lower tail so real dips survive.
    clipped = sigma_clip(flux_raw, sigma_lower=8, sigma_upper=4, masked=True)
    keep = ~np.asarray(clipped.mask, dtype=bool)
    if keep.sum() < 128:
        raise HTTPException(status_code=500, detail="Sigma clipping removed too many cadences")
    return time[keep], flux_raw, flux_raw[keep]


def preprocess_light_curve(tic_id: int, sector: int) -> dict:
    if tic_id <= 0 or sector <= 0:
        raise HTTPException(status_code=422, detail="tic_id and sector must be positive integers")
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="lightkurve is not installed") from exc

    query = f"TIC {tic_id}"
    search = lk.search_lightcurve(query, mission="TESS", sector=sector)
    if len(search) == 0:
        raise HTTPException(status_code=404, detail="FITS not found")

    try:
        lc = search.download()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"FITS download failed: {exc}") from exc
    if lc is None:
        raise HTTPException(status_code=500, detail="FITS download failed")

    time = np.asarray(lc.time.value, dtype=np.float64)
    flux_raw = np.asarray(lc.flux.value, dtype=np.float64)
    time_clean, flux_raw_full, flux_clean = _clean_arrays(time, flux_raw)

    try:
        flattened = flatten(time_clean, flux_clean, method="biweight", window_length=settings.wotan_window_length_days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Wotan flattening failed: {exc}") from exc

    median = np.nanmedian(flattened)
    if not np.isfinite(median) or median == 0:
        raise HTTPException(status_code=500, detail="Invalid median during flux normalization")
    flux_norm = (flattened - median) / median
    if not np.all(np.isfinite(flux_norm)):
        raise HTTPException(status_code=500, detail="Normalized flux contains non-finite values")

    return {"time": time_clean, "flux_raw": flux_raw_full, "flux_clean": flux_clean, "flux_norm": flux_norm}
