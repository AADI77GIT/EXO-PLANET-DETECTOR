import numpy as np
from scipy.optimize import curve_fit


def trapezoid_model(phase: np.ndarray, t0: float, depth: float, duration_fraction: float, ingress_fraction: float) -> np.ndarray:
    x = np.abs(phase - t0)
    ingress_width = np.clip(ingress_fraction, 1e-4, 0.5) * duration_fraction
    flat_half = max(duration_fraction / 2 - ingress_width, 0)
    y = np.zeros_like(phase, dtype=np.float64)
    full = x <= flat_half
    slope = (x > flat_half) & (x <= duration_fraction / 2)
    y[full] = -depth
    y[slope] = -depth * (duration_fraction / 2 - x[slope]) / max(ingress_width, 1e-6)
    return y


def _clean_err(value: float | np.floating) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def fit_transit_parameters(phase_folded_time, phase_folded_flux, period_days) -> dict:
    x = np.asarray(phase_folded_time, dtype=np.float64)
    y = np.asarray(phase_folded_flux, dtype=np.float64)
    y = y - np.nanmedian(y)
    depth0 = abs(float(np.nanmin(y))) if y.size else 0.01
    p0 = [0.0, max(depth0, 1e-5), 0.1, 0.2]
    bounds = ([-0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5])
    try:
        params, cov = curve_fit(trapezoid_model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        diag = np.diag(cov) if cov is not None and cov.shape == (4, 4) else np.full(4, np.nan)
        errs = np.sqrt(np.where(np.isfinite(diag) & (diag >= 0), diag, np.nan))
    except Exception:
        params = np.asarray(p0)
        errs = np.full(4, np.nan)
    model = trapezoid_model(x, *params)
    chi_squared = float(np.nansum((y - model) ** 2))
    t0, depth, duration_fraction, _ = map(float, params)
    return {
        "period_days": float(period_days),
        "period_err": 0.0,
        "duration_hours": duration_fraction * period_days * 24,
        "duration_err": None if not np.isfinite(errs[2]) else float(errs[2] * period_days * 24),
        "depth_ppt": depth * 1000,
        "depth_err": None if not np.isfinite(errs[1]) else float(errs[1] * 1000),
        "t0": t0,
        "t0_err": _clean_err(errs[0]),
        "chi_squared": chi_squared,
        "model_flux": model,
    }
