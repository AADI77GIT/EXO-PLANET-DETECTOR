import numpy as np
from astropy.timeseries import BoxLeastSquares

from app.config import settings


def _fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    return ((time - t0 + 0.5 * period) % period) / period - 0.5


def _transit_depth(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration_days: float) -> float:
    phase_days = _fold(time, period, t0) * period
    in_transit = np.abs(phase_days) <= duration_days / 2
    out = ~in_transit
    if not np.any(in_transit) or not np.any(out):
        return 0.0
    return float(np.nanmedian(flux[out]) - np.nanmedian(flux[in_transit]))


def _odd_even_diff(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration_days: float) -> float:
    epochs = np.floor((time - t0) / period).astype(int)
    phase_days = _fold(time, period, t0) * period
    in_transit = np.abs(phase_days) <= duration_days / 2
    depths = []
    for parity in (0, 1):
        mask = in_transit & ((epochs % 2) == parity)
        out = (~in_transit) & ((epochs % 2) == parity)
        if np.any(mask) and np.any(out):
            depths.append(float(np.nanmedian(flux[out]) - np.nanmedian(flux[mask])))
    if len(depths) != 2:
        return 0.0
    return abs(depths[0] - depths[1])


def run_bls(time, flux_denoised) -> dict:
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux_denoised, dtype=np.float64)
    if time.size != flux.size or time.size < 128:
        raise ValueError("BLS input arrays must have matching length >= 128")
    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(settings.bls_min_period_days, settings.bls_max_period_days, settings.bls_period_samples)
    durations = np.array([0.05, 0.1, 0.15, 0.2])
    power = bls.power(periods, durations)
    best = int(np.nanargmax(power.power))
    period = float(power.period[best])
    t0 = float(power.transit_time[best])
    duration = float(power.duration[best])
    stats = bls.compute_stats(period, duration, t0)
    raw_depth = stats.get("depth", power.depth[best])
    if isinstance(raw_depth, tuple):
        raw_depth = raw_depth[0]
    depth = abs(float(raw_depth))
    if not np.isfinite(depth) or depth == 0:
        depth = abs(_transit_depth(time, flux, period, t0, duration))

    secondary_depth = abs(_transit_depth(time, flux, period / 2, t0, duration))
    odd_even = _odd_even_diff(time, flux, period, t0, duration)
    return {
        "period": period,
        "t0": t0,
        "duration_hours": duration * 24,
        "duration_days": duration,
        "depth_ppt": depth * 1000,
        "secondary_eclipse_flag": bool(secondary_depth > 0.5 * depth),
        "odd_even_diff": float(odd_even * 1000),
        "power_spectrum": {"period": power.period.astype(float).tolist(), "power": power.power.astype(float).tolist()},
        "period_grid_range": [settings.bls_min_period_days, settings.bls_max_period_days, settings.bls_period_samples],
    }
