import numpy as np
import pytest

from app.ml.autoencoder import denoise
from app.services.fitter import fit_transit_parameters
from app.services.pipeline import phase_fold
from app.services.preprocess import _clean_arrays


def test_phase_fold_centered_on_t0():
    time = np.linspace(0, 10, 1000)
    flux = np.ones_like(time)
    phase, folded = phase_fold(time, flux, period=2.0, t0=1.0, bins=128)
    assert phase.min() >= -0.5
    assert phase.max() <= 0.5
    assert len(folded) == 128


def test_fitter_errors_are_none_not_nan():
    phase = np.linspace(-0.5, 0.5, 128)
    flux = np.ones_like(phase)
    result = fit_transit_parameters(phase, flux, 2.0)
    for key in ("duration_err", "depth_err", "t0_err"):
        assert result[key] is None or np.isfinite(result[key])


def test_autoencoder_denoise_same_length():
    flux = np.random.normal(0, 0.001, 777).astype(np.float32)
    assert len(denoise(flux)) == len(flux)


def test_all_nan_flux_fails_gracefully():
    time = np.arange(200, dtype=float)
    flux = np.full(200, np.nan)
    with pytest.raises(Exception):
        _clean_arrays(time, flux)
