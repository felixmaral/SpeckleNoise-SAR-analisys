import numpy as np
from sar_speckle.simulation import apply_speckle_intensity
from sar_speckle.filters import lee_filter, kuan_filter, frost_filter
from sar_speckle.metrics import estimate_enl


def _noisy_patch(enl=3.0, seed=0):
    rng = np.random.default_rng(seed)
    I = np.ones((128, 128), np.float32)
    I_noisy = apply_speckle_intensity(I, L=enl, rng=rng)
    return I_noisy


def test_lee_reduces_variance():
    x = _noisy_patch(enl=3.0, seed=0)
    var_before = x.var()
    y = lee_filter(x, size=7, enl=3.0)
    var_after = y.var()
    assert var_after < var_before


def test_kuan_reduces_variance():
    x = _noisy_patch(enl=3.0, seed=1)
    var_before = x.var()
    y = kuan_filter(x, size=7, enl=3.0)
    var_after = y.var()
    assert var_after < var_before


def test_frost_runs_and_reduces_variance():
    x = _noisy_patch(enl=3.0, seed=2)
    var_before = x.var()
    y = frost_filter(x, size=5, enl=3.0, damping=1.0)
    var_after = y.var()
    assert var_after < var_before
