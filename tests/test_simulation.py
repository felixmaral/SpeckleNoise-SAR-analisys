import numpy as np
from sar_speckle.simulation import apply_speckle_intensity, gamma_speckle_field
from sar_speckle.metrics import estimate_enl


def test_gamma_field_mean_var():
    L = 4.0
    n = gamma_speckle_field((512, 512), L, rng=0)
    assert 0.98 < n.mean() < 1.02
    assert 0.22 < n.var() < 0.28  # around 1/L = 0.25


def test_apply_speckle_enl_estimation():
    I = np.ones((256, 256), np.float32)
    L = 3.0
    I_noisy = apply_speckle_intensity(I, L, rng=0)
    enl = estimate_enl(I_noisy[:128, :128])
    assert 2.0 < enl < 4.5