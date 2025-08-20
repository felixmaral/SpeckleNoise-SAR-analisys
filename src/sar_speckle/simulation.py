"""Speckle simulation utilities.
Variable names and comments in English.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
import argparse
from pathlib import Path
from PIL import Image

def _rng_from_seed(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def gamma_speckle_field(shape: tuple[int, int], L: float, rng: np.random.Generator | int | None = None,
                        correlate_sigma: float | None = None) -> np.ndarray:
    """Generate a multiplicative speckle field ~ Gamma(L, 1/L) with E[n]=1, Var[n]=1/L.

    Args:
        shape: (H, W)
        L: equivalent number of looks (ENL). Can be non-integer.
        rng: numpy Generator or seed.
        correlate_sigma: if provided, apply Gaussian blur to induce spatial correlation
                         (sigma in pixels, applied to log-field to preserve mean).
    Returns:
        float32 array with mean ~1.
    """
    if L <= 0:
        raise ValueError("L must be > 0")
    g = _rng_from_seed(rng).gamma(shape=L, scale=1.0 / L, size=shape).astype(np.float32)
    if correlate_sigma and correlate_sigma > 0:
        # Work in log domain to keep multiplicative structure and mean ≈ 1 after exponentiation
        logg = np.log(np.maximum(g, 1e-12))
        logg = gaussian_filter(logg, sigma=correlate_sigma, mode="reflect")
        g = np.exp(logg).astype(np.float32)
        # Re-normalize mean to 1 (small bias may appear after blur)
        g /= max(g.mean(), 1e-12)
    return g


def apply_speckle_intensity(I: np.ndarray, L: float, rng: np.random.Generator | int | None = None,
                            correlate_sigma: float | None = None) -> np.ndarray:
    """Apply multiplicative speckle to an intensity image.

    I_obs = I * n,  n ~ Gamma(L, 1/L)
    """
    if I.ndim != 2:
        raise ValueError("Intensity image must be 2D")
    n = gamma_speckle_field(I.shape, L, rng=rng, correlate_sigma=correlate_sigma)
    return (I.astype(np.float32) * n).astype(np.float32)


def coherent_multilook_intensity(I: np.ndarray, L: int, rng: np.random.Generator | int | None = None,
                                 correlate_sigma: float | None = None) -> np.ndarray:
    """Coherent simulation: generate L complex looks CN(0, I) and multilook intensity.

    If correlate_sigma is set, a Gaussian blur is applied to the *complex* field magnitude
    (via blur in real/imag) before forming intensity, approximating a PSF.
    """
    if I.ndim != 2:
        raise ValueError("Intensity image must be 2D")
    if L < 1:
        raise ValueError("L must be >= 1 for coherent model")
    rng = _rng_from_seed(rng)
    H, W = I.shape
    # Draw complex Gaussian looks with variance I per pixel
    sigma = np.sqrt(np.maximum(I, 0) / 2.0).astype(np.float32)
    re = rng.normal(loc=0.0, scale=sigma, size=(L, H, W)).astype(np.float32)
    im = rng.normal(loc=0.0, scale=sigma, size=(L, H, W)).astype(np.float32)
    if correlate_sigma and correlate_sigma > 0:
        for l in range(L):
            re[l] = gaussian_filter(re[l], sigma=correlate_sigma, mode="reflect")
            im[l] = gaussian_filter(im[l], sigma=correlate_sigma, mode="reflect")
    I_ml = (re * re + im * im).mean(axis=0)
    return I_ml.astype(np.float32)

def simulate_speckle_image(shape=(256, 256), gray_level=0.5, L=3.0, rng=None, correlate_sigma=None):
    I = np.full(shape, float(gray_level), dtype=np.float32)
    return apply_speckle_intensity(I, L=L, rng=rng, correlate_sigma=correlate_sigma)

def simulate_main():
    import argparse
    from pathlib import Path
    from PIL import Image

    p = argparse.ArgumentParser(description="Generate single synthetic speckled image")
    p.add_argument("--L", type=float, default=3.0, help="Equivalent number of looks (ENL)")
    p.add_argument("--gray-level", type=float, default=0.5, help="Uniform background [0-1]")
    p.add_argument("--size", type=int, default=256, help="Square size of the image")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    args = p.parse_args()

    img = simulate_speckle_image(
        shape=(args.size, args.size),
        gray_level=args.gray_level,
        L=args.L,
        rng=args.seed,
    )

    outdir = Path("outputs/single_sim")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"sim_gray{args.gray_level}_L{args.L}.png"

    arr = (np.clip(img, 0, 1) * 255).astype("uint8")
    Image.fromarray(arr).save(outpath)
    print(f"Saved simulated image to {outpath}")