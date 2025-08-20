"""Classical despeckling filters for SAR (Lee, Kuan, Frost).

All functions assume a **2D intensity image** (float32/float64, non-negative) and an
**Equivalent Number of Looks (ENL)** provided by the user (or estimated elsewhere).

References (high-level):
- Lee, J.-S. "Digital image enhancement and noise filtering by use of local statistics." IEEE TPAMI, 1980.
- Kuan, D. T. et al. "Adaptive noise smoothing filter for images with signal-dependent noise." IEEE TPAMI, 1985.
- Frost, V. S. et al. "A model for radar images and its application to adaptive digital filtering of multiplicative noise." IEEE T-PAMI, 1982.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, generic_filter


def _check_image(I: np.ndarray) -> np.ndarray:
    if I.ndim != 2:
        raise ValueError("Input must be a 2D intensity image")
    if not np.issubdtype(I.dtype, np.floating):
        I = I.astype(np.float32)
    return I


def _local_stats(I: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute local mean and variance via uniform box filter (O(1) per pixel).
    """
    if size % 2 == 0 or size < 3:
        raise ValueError("size must be odd and >= 3")
    I = I.astype(np.float32)
    mean = uniform_filter(I, size=size, mode="reflect")
    mean_sq = uniform_filter(I * I, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, var


def lee_filter(I: np.ndarray, size: int = 7, enl: float = 1.0) -> np.ndarray:
    """Lee filter (adaptive local mean).

    Formula (intensity-domain):
        Cx^2 = var / (mean^2)               # local coefficient of variation^2
        Cu^2 = 1 / enl                      # speckle variance (Gamma model)
        W = max(0, (Cx^2 - Cu^2) / (Cx^2 * (1 + Cu^2)))
        I_hat = mean + W * (I - mean)
    """
    I = _check_image(I)
    mean, var = _local_stats(I, size)
    eps = 1e-12
    Cu2 = 1.0 / max(enl, eps)
    mean2 = np.maximum(mean * mean, eps)
    Cx2 = var / mean2
    W = (Cx2 - Cu2) / np.maximum(Cx2 * (1.0 + Cu2), eps)
    W = np.clip(W, 0.0, 1.0)
    out = mean + W * (I - mean)
    return np.clip(out, 0, None)


def kuan_filter(I: np.ndarray, size: int = 7, enl: float = 1.0) -> np.ndarray:
    """Kuan filter (linear MMSE under multiplicative noise).

    In intensity domain, the Kuan weight simplifies to:
        Cx^2 = var / mean^2
        Cu^2 = 1 / enl
        W = (1 - Cu^2 / (1 + Cx^2))
        I_hat = mean + W * (I - mean)
    """
    I = _check_image(I)
    mean, var = _local_stats(I, size)
    eps = 1e-12
    Cu2 = 1.0 / max(enl, eps)
    Cx2 = var / np.maximum(mean * mean, eps)
    W = 1.0 - (Cu2 / np.maximum(1.0 + Cx2, eps))
    W = np.clip(W, 0.0, 1.0)
    out = mean + W * (I - mean)
    return np.clip(out, 0, None)


def frost_filter(I: np.ndarray, size: int = 7, enl: float = 1.0, damping: float = 1.0) -> np.ndarray:
    """Frost filter (exponential kernel with variance-adaptive damping).

    This implementation uses a per-window exponential weighting:
        k = max(0, (Cv^2 - Cu^2) / (Cv^2 * (1 + Cu^2)))  # similar to Lee's gain
        alpha = damping * k
        w(d) = exp(-alpha * d)  with d = city-block distance to window center
        I_hat = sum(w * I_win) / sum(w)

    Notes:
    - This is a practical isotropic version using generic_filter; slower than Lee/Kuan
      but fine for research-sized images. Increase performance later if needed.
    - enl controls Cu^2 = 1/enl.
    - damping (>0) increases smoothing in highly variable areas.
    """
    I = _check_image(I)
    if size % 2 == 0 or size < 3:
        raise ValueError("size must be odd and >= 3")

    half = size // 2
    # Precompute distance map (city-block / Manhattan distance)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    D = np.abs(yy) + np.abs(xx)

    # Local statistics for gain (reuse Lee-like gain)
    mean, var = _local_stats(I, size)
    Cu2 = 1.0 / max(enl, 1e-12)
    Cx2 = var / np.maximum(mean * mean, 1e-12)
    k = np.clip((Cx2 - Cu2) / np.maximum(Cx2 * (1.0 + Cu2), 1e-12), 0.0, 1.0)
    alpha = damping * k

    # We need per-pixel alpha; implement using generic_filter that receives flattened window
    footprint = np.ones((size, size), dtype=bool)

    def _frost_func(window: np.ndarray) -> float:
        # window comes flattened, center value is at position len//2
        win = window.reshape((size, size))
        # alpha at the window center (precomputed outside)
        # We'll inject the center alpha by closure via nonlocal lookup later
        raise RuntimeError("Internal placeholder; replaced below")

    # Trick: we pass a lambda that closes over a view of alpha for each block via mode='reflect'.
    # generic_filter doesn't pass coordinates, so we build a small wrapper that updates a global
    # pointer through a mutable object.
    class _AlphaCursor:
        def __init__(self, A: np.ndarray):
            self.A = A
            self.pos = [0, 0]

    ac = _AlphaCursor(alpha)

    def _walker(window: np.ndarray) -> float:
        win = window.reshape((size, size))
        a = ac.A[ac.pos[0], ac.pos[1]]
        w = np.exp(-a * D)
        num = float((w * win).sum())
        den = float(w.sum())
        return num / max(den, 1e-12)

    # We emulate a moving cursor by iterating rows; but generic_filter doesn't expose indices.
    # Workaround: apply per-row using a loop updating the cursor row index and slicing.
    out = np.empty_like(I, dtype=np.float32)
    # Pad image reflectively to extract valid windows per-row
    pad = half
    Ip = np.pad(I, pad_width=pad, mode="reflect")
    Ap = np.pad(alpha, pad_width=pad, mode="reflect")

    for r in range(I.shape[0]):
        ac.pos[0] = r + pad
        # Compute the whole row by sliding window on the padded arrays
        # For each column we need to update ac.pos[1]; to avoid inner Python loops,
        # we call generic_filter on a one-row slice and update pos inside a small loop.
        row_slice = Ip[r : r + size + (size - 1), :]  # extra rows to allow footprint
        # Fallback: per-column loop (still acceptable for prototyping)
        for c in range(I.shape[1]):
            ac.pos[1] = c + pad
            win = Ip[r : r + size, c : c + size]
            a = Ap[r + pad, c + pad]
            w = np.exp(-a * D)
            out[r, c] = (w * win).sum() / max(w.sum(), 1e-12)

    return np.clip(out, 0, None)