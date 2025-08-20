"""Simple quality/statistics metrics."""
from __future__ import annotations
import numpy as np


def estimate_enl(region: np.ndarray) -> float:
    """Estimate ENL (L) from a homogeneous region using L ≈ (mean^2)/var.
    Use on multiplicative noise field or on intensity in a flat area.
    """
    x = region.astype(np.float64)
    m = x.mean()
    v = x.var(ddof=1)
    if v <= 0:
        return float("inf")
    return float((m * m) / v)