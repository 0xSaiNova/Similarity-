"""Shared cosine + unit clamp helpers used by every embedding backend."""
from __future__ import annotations

import numpy as np


def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity of two 1D vectors; 0.0 if either has zero norm."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def clamp_unit(value: float) -> float:
    """Clamp a scalar into [0.0, 1.0]."""
    return max(0.0, min(1.0, value))
