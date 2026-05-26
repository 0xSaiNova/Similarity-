"""Tests for the shared cosine + clamp helpers used by every embedding backend."""
from __future__ import annotations

import numpy as np
import pytest

from backends.embedding_utils import clamp_unit, cosine


def test_cosine_orthogonal_returns_zero() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine(a, b) == pytest.approx(0.0)


def test_cosine_identical_vectors_returns_one() -> None:
    v = np.array([0.3, 0.4, 0.5])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_opposite_vectors_returns_negative_one() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, -v) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero() -> None:
    a = np.zeros(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    assert cosine(a, b) == 0.0
    assert cosine(b, a) == 0.0


def test_clamp_unit_below_zero() -> None:
    assert clamp_unit(-0.4) == 0.0


def test_clamp_unit_above_one() -> None:
    assert clamp_unit(1.5) == 1.0


def test_clamp_unit_in_range_passthrough() -> None:
    assert clamp_unit(0.7) == 0.7
