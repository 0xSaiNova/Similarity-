"""Slow integration tests: actually load USE v4 from TF Hub. Skipped without --runslow."""
from __future__ import annotations

import pytest


def _require_tf_extras() -> None:
    try:
        import tensorflow  # noqa: F401
        import tensorflow_hub  # noqa: F401
    except ImportError:
        pytest.skip("tensorflow extras not installed; install with requirements-use.txt")


@pytest.mark.slow
def test_use_identical_phrases_score_near_one() -> None:
    _require_tf_extras()
    from backends.use import UseBackend
    backend = UseBackend(["hello world"])
    score = backend.score_pair("hello world", "hello world")
    assert score > 0.99


@pytest.mark.slow
def test_use_unrelated_phrases_score_low() -> None:
    _require_tf_extras()
    from backends.use import UseBackend
    backend = UseBackend(["seed"])
    score = backend.score_pair(
        "the cat sat on the mat",
        "quantum cryptography research uses lattice problems",
    )
    assert score < 0.5
