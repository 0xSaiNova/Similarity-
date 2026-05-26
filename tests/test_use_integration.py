"""Slow integration tests: actually load USE v4 from TF Hub and hit OpenRouter for GPT."""
from __future__ import annotations

import os

import pytest

SIMILAR = ("the cat sat on the mat", "a cat is on a mat")
UNRELATED = (
    "the cat sat on the mat",
    "quantum cryptography research uses lattice problems",
)


def _require_tf_extras() -> None:
    try:
        import tensorflow  # noqa: F401
        import tensorflow_hub  # noqa: F401
    except ImportError:
        pytest.skip("tensorflow extras not installed; install with requirements-use.txt")


def _require_gpt_extras() -> None:
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai extra not installed; install with requirements-gpt.txt")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; GPT backend regression test needs a real key")


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
    score = backend.score_pair(*UNRELATED)
    assert score < 0.5


@pytest.mark.slow
def test_use_similar_pair_scores_clearly_above_unrelated() -> None:
    _require_tf_extras()
    from backends.use import UseBackend
    backend = UseBackend(["seed"])
    similar = backend.score_pair(*SIMILAR)
    unrelated = backend.score_pair(*UNRELATED)
    assert similar - unrelated > 0.3, (
        f"USE failed to separate similar from unrelated: similar={similar:.3f}, "
        f"unrelated={unrelated:.3f}"
    )


@pytest.mark.slow
def test_gpt_similar_pair_scores_clearly_above_unrelated() -> None:
    _require_gpt_extras()
    from backends.gpt import GptBackend
    backend = GptBackend(["seed"])
    similar = backend.score_pair(*SIMILAR)
    unrelated = backend.score_pair(*UNRELATED)
    assert similar - unrelated > 0.3, (
        f"GPT failed to separate similar from unrelated: similar={similar:.3f}, "
        f"unrelated={unrelated:.3f}"
    )
