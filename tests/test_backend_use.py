"""Fast unit tests for backends/use.py: cosine, clamp, and score_pair with a mocked encoder."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from backends import available, get_backend
from backends.use import USE_HIGH, USE_LOW, UseBackend, clamp_unit, cosine


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


def test_thresholds_are_use_constants() -> None:
    backend = UseBackend(["x"])
    assert backend.thresholds == (USE_LOW, USE_HIGH)


def test_use_registered_in_registry() -> None:
    assert "use" in available()


def test_get_backend_use_does_not_call_model_loader(monkeypatch) -> None:
    from backends import use as use_module

    def _explode() -> None:
        raise AssertionError("model loader called during construction")

    monkeypatch.setattr(use_module, "_load_use_model", _explode)
    backend = get_backend("use", ["x"])
    assert isinstance(backend, UseBackend)


def test_importing_backends_package_does_not_import_tensorflow() -> None:
    assert "tensorflow" not in sys.modules
    assert "tensorflow_hub" not in sys.modules


class _MockEncoder:
    """Callable that returns fixed embeddings per phrase, mimicking a TF Hub module."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def __call__(self, phrases):
        return np.stack([self._vectors[p] for p in phrases])


@pytest.fixture
def mocked(monkeypatch) -> UseBackend:
    vectors = {
        "alpha": np.array([1.0, 0.0, 0.0]),
        "beta": np.array([0.0, 1.0, 0.0]),
        "alpha_clone": np.array([1.0, 0.0, 0.0]),
        "anti": np.array([-1.0, 0.0, 0.0]),
    }
    from backends import use as use_module
    encoder = _MockEncoder(vectors)
    monkeypatch.setattr(use_module, "_load_use_model", lambda: encoder)
    return UseBackend(list(vectors.keys()))


def test_score_pair_identical_phrases_is_one(mocked: UseBackend) -> None:
    assert mocked.score_pair("alpha", "alpha") == pytest.approx(1.0)


def test_score_pair_same_embedding_different_phrase_is_one(mocked: UseBackend) -> None:
    assert mocked.score_pair("alpha", "alpha_clone") == pytest.approx(1.0)


def test_score_pair_orthogonal_phrases_is_zero(mocked: UseBackend) -> None:
    assert mocked.score_pair("alpha", "beta") == pytest.approx(0.0)


def test_score_pair_clamps_negative_cosine_to_zero(mocked: UseBackend) -> None:
    assert mocked.score_pair("alpha", "anti") == 0.0


def test_score_pair_caches_embeddings(mocked: UseBackend) -> None:
    mocked.score_pair("alpha", "beta")
    assert "alpha" in mocked._cache
    assert "beta" in mocked._cache


def test_score_with_explain_returns_score_and_none(mocked: UseBackend) -> None:
    score, signals = mocked.score_with_explain("alpha", "alpha")
    assert score == pytest.approx(1.0)
    assert signals is None


def test_load_use_model_raises_when_tf_hub_missing(monkeypatch) -> None:
    from backends import use as use_module

    use_module._load_use_model.cache_clear()
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow_hub":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="USE backend requires"):
        use_module._load_use_model()
    use_module._load_use_model.cache_clear()
