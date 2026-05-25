"""Fast unit tests for backends/gpt.py: cosine, clamp, cache, errors with a mocked OpenAI client."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from backends import available, get_backend
from backends.gpt import GPT_DEFAULT_MODEL, GPT_HIGH, GPT_LOW, GptBackend


class _FakeDatum:
    """Mimics openai's embedding response item shape: object with `.embedding`."""

    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeResponse:
    """Mimics openai's response shape: object with `.data` list of items."""

    def __init__(self, embedding: list[float]) -> None:
        self.data = [_FakeDatum(embedding)]


class _FakeEmbeddings:
    """Records calls and returns embeddings from a fixed phrase->vector map."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.call_count = 0
        self.calls: list[tuple[str, str]] = []

    def create(self, model: str, input: str) -> _FakeResponse:
        self.call_count += 1
        self.calls.append((model, input))
        return _FakeResponse(self.vectors[input])


class _FakeClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.embeddings = _FakeEmbeddings(vectors)


def _vectors() -> dict[str, list[float]]:
    return {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "alpha_clone": [1.0, 0.0, 0.0],
        "anti": [-1.0, 0.0, 0.0],
    }


@pytest.fixture
def fake_client(monkeypatch) -> _FakeClient:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = _FakeClient(_vectors())
    from backends import gpt as gpt_module
    monkeypatch.setattr(gpt_module, "_load_client", lambda: client)
    return client


@pytest.fixture
def backend(fake_client: _FakeClient, tmp_path: Path) -> GptBackend:
    return GptBackend(
        corpus=list(_vectors().keys()),
        cache_path=tmp_path / "cache.json",
    )


def test_score_pair_identical_phrases_is_one(backend: GptBackend) -> None:
    assert backend.score_pair("alpha", "alpha") == pytest.approx(1.0)


def test_score_pair_same_embedding_different_phrase_is_one(backend: GptBackend) -> None:
    assert backend.score_pair("alpha", "alpha_clone") == pytest.approx(1.0)


def test_score_pair_orthogonal_phrases_is_zero(backend: GptBackend) -> None:
    assert backend.score_pair("alpha", "beta") == pytest.approx(0.0)


def test_score_pair_clamps_negative_cosine_to_zero(backend: GptBackend) -> None:
    assert backend.score_pair("alpha", "anti") == 0.0


def test_score_pair_caches_in_memory_no_repeat_api(
    backend: GptBackend, fake_client: _FakeClient,
) -> None:
    backend.score_pair("alpha", "beta")
    first = fake_client.embeddings.call_count
    backend.score_pair("alpha", "beta")
    assert fake_client.embeddings.call_count == first


def test_cache_persists_to_disk(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = _FakeClient(_vectors())
    from backends import gpt as gpt_module
    monkeypatch.setattr(gpt_module, "_load_client", lambda: client)
    cache_path = tmp_path / "cache.json"
    backend = GptBackend(corpus=["alpha", "beta"], cache_path=cache_path)
    backend.score_pair("alpha", "beta")
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert "alpha" in payload
    assert "beta" in payload


def test_cache_skips_api_when_phrase_in_disk_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = _FakeClient(_vectors())
    from backends import gpt as gpt_module
    monkeypatch.setattr(gpt_module, "_load_client", lambda: client)
    cache_path = tmp_path / "cache.json"

    first = GptBackend(corpus=["alpha", "beta"], cache_path=cache_path)
    first.score_pair("alpha", "beta")
    primed_calls = client.embeddings.call_count
    assert primed_calls == 2

    second = GptBackend(corpus=["alpha", "beta"], cache_path=cache_path)
    second.score_pair("alpha", "beta")
    assert client.embeddings.call_count == primed_calls


def test_load_client_raises_when_key_missing(monkeypatch) -> None:
    from backends import gpt as gpt_module
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        gpt_module._load_client()


def test_load_client_raises_when_openai_missing(monkeypatch) -> None:
    from backends import gpt as gpt_module
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="GPT backend requires"):
        gpt_module._load_client()


def test_gpt_registered_in_registry() -> None:
    assert "gpt" in available()


def test_get_backend_gpt_does_not_call_client_loader(monkeypatch, tmp_path: Path) -> None:
    from backends import gpt as gpt_module

    def explode() -> None:
        raise AssertionError("client loader called during construction")

    monkeypatch.setattr(gpt_module, "_load_client", explode)
    monkeypatch.setattr(
        gpt_module, "_default_cache_path", lambda model: tmp_path / f"{model}.json",
    )
    backend = get_backend("gpt", ["x"])
    assert isinstance(backend, GptBackend)


def test_score_with_explain_returns_score_and_none(backend: GptBackend) -> None:
    score, signals = backend.score_with_explain("alpha", "alpha")
    assert score == pytest.approx(1.0)
    assert signals is None


def test_importing_backends_package_does_not_import_openai() -> None:
    assert "openai" not in sys.modules


def test_thresholds_are_gpt_constants(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    b = GptBackend(corpus=["x"], cache_path=tmp_path / "c.json")
    assert b.thresholds == (GPT_LOW, GPT_HIGH)


def test_default_model_is_text_embedding_3_small(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    b = GptBackend(corpus=["x"], cache_path=tmp_path / "c.json")
    assert b._model == "text-embedding-3-small"
    assert GPT_DEFAULT_MODEL == "text-embedding-3-small"


def test_custom_model_passed_to_api(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = _FakeClient(_vectors())
    from backends import gpt as gpt_module
    monkeypatch.setattr(gpt_module, "_load_client", lambda: client)
    b = GptBackend(
        corpus=["alpha"], model="text-embedding-3-large", cache_path=tmp_path / "c.json",
    )
    b.score_pair("alpha", "alpha")
    assert client.embeddings.calls[0][0] == "text-embedding-3-large"
