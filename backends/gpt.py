"""GPT backend: OpenAI text-embedding cosine similarity with persistent disk cache."""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backends.base import Backend
from backends.use import clamp_unit, cosine

GPT_DEFAULT_MODEL: str = "text-embedding-3-small"
GPT_LOW: float = 0.65
GPT_HIGH: float = 0.85
_DEFAULT_CACHE_DIR: Path = Path(".cache")
_MISSING_KEY_MSG: str = (
    "GPT backend requires the OPENAI_API_KEY environment variable. "
    "Set it before constructing GptBackend, e.g. export OPENAI_API_KEY=sk_..."
)
_MISSING_DEP_MSG: str = (
    "GPT backend requires the 'openai' extra. "
    "Install with: pip install -r requirements-gpt.txt"
)


def _load_client() -> Any:
    """Lazy import openai, return an authenticated client. Raises if key or library missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(_MISSING_KEY_MSG)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(_MISSING_DEP_MSG) from exc
    return OpenAI()


def _default_cache_path(model: str) -> Path:
    """Per model cache file under .cache/ in the working directory."""
    return _DEFAULT_CACHE_DIR / f"gpt_embeddings_{model}.json"


class GptBackend(Backend):
    """Cosine similarity over OpenAI text embedding vectors with a persistent on disk cache."""

    def __init__(
        self,
        corpus: Sequence[str],
        model: str = GPT_DEFAULT_MODEL,
        cache_path: Path | str | None = None,
    ) -> None:
        super().__init__(corpus)
        self._model: str = model
        self._cache_path: Path = (
            Path(cache_path) if cache_path is not None else _default_cache_path(model)
        )
        self._cache: dict[str, np.ndarray] = self._load_cache()
        self._client: Any = None

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self._cache_path.exists():
            return {}
        raw = json.loads(self._cache_path.read_text())
        return {phrase: np.asarray(vec, dtype=np.float64) for phrase, vec in raw.items()}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {phrase: vec.tolist() for phrase, vec in self._cache.items()}
        self._cache_path.write_text(json.dumps(payload))

    def _embed(self, phrase: str) -> np.ndarray:
        cached = self._cache.get(phrase)
        if cached is not None:
            return cached
        if self._client is None:
            self._client = _load_client()
        resp = self._client.embeddings.create(model=self._model, input=phrase)
        vec = np.asarray(resp.data[0].embedding, dtype=np.float64)
        self._cache[phrase] = vec
        self._save_cache()
        return vec

    def score_pair(self, phrase_a: str, phrase_b: str) -> float:
        return clamp_unit(cosine(self._embed(phrase_a), self._embed(phrase_b)))

    @property
    def thresholds(self) -> tuple[float, float]:
        return GPT_LOW, GPT_HIGH
