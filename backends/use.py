"""USE backend: Universal Sentence Encoder v4 via TensorFlow Hub."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from backends.base import Backend

USE_MODEL_URL: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
USE_LOW: float = 0.50
USE_HIGH: float = 0.75
_MISSING_DEPS_MSG: str = (
    "USE backend requires the 'tensorflow' and 'tensorflow-hub' extras. "
    "Install them with: pip install -r requirements-use.txt"
)


def _load_use_model():
    try:
        import tensorflow_hub as hub
    except ImportError as exc:
        raise ImportError(_MISSING_DEPS_MSG) from exc
    return hub.load(USE_MODEL_URL)


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


class UseBackend(Backend):
    """Cosine similarity over Universal Sentence Encoder v4 embeddings."""

    _model = None

    def __init__(self, corpus: Sequence[str]) -> None:
        super().__init__(corpus)
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._model = _load_use_model()
        return cls._model

    def _embed(self, phrase: str) -> np.ndarray:
        cached = self._cache.get(phrase)
        if cached is not None:
            return cached
        model = self._get_model()
        vec = np.asarray(model([phrase]))[0]
        self._cache[phrase] = vec
        return vec

    def score_pair(self, phrase_a: str, phrase_b: str) -> float:
        return clamp_unit(cosine(self._embed(phrase_a), self._embed(phrase_b)))

    @property
    def thresholds(self) -> tuple[float, float]:
        return USE_LOW, USE_HIGH
