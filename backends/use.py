"""USE backend: Universal Sentence Encoder v4 via TensorFlow Hub."""
from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from backends.base import Backend
from backends.embedding_utils import clamp_unit, cosine
from combiner import load_backend_thresholds

USE_MODEL_URL: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
USE_LOW: float = 0.50
USE_HIGH: float = 0.75
DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config.json"
_MISSING_DEPS_MSG: str = (
    "USE backend requires the 'tensorflow' and 'tensorflow-hub' extras. "
    "Install with: pip install -r requirements-use.txt (Python 3.10 to 3.13; "
    "no TensorFlow wheel for 3.14+ yet)."
)


@functools.lru_cache(maxsize=1)
def _load_use_model():
    try:
        import tensorflow_hub as hub
    except ImportError as exc:
        raise ImportError(_MISSING_DEPS_MSG) from exc
    return hub.load(USE_MODEL_URL)


class UseBackend(Backend):
    """Cosine similarity over Universal Sentence Encoder v4 embeddings."""

    def __init__(
        self, corpus: Sequence[str], config_path: str | Path | None = None,
    ) -> None:
        super().__init__(corpus)
        self._cache: dict[str, np.ndarray] = {}
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        self._thresholds = load_backend_thresholds("use", config_path, (USE_LOW, USE_HIGH))

    def _embed(self, phrase: str) -> np.ndarray:
        cached = self._cache.get(phrase)
        if cached is not None:
            return cached
        model = _load_use_model()
        vec = np.asarray(model([phrase]))[0]
        self._cache[phrase] = vec
        return vec

    def score_pair(self, phrase_a: str, phrase_b: str) -> float:
        return clamp_unit(cosine(self._embed(phrase_a), self._embed(phrase_b)))

    @property
    def thresholds(self) -> tuple[float, float]:
        return self._thresholds
