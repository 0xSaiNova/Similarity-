"""TF-IDF index over a candidate set; exposes pairwise cosine."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import normalize


def _tokenize(text: str) -> list[str]:
    return normalize(text)


class PhraseIndex:
    """TF-IDF index over a fixed candidate set."""

    def __init__(self, candidates: Sequence[str]) -> None:
        self.candidates: tuple[str, ...] = tuple(candidates)
        if not self.candidates:
            raise ValueError("PhraseIndex requires at least one candidate")
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            lowercase=False,
            token_pattern=None,
        )
        self._matrix = self._vectorizer.fit_transform(self.candidates)

    def _vector(self, text: str):
        return self._vectorizer.transform([text])

    def tfidf_cosine(self, text_a: str, text_b: str) -> float:
        """Cosine similarity of TF-IDF vectors for two raw texts."""
        vec_a = self._vector(text_a)
        vec_b = self._vector(text_b)
        norm_a = np.sqrt(vec_a.multiply(vec_a).sum())
        norm_b = np.sqrt(vec_b.multiply(vec_b).sum())
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        dot = vec_a.multiply(vec_b).sum()
        return float(dot / (norm_a * norm_b))
