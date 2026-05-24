"""TF-IDF index over a candidate set + blocking via top-K cosine."""
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
        self._cand_norms = np.sqrt(self._matrix.multiply(self._matrix).sum(axis=1)).A1

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

    def top_k(self, query_text: str, k: int) -> list[str]:
        """Return the k candidates with the highest TF-IDF cosine to query_text."""
        if k <= 0 or not self.candidates:
            return []
        query_vec = self._vector(query_text)
        query_norm = np.sqrt(query_vec.multiply(query_vec).sum())
        if query_norm == 0.0:
            return []
        scores = self._matrix.dot(query_vec.T).toarray().ravel()
        denom = self._cand_norms * query_norm
        with np.errstate(divide="ignore", invalid="ignore"):
            cosines = np.where(denom > 0, scores / denom, 0.0)
        k = min(k, len(self.candidates))
        idx = np.argpartition(-cosines, k - 1)[:k]
        idx = idx[np.argsort(-cosines[idx])]
        return [self.candidates[i] for i in idx]
