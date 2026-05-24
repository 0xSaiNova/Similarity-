"""Classical backend: thin wrapper around the existing WordNet ensemble engine."""
from __future__ import annotations

from collections.abc import Sequence

from backends.base import Backend
from combiner import DEFAULT_PENALTIES, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, _label_for, combine
from index import PhraseIndex
from matcher import compute_signals
from preprocess import build_phrase, detect_antonym_mismatch
from signals import detect_order_mismatch


class ClassicalBackend(Backend):
    """Wraps the existing engine: TF-IDF index + 6 signals + 3 gates + combine."""

    def __init__(self, corpus: Sequence[str]) -> None:
        super().__init__(corpus)
        self._index = PhraseIndex(self._corpus)

    def _combine_for(self, phrase_a: str, phrase_b: str) -> tuple[float, str]:
        a = build_phrase(phrase_a)
        b = build_phrase(phrase_b)
        signals = compute_signals(phrase_a, phrase_b, self._index, a, b)
        return combine(
            signals,
            a.has_negation != b.has_negation,
            detect_antonym_mismatch(phrase_a, phrase_b),
            detect_order_mismatch(a.tokens, b.tokens),
            DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS, DEFAULT_PENALTIES,
        )

    def score_pair(self, phrase_a: str, phrase_b: str) -> float:
        score, _ = self._combine_for(phrase_a, phrase_b)
        return score

    @property
    def thresholds(self) -> tuple[float, float]:
        return DEFAULT_THRESHOLDS["low"], DEFAULT_THRESHOLDS["high"]

    def label(self, score: float) -> str:
        return _label_for(score, DEFAULT_THRESHOLDS)

    def explain(self, phrase_a: str, phrase_b: str) -> dict[str, float]:
        a = build_phrase(phrase_a)
        b = build_phrase(phrase_b)
        return compute_signals(phrase_a, phrase_b, self._index, a, b)
