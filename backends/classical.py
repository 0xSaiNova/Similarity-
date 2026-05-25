"""Classical backend: thin wrapper around the existing WordNet ensemble engine."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from backends.base import Backend
from combiner import combine, label_for, load_config
from index import PhraseIndex
from matcher import compute_signals
from preprocess import build_phrase, detect_antonym_mismatch
from signals import detect_order_mismatch

DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config.json"


class ClassicalBackend(Backend):
    """Wraps the existing engine: TF IDF index + 6 signals + 3 gates + combine."""

    def __init__(
        self, corpus: Sequence[str], config_path: str | Path | None = None,
    ) -> None:
        super().__init__(corpus)
        self._index = PhraseIndex(self._corpus)
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        self._weights, self._thresholds_dict, self._penalties = load_config(config_path)

    def _compute(self, phrase_a: str, phrase_b: str) -> tuple[float, dict[str, float]]:
        a = build_phrase(phrase_a)
        b = build_phrase(phrase_b)
        signals = compute_signals(phrase_a, phrase_b, self._index, a, b)
        score, _ = combine(
            signals,
            a.has_negation != b.has_negation,
            detect_antonym_mismatch(phrase_a, phrase_b),
            detect_order_mismatch(a.tokens, b.tokens),
            self._weights, self._thresholds_dict, self._penalties,
        )
        return score, signals

    def score_pair(self, phrase_a: str, phrase_b: str) -> float:
        score, _ = self._compute(phrase_a, phrase_b)
        return score

    def score_with_explain(
        self, phrase_a: str, phrase_b: str,
    ) -> tuple[float, dict[str, float]]:
        return self._compute(phrase_a, phrase_b)

    @property
    def thresholds(self) -> tuple[float, float]:
        return self._thresholds_dict["low"], self._thresholds_dict["high"]

    def label(self, score: float) -> str:
        return label_for(score, self._thresholds_dict)
