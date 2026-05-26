"""Backend interface: corpus-bound pairwise scorer with optional signal breakdown."""
from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass

from combiner import label_for


@dataclass(frozen=True)
class BackendMatchResult:
    """One scored candidate produced by any backend."""
    candidate: str
    score: float
    label: str
    signals: dict[str, float] | None


class Backend(abc.ABC):
    """A backend is constructed with a corpus and scores phrase pairs in [0, 1]."""

    def __init__(self, corpus: Sequence[str]) -> None:
        if not corpus:
            raise ValueError("Backend requires at least one candidate")
        self._corpus: tuple[str, ...] = tuple(corpus)

    @abc.abstractmethod
    def score_pair(self, phrase_a: str, phrase_b: str) -> float:
        """Return a similarity score in [0, 1] for the pair."""

    @property
    @abc.abstractmethod
    def thresholds(self) -> tuple[float, float]:
        """Return (low, high) thresholds used by the default label() impl."""

    def label(self, score: float) -> str:
        """Default labeling: MATCH >= high, PARTIAL >= low, else NO_MATCH."""
        return label_for(score, *self.thresholds)

    def score_with_explain(
        self, phrase_a: str, phrase_b: str,
    ) -> tuple[float, dict[str, float] | None]:
        """Return (score, signals). Override to expose a per-signal breakdown alongside the score."""
        return self.score_pair(phrase_a, phrase_b), None
