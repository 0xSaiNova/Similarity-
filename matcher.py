"""Match orchestration: blocking + per-signal scoring + combination."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from combiner import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, combine, load_config
from index import PhraseIndex
from preprocess import build_phrase
from signals import char_ngram_sim, jaccard, order_sim
from wordnet_sim import alignment_sim


@dataclass(frozen=True)
class MatchResult:
    """One scored candidate."""
    candidate: str
    score: float
    label: str
    signals: dict[str, float]
    negation_mismatch: bool


def compute_signals(phrase_a: str, phrase_b: str, index: PhraseIndex) -> dict[str, float]:
    """Compute the 5-signal dict for a phrase pair."""
    a = build_phrase(phrase_a)
    b = build_phrase(phrase_b)
    return {
        "tfidf": index.tfidf_cosine(phrase_a, phrase_b),
        "jaccard": jaccard(a.tokens, b.tokens),
        "wordnet": alignment_sim(a.tokens, b.tokens),
        "ngram": char_ngram_sim(phrase_a, phrase_b),
        "order": order_sim(a.tokens, b.tokens),
    }


class Matcher:
    """Match a query phrase against a fixed candidate set."""

    def __init__(
        self,
        candidates: Sequence[str],
        config_path: str | Path | None = None,
    ) -> None:
        self.index = PhraseIndex(candidates)
        if config_path is None:
            self.weights = dict(DEFAULT_WEIGHTS)
            self.thresholds = dict(DEFAULT_THRESHOLDS)
        else:
            self.weights, self.thresholds = load_config(config_path)

    def match(self, query: str, k: int = 20) -> list[MatchResult]:
        """Block to top-k candidates, score each, return sorted by score desc."""
        query_phrase = build_phrase(query)
        candidates = self.index.top_k(query, k)
        results: list[MatchResult] = []
        for cand in candidates:
            cand_phrase = build_phrase(cand)
            signals = compute_signals(query, cand, self.index)
            neg_mismatch = query_phrase.has_negation != cand_phrase.has_negation
            score, label = combine(signals, neg_mismatch, self.weights, self.thresholds)
            results.append(MatchResult(cand, score, label, signals, neg_mismatch))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
