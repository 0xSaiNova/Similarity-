"""Match orchestration: blocking + per-signal scoring + combination."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from combiner import DEFAULT_PENALTIES, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, combine, load_config
from index import PhraseIndex
from preprocess import Phrase, build_phrase, detect_antonym_mismatch
from signals import char_ngram_sim, detect_order_mismatch, jaccard, order_sim
from wordnet_sim import alignment_sim, soft_overlap


# todo(e5): kill Matcher + MatchResult, only their own tests still hit them. keep compute_signals though, classical needs it.
@dataclass(frozen=True)
class MatchResult:
    """One scored candidate."""
    candidate: str
    score: float
    label: str
    signals: dict[str, float]
    negation_mismatch: bool
    antonym_mismatch: bool
    order_mismatch: bool


def compute_signals(
    phrase_a: str,
    phrase_b: str,
    index: PhraseIndex,
    a: Phrase | None = None,
    b: Phrase | None = None,
) -> dict[str, float]:
    """Compute the 6-signal dict for a phrase pair. Pre-built Phrases reused if supplied."""
    if a is None:
        a = build_phrase(phrase_a)
    if b is None:
        b = build_phrase(phrase_b)
    return {
        "tfidf": index.tfidf_cosine(phrase_a, phrase_b),
        "jaccard": jaccard(a.tokens, b.tokens),
        "wordnet": alignment_sim(a.tokens, b.tokens),
        "ngram": char_ngram_sim(phrase_a, phrase_b),
        "order": order_sim(a.tokens, b.tokens),
        "soft_overlap": soft_overlap(a.tokens, b.tokens),
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
            self.weights = deepcopy(DEFAULT_WEIGHTS)
            self.thresholds = dict(DEFAULT_THRESHOLDS)
            self.penalties = dict(DEFAULT_PENALTIES)
        else:
            self.weights, self.thresholds, self.penalties = load_config(config_path)

    def match(self, query: str, k: int = 20) -> list[MatchResult]:
        """Block to top-k candidates, score each, return sorted by score desc."""
        query_phrase = build_phrase(query)
        candidates = self.index.top_k(query, k)
        results: list[MatchResult] = []
        for cand in candidates:
            cand_phrase = build_phrase(cand)
            signals = compute_signals(query, cand, self.index, query_phrase, cand_phrase)
            neg_mismatch = query_phrase.has_negation != cand_phrase.has_negation
            ant_mismatch = detect_antonym_mismatch(query, cand)
            ord_mismatch = detect_order_mismatch(query_phrase.tokens, cand_phrase.tokens)
            score, label = combine(
                signals, neg_mismatch, ant_mismatch, ord_mismatch,
                self.weights, self.thresholds, self.penalties,
            )
            results.append(MatchResult(
                cand, score, label, signals, neg_mismatch, ant_mismatch, ord_mismatch,
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
