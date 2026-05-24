"""WordNet-based word and alignment similarity."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from nltk.corpus import wordnet

from preprocess import ensure_nltk_data


_ALL_POS: tuple = (wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV)

WORDNET_FLOOR: float = 0.45
SOFT_MATCH_THRESHOLD: float = 0.6


@lru_cache(maxsize=None)
def _all_synsets(word: str) -> tuple:
    ensure_nltk_data()
    out: list = []
    for pos in _ALL_POS:
        out.extend(wordnet.synsets(word, pos=pos))
    return tuple(out)


@lru_cache(maxsize=None)
def _word_similarity_ordered(w1: str, w2: str) -> float:
    syns_a = _all_synsets(w1)
    syns_b = _all_synsets(w2)
    if not syns_a or not syns_b:
        return 0.0
    set_b = set(syns_b)
    best = 0.0
    for a in syns_a:
        if a in set_b:
            return 1.0
        for b in syns_b:
            score = a.wup_similarity(b)
            if score is not None and score > best:
                best = score
    return best


def word_similarity(w1: str, w2: str) -> float:
    """Max Wu-Palmer similarity over all POS, or 1.0 if any synset is shared."""
    a, b = (w1, w2) if w1 <= w2 else (w2, w1)
    return _word_similarity_ordered(a, b)


def _directional_best_avg(src: Sequence[str], tgt: Sequence[str]) -> float:
    return sum(max(word_similarity(s, t) for t in tgt) for s in src) / len(src)


def alignment_sim(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Avg best-match similarity in both directions, rescaled so the noise floor maps to 0."""
    if not tokens_a or not tokens_b:
        return 0.0
    raw = (
        _directional_best_avg(tokens_a, tokens_b)
        + _directional_best_avg(tokens_b, tokens_a)
    ) / 2.0
    span = 1.0 - WORDNET_FLOOR
    if span <= 0.0:
        return 0.0
    return max(0.0, (raw - WORDNET_FLOOR) / span)


def _has_soft_match(token: str, others: Iterable[str]) -> bool:
    for other in others:
        if token == other or word_similarity(token, other) >= SOFT_MATCH_THRESHOLD:
            return True
    return False


def soft_overlap(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Fraction of tokens with a literal or WordNet-thresholded match in the other list."""
    total = len(tokens_a) + len(tokens_b)
    if total == 0:
        return 0.0
    matches_a = sum(1 for t in tokens_a if _has_soft_match(t, tokens_b))
    matches_b = sum(1 for t in tokens_b if _has_soft_match(t, tokens_a))
    return (matches_a + matches_b) / total
