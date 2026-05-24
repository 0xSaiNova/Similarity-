"""WordNet-based word and alignment similarity."""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from nltk.corpus import wordnet

from preprocess import ensure_nltk_data


_ALL_POS: tuple = (wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV)


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
    """Average best match word similarity in both directions; 0.0 if either list empty."""
    if not tokens_a or not tokens_b:
        return 0.0
    return (
        _directional_best_avg(tokens_a, tokens_b)
        + _directional_best_avg(tokens_b, tokens_a)
    ) / 2.0
