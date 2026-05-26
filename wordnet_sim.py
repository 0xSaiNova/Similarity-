"""WordNet-based word and alignment similarity, POS-matched and IC-weighted."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from nltk.corpus import wordnet

from backends.embedding_utils import clamp_unit
from preprocess import ensure_nltk_data

# POS groups: nouns and verbs use Lin similarity (IC-weighted; collapses unrelated
# abstract LCSs like entity/act/change). Adjectives and adverbs fall back to
# Wu-Palmer since the Brown IC file only covers noun and verb hierarchies and
# adjective WordNet structure is shallow clusters that don't suffer the same
# inflation. `s` is the adjective-satellite POS; treat it as adjective.
_NOUN_VERB_POS: tuple[str, ...] = (wordnet.NOUN, wordnet.VERB)
_ADJ_ADV_POS: tuple[str, ...] = (wordnet.ADJ, wordnet.ADJ_SAT, wordnet.ADV)
_ALL_POS: tuple[str, ...] = _NOUN_VERB_POS + _ADJ_ADV_POS

WORDNET_FLOOR: float = 0.4
SOFT_MATCH_THRESHOLD: float = 0.7


@lru_cache(maxsize=1)
def _brown_ic():
    ensure_nltk_data()
    from nltk.corpus import wordnet_ic
    return wordnet_ic.ic("ic-brown.dat")


@lru_cache(maxsize=50000)
def _synsets_by_pos(word: str) -> dict[str, tuple]:
    ensure_nltk_data()
    return {pos: tuple(wordnet.synsets(word, pos=pos)) for pos in _ALL_POS}


def _pair_score(sa, sb, pos: str, ic) -> float:
    if pos in _NOUN_VERB_POS:
        score = sa.lin_similarity(sb, ic)
    else:
        score = sa.wup_similarity(sb)
    if score is None:
        return 0.0
    return clamp_unit(float(score))


@lru_cache(maxsize=200000)
def _word_similarity_ordered(w1: str, w2: str) -> float:
    syns_a = _synsets_by_pos(w1)
    syns_b = _synsets_by_pos(w2)
    if not any(syns_a.values()) or not any(syns_b.values()):
        return 0.0
    ic = _brown_ic()
    best = 0.0
    for pos in _ALL_POS:
        a_list = syns_a[pos]
        b_list = syns_b[pos]
        if not a_list or not b_list:
            continue
        b_set = set(b_list)
        for sa in a_list:
            if sa in b_set:
                return 1.0
            for sb in b_list:
                score = _pair_score(sa, sb, pos, ic)
                if score > best:
                    best = score
    return best


def word_similarity(w1: str, w2: str) -> float:
    """Max IC-weighted (noun/verb) or Wu-Palmer (adj/adv) similarity, POS-matched. 1.0 on shared synset."""
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


def _best_gated_match(token: str, others: Iterable[str]) -> float:
    best = 0.0
    for other in others:
        if token == other:
            return 1.0
        sim = word_similarity(token, other)
        if sim >= SOFT_MATCH_THRESHOLD and sim > best:
            best = sim
    return best


def soft_overlap(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Symmetric average of best-match similarity per token, gated by SOFT_MATCH_THRESHOLD."""
    if not tokens_a or not tokens_b:
        return 0.0
    avg_a = sum(_best_gated_match(t, tokens_b) for t in tokens_a) / len(tokens_a)
    avg_b = sum(_best_gated_match(t, tokens_a) for t in tokens_b) / len(tokens_b)
    return (avg_a + avg_b) / 2.0
