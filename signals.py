"""Pairwise similarity signals: token Jaccard, character n-gram cosine, word order."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

SET_OVERLAP_THRESHOLD: float = 0.9
ORDER_SCRAMBLE_THRESHOLD: float = 0.5


def jaccard(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Intersection over union of the two token sets; 0.0 if both empty."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _char_ngrams(text: str, n: int) -> Counter[str]:
    padded = " " * (n - 1) + text + " " * (n - 1)
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


def char_ngram_sim(text_a: str, text_b: str, n: int = 3) -> float:
    """Cosine similarity over character n-gram counts; 0.0 if either text empty."""
    if not text_a or not text_b:
        return 0.0
    counts_a = _char_ngrams(text_a, n)
    counts_b = _char_ngrams(text_b, n)
    dot = sum(counts_a[gram] * counts_b[gram] for gram in counts_a.keys() & counts_b.keys())
    norm_a = math.sqrt(sum(c * c for c in counts_a.values()))
    norm_b = math.sqrt(sum(c * c for c in counts_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def order_sim(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Jaccard over adjacent token bigrams; collapses sharply on word-order changes."""
    if not tokens_a or not tokens_b:
        return 0.0
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return 1.0 if set(tokens_a) == set(tokens_b) else 0.0
    bigrams_a = set(zip(tokens_a[:-1], tokens_a[1:]))
    bigrams_b = set(zip(tokens_b[:-1], tokens_b[1:]))
    union = bigrams_a | bigrams_b
    if not union:
        return 0.0
    return len(bigrams_a & bigrams_b) / len(union)


def detect_order_mismatch(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> bool:
    """True when token sets nearly match but adjacent bigrams diverge: scrambled wording."""
    if not tokens_a or not tokens_b:
        return False
    if jaccard(tokens_a, tokens_b) < SET_OVERLAP_THRESHOLD:
        return False
    return order_sim(tokens_a, tokens_b) < ORDER_SCRAMBLE_THRESHOLD
