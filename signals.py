"""Pairwise similarity signals: token Jaccard, character n-gram cosine, word order."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


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


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def order_sim(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """LCS length normalized by the longer list; 0.0 if either list empty."""
    if not tokens_a or not tokens_b:
        return 0.0
    return _lcs_length(tokens_a, tokens_b) / max(len(tokens_a), len(tokens_b))
