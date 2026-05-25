"""Per pair signal computation shared by the classical backend and the tuner."""
from __future__ import annotations

from index import PhraseIndex
from preprocess import Phrase, build_phrase
from signals import char_ngram_sim, jaccard, order_sim
from wordnet_sim import alignment_sim, soft_overlap


def compute_signals(
    phrase_a: str,
    phrase_b: str,
    index: PhraseIndex,
    a: Phrase | None = None,
    b: Phrase | None = None,
) -> dict[str, float]:
    """Compute the 6 signal dict for a phrase pair. Pre built Phrases reused if supplied."""
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
