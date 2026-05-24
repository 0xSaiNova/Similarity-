"""Tests for signals module."""
import math

import pytest

from signals import char_ngram_sim, jaccard, order_sim


def test_jaccard_identical_sets():
    assert jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_partial_overlap():
    # one shared token, three total
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_jaccard_set_semantics_ignore_repeats():
    # duplicates collapse to {a, b} on both sides
    assert jaccard(["a", "a", "b"], ["a", "b", "b"]) == 1.0


def test_jaccard_both_empty_returns_zero():
    assert jaccard([], []) == 0.0


def test_jaccard_one_empty_returns_zero():
    assert jaccard(["a", "b"], []) == 0.0


def test_char_ngram_identical_text_is_one():
    assert char_ngram_sim("hello", "hello") == pytest.approx(1.0)


def test_char_ngram_typo_is_high():
    # helo is hello with one missing letter
    score = char_ngram_sim("hello", "helo")
    assert score > 0.5


def test_char_ngram_unrelated_words_is_low():
    score = char_ngram_sim("hello", "zebra")
    assert score < 0.2


def test_char_ngram_typo_beats_unrelated():
    typo = char_ngram_sim("database", "datbase")
    unrelated = char_ngram_sim("database", "umbrella")
    assert typo > unrelated


def test_char_ngram_empty_text_returns_zero():
    assert char_ngram_sim("", "hello") == 0.0
    assert char_ngram_sim("hello", "") == 0.0
    assert char_ngram_sim("", "") == 0.0


def test_order_sim_identical_tokens_is_one():
    assert order_sim(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_order_sim_reversed_subject_object_is_lower():
    forward = order_sim(["client", "call", "server"], ["client", "call", "server"])
    reversed_ = order_sim(["client", "call", "server"], ["server", "call", "client"])
    assert reversed_ < forward
    assert reversed_ < 0.8


def test_order_sim_completely_disjoint_is_zero():
    assert order_sim(["a", "b"], ["x", "y"]) == 0.0


def test_order_sim_both_empty_returns_zero():
    assert order_sim([], []) == 0.0


def test_order_sim_one_empty_returns_zero():
    assert order_sim(["a", "b"], []) == 0.0
    assert order_sim([], ["a", "b"]) == 0.0


def test_order_sim_inserted_token_breaks_bigrams():
    # inserting "x" between "a" and "c" wipes the shared bigrams
    assert order_sim(["a", "b", "c"], ["a", "x", "c"]) == 0.0


def test_order_sim_partial_bigram_overlap():
    # bigrams a: {(a,b),(b,c),(c,d)}; bigrams b: {(a,b),(b,x),(x,d)} -> 1 / 5
    assert order_sim(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == pytest.approx(1 / 5)


def test_order_sim_single_token_phrase_set_compare():
    # too short for bigrams; falls back to set compare
    assert order_sim(["a"], ["a"]) == 1.0
    assert order_sim(["a"], ["b"]) == 0.0


def test_order_sim_swapped_pair_is_zero():
    # role swap: (subject, verb, object) vs (object, verb, subject) shares no bigrams
    assert order_sim(["producer", "push", "consumer"], ["consumer", "push", "producer"]) == 0.0
