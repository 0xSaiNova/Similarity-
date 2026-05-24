"""Tests for wordnet_sim module."""
import pytest

from wordnet_sim import alignment_sim, word_similarity


def test_word_similarity_synonym_pair_is_high():
    # pause and halt sit in the same wordnet neighborhood as verbs and nouns
    assert word_similarity("pause", "halt") > 0.7


def test_word_similarity_identical_word_is_one():
    assert word_similarity("dog", "dog") == pytest.approx(1.0)


def test_word_similarity_unrelated_pair_is_low():
    # pause and laptop have nothing to do with each other
    assert word_similarity("pause", "laptop") < 0.4


def test_word_similarity_no_synset_returns_zero():
    # gibberish token has no synsets
    assert word_similarity("xyzzyqq", "dog") == 0.0
    assert word_similarity("dog", "xyzzyqq") == 0.0


def test_word_similarity_synonym_beats_unrelated():
    syn = word_similarity("car", "automobile")
    unrelated = word_similarity("car", "banana")
    assert syn > unrelated


def test_word_similarity_is_memoized():
    # calling twice should give the same value
    a = word_similarity("pause", "halt")
    b = word_similarity("pause", "halt")
    assert a == b


def test_alignment_sim_synonym_paraphrase_is_high():
    a = ["car", "pause"]
    b = ["automobile", "halt"]
    assert alignment_sim(a, b) > 0.7


def test_alignment_sim_unrelated_phrases_is_low():
    # wu palmer never hits zero on real nouns since they all share the entity root
    a = ["banana", "laptop"]
    b = ["volcano", "umbrella"]
    assert alignment_sim(a, b) < 0.5


def test_alignment_sim_synonym_beats_unrelated():
    syn = alignment_sim(["car", "pause"], ["automobile", "halt"])
    unrelated = alignment_sim(["car", "pause"], ["banana", "umbrella"])
    assert syn > unrelated


def test_alignment_sim_empty_returns_zero():
    assert alignment_sim([], ["dog"]) == 0.0
    assert alignment_sim(["dog"], []) == 0.0
    assert alignment_sim([], []) == 0.0


def test_alignment_sim_identical_tokens_is_one():
    assert alignment_sim(["dog", "cat"], ["dog", "cat"]) == pytest.approx(1.0)
