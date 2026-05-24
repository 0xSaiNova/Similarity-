"""Tests for wordnet_sim module."""
import pytest

from wordnet_sim import (
    SOFT_MATCH_THRESHOLD,
    WORDNET_FLOOR,
    alignment_sim,
    soft_overlap,
    word_similarity,
)


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


def test_word_similarity_adjective_synonyms_score_high():
    # slow and sluggish share the adjective synset dull.s.05
    assert word_similarity("slow", "sluggish") >= 0.9


def test_word_similarity_adjective_unrelated_stays_below_synonym():
    # unrelated adj pair must not approach the synonym ceiling
    assert word_similarity("sluggish", "purple") < word_similarity("slow", "sluggish")


def test_alignment_sim_floor_constant_exists():
    assert 0.0 <= WORDNET_FLOOR < 1.0


def test_alignment_sim_unrelated_noun_pair_now_near_zero():
    # unrelated nouns used to sit ~0.47 from Wu-Palmer noise; rescale must crush them
    score = alignment_sim(["banana", "umbrella"], ["volcano", "laptop"])
    assert score < 0.1


def test_alignment_sim_identical_after_rescale_is_one():
    assert alignment_sim(["dog", "cat"], ["dog", "cat"]) == pytest.approx(1.0)


def test_soft_overlap_threshold_constant_exists():
    assert 0.0 < SOFT_MATCH_THRESHOLD < 1.0


def test_soft_overlap_identical_tokens_is_one():
    assert soft_overlap(["car", "dog"], ["car", "dog"]) == pytest.approx(1.0)


def test_soft_overlap_full_synonym_phrase_reaches_literal_magnitude():
    # every word has a synonym partner -> close to literal match magnitude
    score = soft_overlap(["car", "pause"], ["automobile", "halt"])
    assert score >= 0.8


def test_soft_overlap_unrelated_phrases_near_zero():
    score = soft_overlap(["banana", "laptop"], ["volcano", "umbrella"])
    assert score < 0.25


def test_soft_overlap_synonym_beats_unrelated():
    syn = soft_overlap(["car", "pause"], ["automobile", "halt"])
    unrel = soft_overlap(["car", "pause"], ["banana", "umbrella"])
    assert syn > unrel + 0.4


def test_soft_overlap_partial_match_proportional():
    # one word matches literally, one does not match at all
    score = soft_overlap(["car", "xyz"], ["car", "qqq"])
    assert score == pytest.approx(0.5)


def test_soft_overlap_both_empty_returns_zero():
    assert soft_overlap([], []) == 0.0


def test_soft_overlap_one_empty_returns_zero():
    assert soft_overlap([], ["dog"]) == 0.0
    assert soft_overlap(["dog"], []) == 0.0
