"""Tests for preprocess module."""
import dataclasses

import pytest

from preprocess import Phrase, build_phrase, detect_antonym_mismatch, detect_negation, normalize


def test_lowercase_and_punctuation_removed():
    tokens = normalize("Hello, World!")
    assert tokens == ["hello", "world"]


def test_stopwords_removed():
    tokens = normalize("the quick brown fox")
    assert "the" not in tokens
    assert tokens == ["quick", "brown", "fox"]


def test_lemmatize_verb_present_participle():
    tokens = normalize("dogs are running fast")
    assert "run" in tokens
    assert "running" not in tokens


def test_lemmatize_noun_plural():
    tokens = normalize("many services available")
    assert "service" in tokens
    assert "services" not in tokens


def test_negation_word_not_survives():
    assert "not" in normalize("this is not good")


def test_negation_word_never_survives():
    assert "never" in normalize("never give up")


def test_negation_word_no_survives():
    assert "no" in normalize("no problem here")


def test_negation_word_without_survives():
    assert "without" in normalize("walk without shoes")


def test_nt_contraction_survives():
    assert "n't" in normalize("don't stop believing")


def test_empty_string_yields_empty_tokens():
    assert normalize("") == []


def test_only_punctuation_yields_empty_tokens():
    assert normalize("!!! ??? ...") == []


def test_detect_negation_not():
    assert detect_negation("this is not great") is True


def test_detect_negation_never():
    assert detect_negation("I never go there") is True


def test_detect_negation_no():
    assert detect_negation("no thanks") is True


def test_detect_negation_without():
    assert detect_negation("life without music") is True


def test_detect_negation_nt_contraction():
    assert detect_negation("don't worry") is True


def test_detect_negation_failed_to():
    assert detect_negation("the test failed to run") is True


def test_detect_negation_fails_to():
    assert detect_negation("she fails to deliver") is True


def test_detect_negation_fail_to():
    assert detect_negation("he will fail to comply") is True


def test_detect_negation_plain_phrase():
    assert detect_negation("the cat is happy") is False


def test_detect_negation_no_fail_without_to():
    assert detect_negation("the test failed badly") is False


def test_build_phrase_populates_fields():
    p = build_phrase("Dogs aren't running!")
    assert isinstance(p, Phrase)
    assert p.raw == "Dogs aren't running!"
    assert p.has_negation is True
    assert "dog" in p.tokens
    assert "run" in p.tokens
    assert "n't" in p.tokens


def test_build_phrase_no_negation():
    p = build_phrase("happy dogs play outside")
    assert p.has_negation is False
    assert "dog" in p.tokens


def test_phrase_is_frozen():
    p = build_phrase("test")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.raw = "other"  # type: ignore[misc]


def test_detect_antonym_mismatch_up_vs_down():
    assert detect_antonym_mismatch("scale up the service", "scale down the service") is True


def test_detect_antonym_mismatch_increase_vs_decrease():
    assert detect_antonym_mismatch("increase the rate limit", "decrease the rate limit") is True


def test_detect_antonym_mismatch_enable_vs_disable():
    assert detect_antonym_mismatch("enable caching", "disable caching") is True


def test_detect_antonym_mismatch_no_antonym_returns_false():
    assert detect_antonym_mismatch("migrate the database", "move the database") is False


def test_detect_antonym_mismatch_synonyms_do_not_trigger():
    assert detect_antonym_mismatch("pause the pipeline", "halt the pipeline") is False


def test_detect_antonym_mismatch_symmetric():
    assert detect_antonym_mismatch("scale down the service", "scale up the service") is True
