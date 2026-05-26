"""Tests for PhraseIndex."""
from __future__ import annotations

import pytest

from index import PhraseIndex


@pytest.fixture
def index() -> PhraseIndex:
    return PhraseIndex([
        "the cat sat on the mat",
        "a dog ran in the park",
        "birds fly south for winter",
        "the cat jumped over the fence",
    ])


def test_tfidf_cosine_identical_text_is_one(index: PhraseIndex) -> None:
    assert index.tfidf_cosine("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_tfidf_cosine_disjoint_is_lower(index: PhraseIndex) -> None:
    same = index.tfidf_cosine("cats sitting on mats", "cats sitting on mats")
    diff = index.tfidf_cosine("cats sitting on mats", "birds flying south")
    assert diff < same


def test_tfidf_cosine_unrelated_is_zero(index: PhraseIndex) -> None:
    assert index.tfidf_cosine("birds fly south", "dogs run parks") < 0.3


def test_empty_candidates_raises() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        PhraseIndex([])
