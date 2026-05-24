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


def test_top_k_returns_closest_lexical_match(index: PhraseIndex) -> None:
    results = index.top_k("the cat sat on the mat", k=1)
    assert results == ["the cat sat on the mat"]


def test_top_k_ranks_cat_phrases_above_unrelated(index: PhraseIndex) -> None:
    results = index.top_k("cat on a mat", k=4)
    assert "the cat sat on the mat" in results[:2]


def test_top_k_returns_at_most_k(index: PhraseIndex) -> None:
    results = index.top_k("cat", k=2)
    assert len(results) == 2


def test_top_k_caps_at_candidate_count(index: PhraseIndex) -> None:
    results = index.top_k("cat", k=999)
    assert len(results) == 4


def test_top_k_zero_returns_empty(index: PhraseIndex) -> None:
    assert index.top_k("anything", k=0) == []


def test_top_k_query_with_no_known_tokens_returns_empty(index: PhraseIndex) -> None:
    assert index.top_k("xyzzy quux", k=3) == []


def test_empty_candidates_raises() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        PhraseIndex([])


def test_single_candidate_returns_self() -> None:
    idx = PhraseIndex(["only one phrase"])
    assert idx.top_k("only one phrase", k=1) == ["only one phrase"]
    assert idx.top_k("only one phrase", k=5) == ["only one phrase"]


def test_top_k_is_deterministic(index: PhraseIndex) -> None:
    a = index.top_k("cat on a mat", k=3)
    b = index.top_k("cat on a mat", k=3)
    assert a == b
