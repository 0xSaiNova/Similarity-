"""Tests for Matcher orchestration."""
from __future__ import annotations

import pytest

from combiner import SIGNAL_NAMES
from index import PhraseIndex
from matcher import Matcher, compute_signals


@pytest.fixture
def candidates() -> list[str]:
    return [
        "the cat sat on the mat",
        "a dog ran in the park",
        "birds fly south for winter",
        "the cat jumped over the fence",
    ]


@pytest.fixture
def matcher(candidates: list[str]) -> Matcher:
    return Matcher(candidates)


def test_compute_signals_returns_all_five(candidates: list[str]) -> None:
    index = PhraseIndex(candidates)
    signals = compute_signals("the cat sat on the mat", "the cat sat on the mat", index)
    assert set(signals.keys()) == set(SIGNAL_NAMES)
    for value in signals.values():
        assert -1e-9 <= value <= 1.0 + 1e-9


def test_compute_signals_identical_phrase_is_high(candidates: list[str]) -> None:
    index = PhraseIndex(candidates)
    signals = compute_signals("the cat sat on the mat", "the cat sat on the mat", index)
    assert signals["tfidf"] == pytest.approx(1.0)
    assert signals["jaccard"] == pytest.approx(1.0)
    assert signals["order"] == pytest.approx(1.0)


def test_match_returns_results_sorted_descending(matcher: Matcher) -> None:
    results = matcher.match("the cat sat on the mat", k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_match_top_result_is_exact_phrase(matcher: Matcher) -> None:
    results = matcher.match("the cat sat on the mat", k=4)
    assert results[0].candidate == "the cat sat on the mat"
    assert results[0].label == "MATCH"


def test_match_results_include_signal_breakdown(matcher: Matcher) -> None:
    results = matcher.match("the cat sat on the mat", k=2)
    for r in results:
        assert set(r.signals.keys()) == set(SIGNAL_NAMES)


def test_match_blocking_caps_result_count(matcher: Matcher) -> None:
    results = matcher.match("cat", k=2)
    assert len(results) == 2


def test_match_antonym_mismatch_demotes_candidate() -> None:
    cands = ["scale up the backend services", "scale down the backend services"]
    m = Matcher(cands)
    results = m.match("scale up the backend services", k=2)
    by_cand = {r.candidate: r for r in results}
    same = by_cand["scale up the backend services"]
    opposite = by_cand["scale down the backend services"]
    assert same.antonym_mismatch is False
    assert opposite.antonym_mismatch is True
    assert opposite.score < same.score


def test_match_negation_mismatch_demotes_candidate() -> None:
    cands = [
        "the system updates the cache",
        "the system does not update the cache",
    ]
    m = Matcher(cands)
    results = m.match("the system updates the cache", k=2)
    by_cand = {r.candidate: r for r in results}
    affirmative = by_cand["the system updates the cache"]
    negated = by_cand["the system does not update the cache"]
    assert affirmative.negation_mismatch is False
    assert negated.negation_mismatch is True
    assert negated.score < affirmative.score


def test_matcher_loads_config_path(tmp_path, candidates: list[str]) -> None:
    import json
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "weights": {"tfidf": 1.0, "jaccard": 0.0, "wordnet": 0.0, "ngram": 0.0, "order": 0.0},
        "thresholds": {"low": 0.1, "high": 0.5},
    }))
    m = Matcher(candidates, config_path=config)
    assert m.weights["tfidf"] == 1.0
    assert m.thresholds["high"] == 0.5


def test_matcher_default_config_matches_defaults(candidates: list[str]) -> None:
    from combiner import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS
    m = Matcher(candidates)
    assert m.weights == DEFAULT_WEIGHTS
    assert m.thresholds == DEFAULT_THRESHOLDS


def test_matcher_empty_candidates_raises() -> None:
    with pytest.raises(ValueError):
        Matcher([])


def test_match_is_deterministic(matcher: Matcher) -> None:
    a = matcher.match("the cat sat on the mat", k=3)
    b = matcher.match("the cat sat on the mat", k=3)
    assert [r.candidate for r in a] == [r.candidate for r in b]
    assert [r.score for r in a] == [r.score for r in b]
