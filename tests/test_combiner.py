"""Tests for the signal combiner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from combiner import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    SIGNAL_NAMES,
    combine,
    load_config,
)


def _full_signals(value: float) -> dict[str, float]:
    return {name: value for name in SIGNAL_NAMES}


def test_combine_perfect_signals_yields_match() -> None:
    score, label = combine(_full_signals(1.0), negation_mismatch=False)
    assert score == pytest.approx(1.0)
    assert label == "MATCH"


def test_combine_zero_signals_yields_no_match() -> None:
    score, label = combine(_full_signals(0.0), negation_mismatch=False)
    assert score == 0.0
    assert label == "NO_MATCH"


def test_combine_partial_score_yields_partial() -> None:
    score, label = combine(_full_signals(0.5), negation_mismatch=False)
    assert label == "PARTIAL"
    assert DEFAULT_THRESHOLDS["low"] <= score < DEFAULT_THRESHOLDS["high"]


def test_combine_negation_mismatch_drops_score_well_below_no_mismatch() -> None:
    signals = _full_signals(0.9)
    clean_score, clean_label = combine(signals, negation_mismatch=False)
    neg_score, neg_label = combine(signals, negation_mismatch=True)
    assert neg_score < clean_score - 0.5
    assert clean_label == "MATCH"
    assert neg_label == "NO_MATCH"


def test_combine_score_is_clamped_to_unit_interval() -> None:
    from combiner import SEMANTIC_SIGNALS, SURFACE_SIGNALS
    heavy = {
        "surface": {name: 5.0 for name in SURFACE_SIGNALS},
        "semantic": {name: 5.0 for name in SEMANTIC_SIGNALS},
    }
    score, _ = combine(_full_signals(1.0), negation_mismatch=False, weights=heavy)
    assert score == 1.0


def test_load_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    from combiner import DEFAULT_PENALTIES
    weights, thresholds, penalties = load_config(tmp_path / "nope.json")
    assert weights == DEFAULT_WEIGHTS
    assert thresholds == DEFAULT_THRESHOLDS
    assert penalties == DEFAULT_PENALTIES


def test_load_config_reads_weights_and_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = {
        "weights": {
            "surface": {"tfidf": 0.1, "jaccard": 0.2, "ngram": 0.4, "order": 0.3},
            "semantic": {"wordnet": 0.6, "soft_overlap": 0.4},
        },
        "thresholds": {"low": 0.35, "high": 0.8},
    }
    path.write_text(json.dumps(payload))
    weights, thresholds, penalties = load_config(path)
    assert weights == payload["weights"]
    assert thresholds == payload["thresholds"]
    # penalties fall back to defaults when omitted
    from combiner import DEFAULT_PENALTIES
    assert penalties == DEFAULT_PENALTIES


def test_load_config_reads_penalties_when_present(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = {
        "weights": {
            "surface": {"tfidf": 0.25, "jaccard": 0.25, "ngram": 0.25, "order": 0.25},
            "semantic": {"wordnet": 0.5, "soft_overlap": 0.5},
        },
        "thresholds": {"low": 0.4, "high": 0.7},
        "penalties": {"negation": 0.2, "antonym": 0.4, "order": 0.6},
    }
    path.write_text(json.dumps(payload))
    _, _, penalties = load_config(path)
    assert penalties == payload["penalties"]


def test_combine_no_kwargs_uses_defaults() -> None:
    signals = _full_signals(1.0)
    score_default, label_default = combine(signals, negation_mismatch=False)
    score_explicit, label_explicit = combine(
        signals, negation_mismatch=False, weights=DEFAULT_WEIGHTS, thresholds=DEFAULT_THRESHOLDS,
    )
    assert score_default == score_explicit
    assert label_default == label_explicit


def test_combine_does_not_mutate_caller_dicts() -> None:
    from copy import deepcopy
    weights = deepcopy(DEFAULT_WEIGHTS)
    thresholds = dict(DEFAULT_THRESHOLDS)
    weights_snapshot = deepcopy(weights)
    thresholds_snapshot = dict(thresholds)
    combine(_full_signals(0.6), negation_mismatch=True, weights=weights, thresholds=thresholds)
    assert weights == weights_snapshot
    assert thresholds == thresholds_snapshot


def test_load_config_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_config(path)


def test_load_config_missing_keys_raises(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"weights": {}, "thresholds": {}}))
    with pytest.raises(KeyError):
        load_config(path)


def test_combine_antonym_mismatch_drops_score_well_below_clean() -> None:
    signals = _full_signals(0.9)
    clean_score, _ = combine(signals, negation_mismatch=False, antonym_mismatch=False)
    ant_score, ant_label = combine(signals, negation_mismatch=False, antonym_mismatch=True)
    assert ant_score < clean_score - 0.5
    assert ant_label == "NO_MATCH"


def test_combine_negation_and_antonym_compound() -> None:
    signals = _full_signals(1.0)
    both_score, _ = combine(signals, negation_mismatch=True, antonym_mismatch=True)
    neg_only, _ = combine(signals, negation_mismatch=True, antonym_mismatch=False)
    assert both_score < neg_only


def test_combine_antonym_default_is_false() -> None:
    signals = _full_signals(1.0)
    no_flag, _ = combine(signals, negation_mismatch=False)
    explicit_false, _ = combine(signals, negation_mismatch=False, antonym_mismatch=False)
    assert no_flag == explicit_false


def test_combine_pure_semantic_match_reaches_match_label() -> None:
    # surface signals all zero, semantic signals all 1.0 -> max() picks the semantic side
    signals = {"tfidf": 0.0, "jaccard": 0.0, "ngram": 0.0, "order": 0.0,
               "wordnet": 1.0, "soft_overlap": 1.0}
    score, label = combine(signals, negation_mismatch=False)
    assert score == pytest.approx(1.0)
    assert label == "MATCH"


def test_combine_pure_surface_match_reaches_match_label() -> None:
    # mirror: surface group fires, semantic zero
    signals = {"tfidf": 1.0, "jaccard": 1.0, "ngram": 1.0, "order": 1.0,
               "wordnet": 0.0, "soft_overlap": 0.0}
    score, label = combine(signals, negation_mismatch=False)
    assert score == pytest.approx(1.0)
    assert label == "MATCH"


def test_combine_max_not_sum_no_double_counting() -> None:
    # both groups firing at 1.0 must not exceed 1.0 (max, not sum)
    signals = _full_signals(1.0)
    score, _ = combine(signals, negation_mismatch=False)
    assert score == pytest.approx(1.0)


def test_combine_order_mismatch_pulls_match_into_partial() -> None:
    signals = _full_signals(1.0)
    clean, clean_label = combine(signals, negation_mismatch=False)
    pulled, pulled_label = combine(signals, negation_mismatch=False, order_mismatch=True)
    assert clean_label == "MATCH"
    assert pulled_label == "PARTIAL"
    assert pulled < clean


def test_combine_order_penalty_is_moderate_not_strong() -> None:
    # order penalty should leave a 1.0 above the low threshold (not crushed to NO_MATCH)
    signals = _full_signals(1.0)
    pulled, label = combine(signals, negation_mismatch=False, order_mismatch=True)
    assert label == "PARTIAL"
    assert pulled >= DEFAULT_THRESHOLDS["low"]


def test_combine_order_default_is_false() -> None:
    signals = _full_signals(1.0)
    no_flag, _ = combine(signals, negation_mismatch=False)
    explicit_false, _ = combine(signals, negation_mismatch=False, order_mismatch=False)
    assert no_flag == explicit_false
