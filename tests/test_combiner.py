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
    heavy_weights = {name: 5.0 for name in SIGNAL_NAMES}
    score, _ = combine(_full_signals(1.0), negation_mismatch=False, weights=heavy_weights)
    assert score == 1.0


def test_load_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    weights, thresholds = load_config(tmp_path / "nope.json")
    assert weights == DEFAULT_WEIGHTS
    assert thresholds == DEFAULT_THRESHOLDS


def test_load_config_reads_weights_and_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = {
        "weights": {"tfidf": 0.1, "jaccard": 0.2, "wordnet": 0.3, "ngram": 0.15, "order": 0.25},
        "thresholds": {"low": 0.35, "high": 0.8},
    }
    path.write_text(json.dumps(payload))
    weights, thresholds = load_config(path)
    assert weights == payload["weights"]
    assert thresholds == payload["thresholds"]


def test_combine_no_kwargs_uses_defaults() -> None:
    signals = _full_signals(1.0)
    score_default, label_default = combine(signals, negation_mismatch=False)
    score_explicit, label_explicit = combine(
        signals, negation_mismatch=False, weights=DEFAULT_WEIGHTS, thresholds=DEFAULT_THRESHOLDS,
    )
    assert score_default == score_explicit
    assert label_default == label_explicit


def test_combine_does_not_mutate_caller_dicts() -> None:
    weights = dict(DEFAULT_WEIGHTS)
    thresholds = dict(DEFAULT_THRESHOLDS)
    weights_snapshot = dict(weights)
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
