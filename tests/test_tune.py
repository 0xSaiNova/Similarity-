"""Tests for tune.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from combiner import DEFAULT_PENALTIES, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, load_config
from evaluate import load_gold
from tune import (
    MIN_THRESHOLD_GAP,
    N_PARAMS_FULL,
    N_PARAMS_THRESHOLDS,
    cache_features,
    full_objective,
    params_to_full_config,
    params_to_thresholds,
    search_full,
    search_thresholds,
    should_adopt,
    threshold_objective,
    write_config,
)

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_pairs.json"


@pytest.fixture(scope="module")
def small_cache():
    gold = load_gold(GOLD_PATH)
    return cache_features(gold[:30])


def test_params_to_thresholds_orders() -> None:
    # high < low input must be reordered
    out = params_to_thresholds([0.8, 0.2])
    assert out["low"] == pytest.approx(0.2)
    assert out["high"] == pytest.approx(0.8)


def test_params_to_full_config_normalizes_weights() -> None:
    params = [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 0.3, 0.6, 0.5, 0.5, 0.5]
    weights, thresholds, penalties = params_to_full_config(params)
    assert sum(weights["surface"].values()) == pytest.approx(1.0)
    assert sum(weights["semantic"].values()) == pytest.approx(1.0)
    assert thresholds["low"] == pytest.approx(0.3)
    assert thresholds["high"] == pytest.approx(0.6)
    assert penalties == {"negation": 0.5, "antonym": 0.5, "order": 0.5}


def test_threshold_objective_returns_valid_macro_f1(small_cache) -> None:
    f1 = threshold_objective([0.4, 0.7], small_cache)
    assert 0.0 <= f1 <= 1.0


def test_threshold_objective_collapses_when_too_close(small_cache) -> None:
    params = [0.5, 0.5 + MIN_THRESHOLD_GAP / 2]
    assert threshold_objective(params, small_cache) == 0.0


def test_full_objective_returns_valid_macro_f1(small_cache) -> None:
    params = [0.25] * 4 + [0.5] * 2 + [0.4, 0.7] + [0.3, 0.3, 0.5]
    f1 = full_objective(params, small_cache)
    assert 0.0 <= f1 <= 1.0


def test_write_config_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_config(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS, DEFAULT_PENALTIES, path)
    loaded_w, loaded_t, loaded_p = load_config(path)
    assert loaded_w == DEFAULT_WEIGHTS
    assert loaded_t == DEFAULT_THRESHOLDS
    assert loaded_p == DEFAULT_PENALTIES


def test_search_thresholds_returns_2_params_in_unit_interval(small_cache) -> None:
    _, params = search_thresholds(small_cache, maxiter=15, popsize=8)
    assert len(params) == N_PARAMS_THRESHOLDS
    thresholds = params_to_thresholds(params)
    assert 0.0 <= thresholds["low"] < thresholds["high"] <= 1.0


def test_search_full_returns_11_params(small_cache) -> None:
    _, params = search_full(small_cache, maxiter=10, popsize=8)
    assert len(params) == N_PARAMS_FULL
    weights, thresholds, penalties = params_to_full_config(params)
    assert sum(weights["surface"].values()) == pytest.approx(1.0)
    assert sum(weights["semantic"].values()) == pytest.approx(1.0)
    assert 0.0 <= thresholds["low"] < thresholds["high"] <= 1.0
    for v in penalties.values():
        assert 0.01 <= v <= 1.0


def test_tuned_thresholds_in_sample_beat_or_equal_default(small_cache) -> None:
    default = threshold_objective([0.4, 0.7], small_cache)
    tuned, _ = search_thresholds(small_cache, maxiter=20, popsize=10)
    assert tuned >= default


def test_should_adopt_strict_greater() -> None:
    assert should_adopt(0.55, 0.50) is True


def test_should_adopt_false_when_equal() -> None:
    assert should_adopt(0.50, 0.50) is False


def test_should_adopt_false_when_worse() -> None:
    assert should_adopt(0.40, 0.50) is False
