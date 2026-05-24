"""Tests for tune.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from combiner import DEFAULT_PENALTIES, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, load_config
from evaluate import load_gold
from tune import (
    MIN_THRESHOLD_GAP,
    N_PARAMS_THRESHOLDS,
    cache_features,
    cross_validate_thresholds,
    params_to_thresholds,
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
    out = params_to_thresholds([0.8, 0.2])
    assert out["low"] == pytest.approx(0.2)
    assert out["high"] == pytest.approx(0.8)


def test_threshold_objective_returns_valid_macro_f1(small_cache) -> None:
    f1 = threshold_objective([0.4, 0.7], small_cache)
    assert 0.0 <= f1 <= 1.0


def test_threshold_objective_collapses_when_too_close(small_cache) -> None:
    params = [0.5, 0.5 + MIN_THRESHOLD_GAP / 2]
    assert threshold_objective(params, small_cache) == 0.0


def test_write_config_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_config(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS, DEFAULT_PENALTIES, path)
    loaded_w, loaded_t, loaded_p = load_config(path)
    assert loaded_w == DEFAULT_WEIGHTS
    assert loaded_t == DEFAULT_THRESHOLDS
    assert loaded_p == DEFAULT_PENALTIES


def test_search_thresholds_returns_2_params_in_unit_interval(small_cache) -> None:
    _, params = search_thresholds(small_cache, step=0.1)
    assert len(params) == N_PARAMS_THRESHOLDS
    thresholds = params_to_thresholds(params)
    assert 0.0 <= thresholds["low"] < thresholds["high"] <= 1.0


def test_search_thresholds_is_deterministic(small_cache) -> None:
    f1_a, params_a = search_thresholds(small_cache, step=0.1)
    f1_b, params_b = search_thresholds(small_cache, step=0.1)
    assert f1_a == f1_b
    assert list(params_a) == list(params_b)


def test_tuned_thresholds_in_sample_beat_or_equal_default(small_cache) -> None:
    default = threshold_objective([0.4, 0.7], small_cache)
    tuned, _ = search_thresholds(small_cache, step=0.1)
    assert tuned >= default


def test_cross_validate_returns_tuned_and_default_macros(small_cache) -> None:
    tuned, default, per_label = cross_validate_thresholds(small_cache, folds=3, step=0.1)
    assert 0.0 <= tuned <= 1.0
    assert 0.0 <= default <= 1.0
    assert set(per_label.keys()) == {"MATCH", "PARTIAL", "NO_MATCH"}


def test_should_adopt_strict_greater() -> None:
    assert should_adopt(0.55, 0.50) is True


def test_should_adopt_false_when_equal() -> None:
    assert should_adopt(0.50, 0.50) is False


def test_should_adopt_false_when_worse() -> None:
    assert should_adopt(0.40, 0.50) is False
