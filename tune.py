"""Tune combiner thresholds on the gold set; adopt only if cross-validated CV beats default."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold

from combiner import (
    DEFAULT_PENALTIES,
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    SEMANTIC_SIGNALS,
    SURFACE_SIGNALS,
    combine,
)
from evaluate import LABELS, GoldPair, PairResult, _per_label_metrics, load_gold
from index import PhraseIndex
from matcher import compute_signals
from preprocess import build_phrase, detect_antonym_mismatch
from signals import detect_order_mismatch

CV_FOLDS: int = 5
SEED: int = 42
N_PARAMS_FULL: int = 11
N_PARAMS_THRESHOLDS: int = 2
MIN_THRESHOLD_GAP: float = 0.05
CONFIG_PATH: Path = Path("config.json")


@dataclass(frozen=True)
class CachedPair:
    """Gold pair with precomputed 6 signals + 3 gate flags."""
    pair: GoldPair
    signals: dict[str, float]
    negation_mismatch: bool
    antonym_mismatch: bool
    order_mismatch: bool


def cache_features(gold: list[GoldPair]) -> list[CachedPair]:
    """Precompute signals + gate flags for every gold pair."""
    all_phrases = sorted({p.phrase_a for p in gold} | {p.phrase_b for p in gold})
    index = PhraseIndex(all_phrases)
    out: list[CachedPair] = []
    for p in gold:
        a = build_phrase(p.phrase_a)
        b = build_phrase(p.phrase_b)
        out.append(CachedPair(
            pair=p,
            signals=compute_signals(p.phrase_a, p.phrase_b, index),
            negation_mismatch=a.has_negation != b.has_negation,
            antonym_mismatch=detect_antonym_mismatch(p.phrase_a, p.phrase_b),
            order_mismatch=detect_order_mismatch(a.tokens, b.tokens),
        ))
    return out


def _predict(
    cache: list[CachedPair],
    weights: dict, thresholds: dict, penalties: dict,
) -> list[PairResult]:
    out: list[PairResult] = []
    for c in cache:
        score, label = combine(
            c.signals,
            c.negation_mismatch, c.antonym_mismatch, c.order_mismatch,
            weights, thresholds, penalties,
        )
        out.append(PairResult(
            pair_id=c.pair.id, category=c.pair.category, phenomenon=c.pair.phenomenon,
            predicted_score=score, gold_score=c.pair.gold_score,
            predicted_label=label, gold_label=c.pair.label,
        ))
    return out


def _macro_f1(results: list[PairResult]) -> float:
    metrics = _per_label_metrics(results)
    return sum(m.f1 for m in metrics.values()) / len(metrics)


def params_to_thresholds(params: Sequence[float]) -> dict[str, float]:
    """Decode a 2-element params vector into {low, high} thresholds, ordered."""
    low, high = sorted((float(params[0]), float(params[1])))
    return {"low": low, "high": high}


def threshold_objective(params: Sequence[float], cache: list[CachedPair]) -> float:
    """Macro-F1 with fixed default weights + penalties, only thresholds vary."""
    thresholds = params_to_thresholds(params)
    if thresholds["high"] - thresholds["low"] < MIN_THRESHOLD_GAP:
        return 0.0
    return _macro_f1(_predict(cache, DEFAULT_WEIGHTS, thresholds, DEFAULT_PENALTIES))


def search_thresholds(
    cache: list[CachedPair],
    seed: int = SEED,
    maxiter: int = 120,
    popsize: int = 20,
) -> tuple[float, np.ndarray]:
    """Differential evolution over 2 threshold params."""
    result = differential_evolution(
        lambda p: -threshold_objective(p, cache),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6,
        polish=True, workers=1, init="sobol",
    )
    return -result.fun, result.x


def cross_validate_thresholds(
    cache: list[CachedPair],
    folds: int = CV_FOLDS,
    seed: int = SEED,
) -> tuple[float, dict[str, float]]:
    """Stratified k-fold CV with threshold-only tuning."""
    labels = np.array([c.pair.label for c in cache])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_macros: list[float] = []
    fold_label_f1: dict[str, list[float]] = {label: [] for label in LABELS}
    for k, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cache)), labels)):
        train = [cache[i] for i in train_idx]
        test = [cache[i] for i in test_idx]
        _, params = search_thresholds(train, seed=seed + k)
        thresholds = params_to_thresholds(params)
        results = _predict(test, DEFAULT_WEIGHTS, thresholds, DEFAULT_PENALTIES)
        metrics = _per_label_metrics(results)
        fold_macros.append(sum(m.f1 for m in metrics.values()) / len(metrics))
        for label, m in metrics.items():
            fold_label_f1[label].append(m.f1)
    mean_macro = sum(fold_macros) / len(fold_macros)
    mean_per_label = {label: sum(fs) / len(fs) for label, fs in fold_label_f1.items()}
    return mean_macro, mean_per_label


def params_to_full_config(
    params: Sequence[float],
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    """Decode an 11-element params vector. DIAGNOSTIC ONLY — never produces shipped config."""
    raw_surface = np.asarray(params[:4], dtype=float)
    raw_surface = raw_surface / raw_surface.sum() if raw_surface.sum() > 0 else np.full(4, 0.25)
    raw_semantic = np.asarray(params[4:6], dtype=float)
    raw_semantic = raw_semantic / raw_semantic.sum() if raw_semantic.sum() > 0 else np.full(2, 0.5)
    low, high = sorted((float(params[6]), float(params[7])))
    return (
        {
            "surface": {n: float(raw_surface[i]) for i, n in enumerate(SURFACE_SIGNALS)},
            "semantic": {n: float(raw_semantic[i]) for i, n in enumerate(SEMANTIC_SIGNALS)},
        },
        {"low": low, "high": high},
        {"negation": float(params[8]), "antonym": float(params[9]), "order": float(params[10])},
    )


def full_objective(params: Sequence[float], cache: list[CachedPair]) -> float:
    """11-param macro-F1 objective. DIAGNOSTIC ONLY."""
    weights, thresholds, penalties = params_to_full_config(params)
    if thresholds["high"] - thresholds["low"] < MIN_THRESHOLD_GAP:
        return 0.0
    return _macro_f1(_predict(cache, weights, thresholds, penalties))


def search_full(
    cache: list[CachedPair],
    seed: int = SEED,
    maxiter: int = 80,
    popsize: int = 15,
) -> tuple[float, np.ndarray]:
    """11-param DE. DIAGNOSTIC ONLY — output never written to config.json."""
    bounds = [(0.0, 1.0)] * 6 + [(0.0, 1.0)] * 2 + [(0.01, 1.0)] * 3
    result = differential_evolution(
        lambda p: -full_objective(p, cache),
        bounds=bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-5,
        polish=True, workers=1, init="sobol",
    )
    return -result.fun, result.x


def cross_validate_full(
    cache: list[CachedPair],
    folds: int = CV_FOLDS,
    seed: int = SEED,
    maxiter: int = 80,
) -> tuple[float, dict[str, float]]:
    """Stratified CV for the 11-param search. DIAGNOSTIC ONLY."""
    labels = np.array([c.pair.label for c in cache])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_macros: list[float] = []
    fold_label_f1: dict[str, list[float]] = {label: [] for label in LABELS}
    for k, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cache)), labels)):
        train = [cache[i] for i in train_idx]
        test = [cache[i] for i in test_idx]
        _, params = search_full(train, seed=seed + k, maxiter=maxiter)
        weights, thresholds, penalties = params_to_full_config(params)
        results = _predict(test, weights, thresholds, penalties)
        metrics = _per_label_metrics(results)
        fold_macros.append(sum(m.f1 for m in metrics.values()) / len(metrics))
        for label, m in metrics.items():
            fold_label_f1[label].append(m.f1)
    mean_macro = sum(fold_macros) / len(fold_macros)
    mean_per_label = {label: sum(fs) / len(fs) for label, fs in fold_label_f1.items()}
    return mean_macro, mean_per_label


def write_config(
    weights: dict, thresholds: dict, penalties: dict,
    path: str | Path = CONFIG_PATH,
) -> None:
    """Serialize tuned config in the load_config schema."""
    data = {"weights": weights, "thresholds": thresholds, "penalties": penalties}
    Path(path).write_text(json.dumps(data, indent=2))


def should_adopt(candidate_cv_macro: float, baseline_macro: float) -> bool:
    """Adopt the tuned config only if it strictly beats the baseline on CV."""
    return candidate_cv_macro > baseline_macro


def _print_label_metrics(per_label: dict) -> None:
    print(f"  {'label':<10} {'P':>6} {'R':>6} {'F1':>6}")
    for label, m in per_label.items():
        if hasattr(m, "precision"):
            print(f"  {label:<10} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f}")
        else:
            print(f"  {label:<10} {'-':>6} {'-':>6} {m:>6.3f}")


def main() -> None:
    gold = load_gold("data/gold_pairs.json")
    cache = cache_features(gold)

    default_results = _predict(cache, DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS, DEFAULT_PENALTIES)
    default_metrics = _per_label_metrics(default_results)
    default_macro = sum(m.f1 for m in default_metrics.values()) / len(default_metrics)
    print("=== Default config (baseline) ===")
    print(f"Macro F1: {default_macro:.3f}")
    _print_label_metrics(default_metrics)

    print("\n=== Minimal mode (2 thresholds only) — cross-validated ===")
    min_cv_macro, min_cv_per_label = cross_validate_thresholds(cache)
    print(f"Mean cross-validated macro F1: {min_cv_macro:.3f}")
    _print_label_metrics(min_cv_per_label)

    print("\n=== 11-param search — DIAGNOSTIC ONLY, never written to config.json ===")
    diag_cv_macro, diag_cv_per_label = cross_validate_full(cache)
    print(f"Mean cross-validated macro F1: {diag_cv_macro:.3f}")
    _print_label_metrics(diag_cv_per_label)

    print("\n=== Adoption decision ===")
    if should_adopt(min_cv_macro, default_macro):
        in_sample_macro, params = search_thresholds(cache)
        thresholds = params_to_thresholds(params)
        write_config(DEFAULT_WEIGHTS, thresholds, DEFAULT_PENALTIES)
        print(f"ADOPTED: minimal CV {min_cv_macro:.3f} > default {default_macro:.3f}")
        print(f"In-sample macro F1 after fitting on all 100: {in_sample_macro:.3f}")
        print(f"Tuned thresholds: low={thresholds['low']:.3f} high={thresholds['high']:.3f}")
        print(f"Written to {CONFIG_PATH}")
    else:
        print(f"REJECTED: minimal CV {min_cv_macro:.3f} <= default {default_macro:.3f}")
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
            print(f"Removed stale {CONFIG_PATH}; evaluate.py will use built-in defaults")
        else:
            print(f"No {CONFIG_PATH} present; evaluate.py already on defaults")


if __name__ == "__main__":
    main()
