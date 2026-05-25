"""Tune classical combiner thresholds on the gold set + dispatch per backend tuners."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from combiner import (
    DEFAULT_PENALTIES,
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    combine,
    write_backend_block,
)
from evaluate import LABELS, GoldPair, PairResult, _per_label_metrics, load_gold
from index import PhraseIndex
from matcher import compute_signals
from preprocess import build_phrase, detect_antonym_mismatch
from signals import detect_order_mismatch
from tune_backend import (
    EMBEDDING_BACKENDS,
    EMBEDDING_DEFAULTS,
    cross_validate_backend_thresholds,
    score_gold_with_backend,
    search_backend_thresholds,
    write_backend_thresholds,
)

CV_FOLDS: int = 5
SEED: int = 42
N_PARAMS_THRESHOLDS: int = 2
MIN_THRESHOLD_GAP: float = 0.05
GRID_STEP: float = 0.02
CONFIG_PATH: Path = Path(__file__).resolve().parent / "config.json"


@dataclass(frozen=True)
class CachedPair:
    """Gold pair with precomputed 6 signals + 3 gate flags."""
    pair: GoldPair
    signals: dict[str, float]
    negation_mismatch: bool
    antonym_mismatch: bool
    order_mismatch: bool


def cache_features(gold: Sequence[GoldPair]) -> list[CachedPair]:
    """Precompute signals + gate flags for every gold pair."""
    all_phrases = sorted({p.phrase_a for p in gold} | {p.phrase_b for p in gold})
    index = PhraseIndex(all_phrases)
    out: list[CachedPair] = []
    for p in gold:
        a = build_phrase(p.phrase_a)
        b = build_phrase(p.phrase_b)
        out.append(CachedPair(
            pair=p,
            signals=compute_signals(p.phrase_a, p.phrase_b, index, a, b),
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


def macro_f1(metrics: dict[str, "object"]) -> float:
    """Mean F1 across labels."""
    return sum(m.f1 for m in metrics.values()) / len(metrics)


def _macro_f1(results: list[PairResult]) -> float:
    return macro_f1(_per_label_metrics(results))


def params_to_thresholds(params: Sequence[float]) -> dict[str, float]:
    """Decode a 2-element params vector into {low, high} thresholds, ordered."""
    low, high = sorted((float(params[0]), float(params[1])))
    return {"low": low, "high": high}


def threshold_objective(params: Sequence[float], cache: list[CachedPair]) -> float:
    """Macro F1 with fixed default weights + penalties, only thresholds vary."""
    thresholds = params_to_thresholds(params)
    if thresholds["high"] - thresholds["low"] < MIN_THRESHOLD_GAP:
        return 0.0
    return _macro_f1(_predict(cache, DEFAULT_WEIGHTS, thresholds, DEFAULT_PENALTIES))


def _grid_axis(step: float) -> np.ndarray:
    n = int(round(1.0 / step)) + 1
    return np.linspace(0.0, 1.0, n)


def search_thresholds(
    cache: list[CachedPair],
    step: float = GRID_STEP,
) -> tuple[float, np.ndarray]:
    """Exhaustive grid over (low, high) with low + MIN_THRESHOLD_GAP <= high."""
    axis = _grid_axis(step)
    best_f1 = -1.0
    best_low, best_high = float(DEFAULT_THRESHOLDS["low"]), float(DEFAULT_THRESHOLDS["high"])
    for low in axis:
        for high in axis:
            if high - low < MIN_THRESHOLD_GAP:
                continue
            f1 = _macro_f1(_predict(
                cache, DEFAULT_WEIGHTS,
                {"low": float(low), "high": float(high)},
                DEFAULT_PENALTIES,
            ))
            if f1 > best_f1:
                best_f1 = f1
                best_low, best_high = float(low), float(high)
    return best_f1, np.array([best_low, best_high])


def cross_validate_thresholds(
    cache: list[CachedPair],
    folds: int = CV_FOLDS,
    seed: int = SEED,
    step: float = GRID_STEP,
) -> tuple[float, float, dict[str, float]]:
    """Stratified k fold CV. Returns (tuned_mean_macro, default_mean_macro, tuned_per_label)."""
    labels = np.array([c.pair.label for c in cache])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    tuned_macros: list[float] = []
    default_macros: list[float] = []
    fold_label_f1: dict[str, list[float]] = {label: [] for label in LABELS}
    for train_idx, test_idx in splitter.split(np.zeros(len(cache)), labels):
        train = [cache[i] for i in train_idx]
        test = [cache[i] for i in test_idx]
        _, params = search_thresholds(train, step=step)
        thresholds = params_to_thresholds(params)
        tuned_results = _predict(test, DEFAULT_WEIGHTS, thresholds, DEFAULT_PENALTIES)
        default_results = _predict(test, DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS, DEFAULT_PENALTIES)
        tuned_metrics = _per_label_metrics(tuned_results)
        default_metrics = _per_label_metrics(default_results)
        tuned_macros.append(macro_f1(tuned_metrics))
        default_macros.append(macro_f1(default_metrics))
        for label, m in tuned_metrics.items():
            fold_label_f1[label].append(m.f1)
    tuned_mean = sum(tuned_macros) / len(tuned_macros)
    default_mean = sum(default_macros) / len(default_macros)
    mean_per_label = {label: sum(fs) / len(fs) for label, fs in fold_label_f1.items()}
    return tuned_mean, default_mean, mean_per_label


def write_config(
    weights: dict, thresholds: dict, penalties: dict,
    path: str | Path = CONFIG_PATH,
) -> None:
    """Serialize the classical block into config.json, preserving other backend blocks."""
    write_backend_block(
        "classical",
        {"weights": weights, "thresholds": thresholds, "penalties": penalties},
        path,
    )


def should_adopt(candidate_cv_macro: float, baseline_cv_macro: float) -> bool:
    """Adopt the tuned config only if it strictly beats the baseline on CV."""
    return candidate_cv_macro > baseline_cv_macro


def _print_label_metrics(per_label: dict) -> None:
    print(f"  {'label':<10} {'F1':>6}")
    for label, value in per_label.items():
        f1 = value.f1 if hasattr(value, "f1") else value
        print(f"  {label:<10} {f1:>6.3f}")


def _run_classical(gold: list[GoldPair]) -> None:
    cache = cache_features(gold)
    print("=== classical threshold grid search, 5 fold CV ===")
    tuned_cv, default_cv, tuned_per_label = cross_validate_thresholds(cache)
    print(f"Tuned CV macro F1:   {tuned_cv:.3f}")
    print(f"Default CV macro F1: {default_cv:.3f}")
    _print_label_metrics(tuned_per_label)
    print("\n=== adoption decision ===")
    if should_adopt(tuned_cv, default_cv):
        in_sample_macro, params = search_thresholds(cache)
        thresholds = params_to_thresholds(params)
        write_config(DEFAULT_WEIGHTS, thresholds, DEFAULT_PENALTIES)
        print(f"ADOPTED: tuned CV {tuned_cv:.3f} > default CV {default_cv:.3f}")
        print(f"In sample macro F1 after fitting on all {len(cache)}: {in_sample_macro:.3f}")
        print(f"Tuned thresholds: low={thresholds['low']:.3f} high={thresholds['high']:.3f}")
        print(f"Classical block written to {CONFIG_PATH}")
    else:
        print(f"REJECTED: tuned CV {tuned_cv:.3f} <= default CV {default_cv:.3f}")
        print(f"Classical block in {CONFIG_PATH} (if any) left untouched")


def _run_embedding_backend(backend_name: str, gold: list[GoldPair]) -> None:
    default = EMBEDDING_DEFAULTS[backend_name]
    print(f"=== {backend_name} threshold grid search, 5 fold CV ===")
    scored = score_gold_with_backend(backend_name, gold)
    tuned_cv, default_cv, tuned_per_label = cross_validate_backend_thresholds(
        scored, default, folds=CV_FOLDS, seed=SEED, step=GRID_STEP, min_gap=MIN_THRESHOLD_GAP,
    )
    print(f"Tuned CV macro F1:   {tuned_cv:.3f}")
    print(f"Default CV macro F1: {default_cv:.3f}")
    _print_label_metrics(tuned_per_label)
    print("\n=== adoption decision ===")
    if should_adopt(tuned_cv, default_cv):
        in_sample_macro, (low, high) = search_backend_thresholds(
            scored, step=GRID_STEP, min_gap=MIN_THRESHOLD_GAP,
        )
        write_backend_thresholds(backend_name, low, high, CONFIG_PATH)
        print(f"ADOPTED: tuned CV {tuned_cv:.3f} > default CV {default_cv:.3f}")
        print(f"In sample macro F1: {in_sample_macro:.3f}")
        print(f"Tuned thresholds: low={low:.3f} high={high:.3f}")
        print(f"{backend_name} block written to {CONFIG_PATH}")
    else:
        print(f"REJECTED: tuned CV {tuned_cv:.3f} <= default CV {default_cv:.3f}")
        print(f"{backend_name} block in {CONFIG_PATH} (if any) left untouched")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune thresholds per backend on the gold set.")
    parser.add_argument(
        "--backend", default="classical",
        choices=("classical",) + EMBEDDING_BACKENDS,
        help="which backend to tune",
    )
    parser.add_argument(
        "--gold", default="data/gold_pairs.json",
        help="path to the gold set JSON",
    )
    args = parser.parse_args(argv)
    gold = load_gold(args.gold)
    if args.backend == "classical":
        _run_classical(gold)
    else:
        _run_embedding_backend(args.backend, gold)


if __name__ == "__main__":
    main()
