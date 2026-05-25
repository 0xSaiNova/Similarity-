"""Tune combiner thresholds on the gold set; adopt only if CV beats default."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from combiner import DEFAULT_PENALTIES, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, combine
from evaluate import LABELS, GoldPair, PairResult, _per_label_metrics, load_gold
from index import PhraseIndex
from matcher import compute_signals
from preprocess import build_phrase, detect_antonym_mismatch
from signals import detect_order_mismatch

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
    """Macro-F1 with fixed default weights + penalties, only thresholds vary."""
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
    """Stratified k-fold CV. Returns (tuned_mean_macro, default_mean_macro, tuned_per_label)."""
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
    """Serialize tuned config in the load_config schema."""
    data = {"weights": weights, "thresholds": thresholds, "penalties": penalties}
    Path(path).write_text(json.dumps(data, indent=2))


def should_adopt(candidate_cv_macro: float, baseline_cv_macro: float) -> bool:
    """Adopt the tuned config only if it strictly beats the baseline on CV."""
    return candidate_cv_macro > baseline_cv_macro


def _print_label_metrics(per_label: dict) -> None:
    print(f"  {'label':<10} {'F1':>6}")
    for label, value in per_label.items():
        f1 = value.f1 if hasattr(value, "f1") else value
        print(f"  {label:<10} {f1:>6.3f}")


def main() -> None:
    gold = load_gold("data/gold_pairs.json")
    cache = cache_features(gold)

    print("=== Threshold grid search — 5-fold CV ===")
    tuned_cv, default_cv, tuned_per_label = cross_validate_thresholds(cache)
    print(f"Tuned CV macro F1:   {tuned_cv:.3f}")
    print(f"Default CV macro F1: {default_cv:.3f}")
    _print_label_metrics(tuned_per_label)

    print("\n=== Adoption decision ===")
    if should_adopt(tuned_cv, default_cv):
        in_sample_macro, params = search_thresholds(cache)
        thresholds = params_to_thresholds(params)
        write_config(DEFAULT_WEIGHTS, thresholds, DEFAULT_PENALTIES)
        print(f"ADOPTED: tuned CV {tuned_cv:.3f} > default CV {default_cv:.3f}")
        print(f"In-sample macro F1 after fitting on all {len(cache)}: {in_sample_macro:.3f}")
        print(f"Tuned thresholds: low={thresholds['low']:.3f} high={thresholds['high']:.3f}")
        print(f"Written to {CONFIG_PATH}")
    else:
        print(f"REJECTED: tuned CV {tuned_cv:.3f} <= default CV {default_cv:.3f}")
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
            print(f"Removed stale {CONFIG_PATH}; evaluate.py will use built-in defaults")
        else:
            print(f"No {CONFIG_PATH} present; evaluate.py already on defaults")


if __name__ == "__main__":
    main()
