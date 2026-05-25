"""Tune (low, high) thresholds for an embedding backend against the gold set."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from backends import get_backend
from backends.gpt import GPT_HIGH, GPT_LOW
from backends.use import USE_HIGH, USE_LOW
from combiner import write_backend_block
from evaluate import LABELS, GoldPair, PairResult, _per_label_metrics

EMBEDDING_BACKENDS: tuple[str, ...] = ("use", "gpt")
EMBEDDING_DEFAULTS: dict[str, tuple[float, float]] = {
    "use": (USE_LOW, USE_HIGH),
    "gpt": (GPT_LOW, GPT_HIGH),
}


@dataclass(frozen=True)
class BackendScoredPair:
    """Gold pair plus its backend score, cached so grid search reuses one pass."""
    pair: GoldPair
    score: float


def score_gold_with_backend(
    backend_name: str, gold: Sequence[GoldPair],
) -> list[BackendScoredPair]:
    """Build the backend over the gold corpus and score every pair once."""
    corpus = sorted({p.phrase_a for p in gold} | {p.phrase_b for p in gold})
    backend = get_backend(backend_name, corpus)
    return [
        BackendScoredPair(pair=p, score=backend.score_pair(p.phrase_a, p.phrase_b))
        for p in gold
    ]


def _label_from_thresholds(score: float, low: float, high: float) -> str:
    if score >= high:
        return "MATCH"
    if score >= low:
        return "PARTIAL"
    return "NO_MATCH"


def _backend_predict(
    scored: Sequence[BackendScoredPair], low: float, high: float,
) -> list[PairResult]:
    return [
        PairResult(
            pair_id=s.pair.id, category=s.pair.category, phenomenon=s.pair.phenomenon,
            predicted_score=s.score, gold_score=s.pair.gold_score,
            predicted_label=_label_from_thresholds(s.score, low, high),
            gold_label=s.pair.label,
        )
        for s in scored
    ]


def _macro_f1(results: list[PairResult]) -> float:
    metrics = _per_label_metrics(results)
    return sum(m.f1 for m in metrics.values()) / len(metrics)


def search_backend_thresholds(
    scored: Sequence[BackendScoredPair],
    step: float,
    min_gap: float,
) -> tuple[float, tuple[float, float]]:
    """Exhaustive grid over (low, high) for a backend score cache, macro F1 objective."""
    n = int(round(1.0 / step)) + 1
    axis = np.linspace(0.0, 1.0, n)
    best_f1 = -1.0
    best_low, best_high = 0.0, 1.0
    for low in axis:
        for high in axis:
            if high - low < min_gap:
                continue
            f1 = _macro_f1(_backend_predict(scored, float(low), float(high)))
            if f1 > best_f1:
                best_f1 = f1
                best_low, best_high = float(low), float(high)
    return best_f1, (best_low, best_high)


def cross_validate_backend_thresholds(
    scored: Sequence[BackendScoredPair],
    default: tuple[float, float],
    folds: int,
    seed: int,
    step: float,
    min_gap: float,
) -> tuple[float, float, dict[str, float]]:
    """Stratified CV over backend scores. Returns (tuned_mean, default_mean, tuned_per_label)."""
    labels = np.array([s.pair.label for s in scored])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    tuned_macros: list[float] = []
    default_macros: list[float] = []
    fold_label_f1: dict[str, list[float]] = {label: [] for label in LABELS}
    for train_idx, test_idx in splitter.split(np.zeros(len(scored)), labels):
        train = [scored[i] for i in train_idx]
        test = [scored[i] for i in test_idx]
        _, (low, high) = search_backend_thresholds(train, step=step, min_gap=min_gap)
        tuned_metrics = _per_label_metrics(_backend_predict(test, low, high))
        default_metrics = _per_label_metrics(_backend_predict(test, default[0], default[1]))
        tuned_macros.append(sum(m.f1 for m in tuned_metrics.values()) / len(tuned_metrics))
        default_macros.append(sum(m.f1 for m in default_metrics.values()) / len(default_metrics))
        for label, m in tuned_metrics.items():
            fold_label_f1[label].append(m.f1)
    tuned_mean = sum(tuned_macros) / len(tuned_macros)
    default_mean = sum(default_macros) / len(default_macros)
    mean_per_label = {label: sum(fs) / len(fs) for label, fs in fold_label_f1.items()}
    return tuned_mean, default_mean, mean_per_label


def write_backend_thresholds(
    name: str, low: float, high: float, path: str | Path,
) -> None:
    """Write the tuned thresholds for an embedding backend, preserving other blocks."""
    write_backend_block(name, {"thresholds": {"low": low, "high": high}}, path)
