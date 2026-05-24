"""Evaluate the matcher against the gold set."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backends import get_backend
from index import PhraseIndex
from matcher import compute_signals
from preprocess import build_phrase, detect_antonym_mismatch
from signals import detect_order_mismatch

LABELS: tuple[str, ...] = ("MATCH", "PARTIAL", "NO_MATCH")
REQUIRED_FIELDS: tuple[str, ...] = (
    "id", "phrase_a", "phrase_b", "label", "gold_score", "category", "phenomenon", "rationale",
)


@dataclass(frozen=True)
class GoldPair:
    """One gold-labelled phrase pair."""
    id: int
    phrase_a: str
    phrase_b: str
    label: str
    gold_score: float
    category: str
    phenomenon: str
    rationale: str


@dataclass(frozen=True)
class PairResult:
    """Predicted vs gold for one pair."""
    pair_id: int
    category: str
    phenomenon: str
    predicted_score: float
    gold_score: float
    predicted_label: str
    gold_label: str


@dataclass(frozen=True)
class LabelMetrics:
    """Precision, recall, F1 for one label."""
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BinaryMetrics:
    """Precision, recall, F1 for the collapsed CANDIDATE vs NO_MATCH problem."""
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class Report:
    """Full evaluation report."""
    total: int
    per_label: dict[str, LabelMetrics]
    macro_f1: float
    confusion: dict[str, dict[str, int]]
    mae: float
    worst: list[PairResult]
    binary: BinaryMetrics
    roc_auc: float


def _validate_pair(raw: dict, line: int) -> GoldPair:
    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(f"pair #{line}: missing field '{field}'")
    if not isinstance(raw["id"], int):
        raise ValueError(f"pair #{line}: 'id' must be int")
    if not isinstance(raw["phrase_a"], str) or not isinstance(raw["phrase_b"], str):
        raise ValueError(f"pair #{line}: 'phrase_a' and 'phrase_b' must be strings")
    if raw["label"] not in LABELS:
        raise ValueError(f"pair #{line}: 'label' must be one of {LABELS}")
    if not isinstance(raw["gold_score"], (int, float)):
        raise ValueError(f"pair #{line}: 'gold_score' must be numeric")
    score = float(raw["gold_score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"pair #{line}: 'gold_score' must be in [0, 1]")
    return GoldPair(
        id=raw["id"],
        phrase_a=raw["phrase_a"],
        phrase_b=raw["phrase_b"],
        label=raw["label"],
        gold_score=score,
        category=raw["category"],
        phenomenon=raw["phenomenon"],
        rationale=raw["rationale"],
    )


def load_gold(path: str | Path) -> list[GoldPair]:
    """Load and validate the gold pair file. Fails loudly on malformed input."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("gold file must be a JSON list")
    return [_validate_pair(item, i + 1) for i, item in enumerate(raw)]


def build_cache(gold: Sequence[GoldPair], factory: Callable[..., Any]) -> list[Any]:
    """Precompute (signals, gate flags) for every gold pair, wrap via factory(pair=, signals=, ...)."""
    all_phrases = sorted({p.phrase_a for p in gold} | {p.phrase_b for p in gold})
    index = PhraseIndex(all_phrases)
    out: list[Any] = []
    for p in gold:
        a = build_phrase(p.phrase_a)
        b = build_phrase(p.phrase_b)
        out.append(factory(
            pair=p,
            signals=compute_signals(p.phrase_a, p.phrase_b, index, a, b),
            negation_mismatch=a.has_negation != b.has_negation,
            antonym_mismatch=detect_antonym_mismatch(p.phrase_a, p.phrase_b),
            order_mismatch=detect_order_mismatch(a.tokens, b.tokens),
        ))
    return out


def macro_f1(metrics: dict[str, LabelMetrics]) -> float:
    """Mean F1 across labels."""
    return sum(m.f1 for m in metrics.values()) / len(metrics)


def _per_label_metrics(results: Sequence[PairResult]) -> dict[str, LabelMetrics]:
    out: dict[str, LabelMetrics] = {}
    for label in LABELS:
        tp = sum(1 for r in results if r.predicted_label == label and r.gold_label == label)
        fp = sum(1 for r in results if r.predicted_label == label and r.gold_label != label)
        fn = sum(1 for r in results if r.predicted_label != label and r.gold_label == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = LabelMetrics(precision, recall, f1)
    return out


def _confusion(results: Sequence[PairResult]) -> dict[str, dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for r in results:
        matrix[r.gold_label][r.predicted_label] += 1
    return matrix


def _is_candidate(label: str) -> bool:
    return label != "NO_MATCH"


def _binary_metrics(results: Sequence[PairResult]) -> BinaryMetrics:
    tp = sum(1 for r in results if _is_candidate(r.predicted_label) and _is_candidate(r.gold_label))
    fp = sum(1 for r in results if _is_candidate(r.predicted_label) and not _is_candidate(r.gold_label))
    fn = sum(1 for r in results if not _is_candidate(r.predicted_label) and _is_candidate(r.gold_label))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(precision, recall, f1)


def _roc_auc(results: Sequence[PairResult]) -> float:
    """ROC AUC of predicted_score for CANDIDATE vs NO_MATCH; NaN if only one class present."""
    from sklearn.metrics import roc_auc_score
    y_true = [1 if _is_candidate(r.gold_label) else 0 for r in results]
    y_score = [r.predicted_score for r in results]
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate(gold: Sequence[GoldPair], backend: str = "classical") -> Report:
    """Score every gold pair through the selected backend and produce metrics."""
    if not gold:
        raise ValueError("evaluate requires at least one gold pair")
    # TODO(E5): re-introduce per-backend config loading (thresholds/weights/penalties).
    # The previous load_config("config.json") path was dropped when the backend
    # abstraction landed; classical thresholds are hard-coded defaults for now.
    corpus = sorted({p.phrase_a for p in gold} | {p.phrase_b for p in gold})
    bk = get_backend(backend, corpus)
    results: list[PairResult] = []
    for p in gold:
        score = bk.score_pair(p.phrase_a, p.phrase_b)
        results.append(PairResult(
            pair_id=p.id, category=p.category, phenomenon=p.phenomenon,
            predicted_score=score, gold_score=p.gold_score,
            predicted_label=bk.label(score), gold_label=p.label,
        ))
    per_label = _per_label_metrics(results)
    mae = sum(abs(r.predicted_score - r.gold_score) for r in results) / len(results)
    worst = sorted(results, key=lambda r: abs(r.predicted_score - r.gold_score), reverse=True)[:10]
    return Report(
        total=len(results),
        per_label=per_label,
        macro_f1=macro_f1(per_label),
        confusion=_confusion(results),
        mae=mae,
        worst=worst,
        binary=_binary_metrics(results),
        roc_auc=_roc_auc(results),
    )


def format_report(report: Report) -> str:
    """Format a Report for printing."""
    lines: list[str] = []
    lines.append(f"Total pairs: {report.total}")
    lines.append("")
    lines.append("Per label:")
    lines.append(f"  {'label':<10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for label, m in report.per_label.items():
        lines.append(f"  {label:<10} {m.precision:>10.3f} {m.recall:>10.3f} {m.f1:>10.3f}")
    lines.append(f"Macro F1: {report.macro_f1:.3f}")
    lines.append(f"Mean absolute error (score): {report.mae:.3f}")
    lines.append("")
    lines.append("Confusion matrix (rows=gold, cols=predicted):")
    header = "  " + "".join(f"{lab:>10}" for lab in LABELS)
    lines.append(header)
    for gold in LABELS:
        row = f"  {gold:<10}" + "".join(f"{report.confusion[gold][pred]:>10}" for pred in LABELS)
        lines.append(row)
    lines.append("")
    lines.append("Binary view (CANDIDATE = MATCH or PARTIAL vs NO_MATCH):")
    b = report.binary
    lines.append(f"  precision {b.precision:.3f}   recall {b.recall:.3f}   f1 {b.f1:.3f}")
    lines.append(f"  ROC AUC of raw score: {report.roc_auc:.3f}")
    lines.append("")
    lines.append("Worst 10 mismatches:")
    lines.append(f"  {'id':>4} {'category':<28} {'phenomenon':<22} {'pred':>6} {'gold':>6} {'pred_label':<10} {'gold_label':<10}")
    for r in report.worst:
        lines.append(
            f"  {r.pair_id:>4} {r.category:<28} {r.phenomenon:<22} "
            f"{r.predicted_score:>6.3f} {r.gold_score:>6.3f} "
            f"{r.predicted_label:<10} {r.gold_label:<10}"
        )
    return "\n".join(lines)


def main(gold_path: str | Path = "data/gold_pairs.json", backend: str = "classical") -> None:
    """Load the gold set, evaluate via the selected backend, print a readable report."""
    gold = load_gold(gold_path)
    report = evaluate(gold, backend=backend)
    print(format_report(report))


if __name__ == "__main__":
    main()
