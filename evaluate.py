"""Evaluate the matcher against the gold set."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backends import get_backend

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
class CategoryMetrics:
    """Metrics restricted to one gold category bucket."""
    count: int
    macro_f1: float
    binary_f1: float
    roc_auc: float


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
    per_category: dict[str, CategoryMetrics]


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


def _per_category_metrics(results: Sequence[PairResult]) -> dict[str, CategoryMetrics]:
    """Group results by gold category and compute macro F1 + binary F1 + ROC AUC for each."""
    by_category: dict[str, list[PairResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)
    out: dict[str, CategoryMetrics] = {}
    for category, items in by_category.items():
        macro = macro_f1(_per_label_metrics(items))
        binary = _binary_metrics(items).f1
        auc = _roc_auc(items)
        out[category] = CategoryMetrics(
            count=len(items), macro_f1=macro, binary_f1=binary, roc_auc=auc,
        )
    return out


def evaluate(gold: Sequence[GoldPair], backend: str = "classical") -> Report:
    """Score every gold pair through the selected backend and produce metrics."""
    if not gold:
        raise ValueError("evaluate requires at least one gold pair")
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
        per_category=_per_category_metrics(results),
    )


def compare_backends(
    gold: Sequence[GoldPair], names: Sequence[str],
) -> dict[str, Report | str]:
    """Run evaluate for each backend; return Report on success or an unavailable string on failure."""
    out: dict[str, Report | str] = {}
    for name in names:
        try:
            out[name] = evaluate(gold, backend=name)
        except (ImportError, RuntimeError, ValueError) as exc:
            out[name] = f"unavailable: {exc}"
    return out


DEFAULT_COMPARE_NAMES: tuple[str, ...] = ("classical", "use", "gpt")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: --backend X picks one backend; --compare runs every backend side by side."""
    import argparse

    from evaluate_report import format_comparison, format_report

    parser = argparse.ArgumentParser(description="Evaluate the matcher against the gold set.")
    parser.add_argument(
        "--gold", default="data/gold_pairs.json",
        help="path to the gold set JSON",
    )
    parser.add_argument(
        "--backend", default="classical",
        help="which backend to evaluate (ignored when --compare is set)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="run every default backend (classical, use, gpt) side by side",
    )
    args = parser.parse_args(argv)
    gold = load_gold(args.gold)
    if args.compare:
        reports = compare_backends(gold, names=DEFAULT_COMPARE_NAMES)
        print(format_comparison(reports))
    else:
        report = evaluate(gold, backend=args.backend)
        print(format_report(report))


if __name__ == "__main__":
    main()
