"""Format Report and multi backend comparison output."""
from __future__ import annotations

from collections.abc import Mapping

from evaluate import LABELS, CategoryMetrics, Report


def _format_per_category(per_category: Mapping[str, CategoryMetrics]) -> list[str]:
    lines = ["Per category:"]
    lines.append(f"  {'category':<32} {'n':>4} {'macroF1':>9} {'binaryF1':>10} {'AUC':>7}")
    for category in sorted(per_category):
        m = per_category[category]
        auc_text = f"{m.roc_auc:>7.3f}" if m.roc_auc == m.roc_auc else f"{'n/a':>7}"
        lines.append(
            f"  {category:<32} {m.count:>4} "
            f"{m.macro_f1:>9.3f} {m.binary_f1:>10.3f} {auc_text}"
        )
    return lines


def format_report(report: Report) -> str:
    """Render a single backend Report for printing."""
    lines: list[str] = []
    lines.append(f"Total pairs: {report.total}")
    lines.append("")
    lines.append("Per label:")
    lines.append(f"  {'label':<10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for label, m in report.per_label.items():
        lines.append(
            f"  {label:<10} {m.precision:>10.3f} {m.recall:>10.3f} {m.f1:>10.3f}"
        )
    lines.append(f"Macro F1: {report.macro_f1:.3f}")
    lines.append(f"Mean absolute error (score): {report.mae:.3f}")
    lines.append("")
    lines.append("Confusion matrix (rows=gold, cols=predicted):")
    header = "  " + "".join(f"{lab:>10}" for lab in LABELS)
    lines.append(header)
    for gold in LABELS:
        row = f"  {gold:<10}" + "".join(
            f"{report.confusion[gold][pred]:>10}" for pred in LABELS
        )
        lines.append(row)
    lines.append("")
    lines.append("Binary view (CANDIDATE = MATCH or PARTIAL vs NO_MATCH):")
    b = report.binary
    lines.append(f"  precision {b.precision:.3f}   recall {b.recall:.3f}   f1 {b.f1:.3f}")
    lines.append(f"  ROC AUC of raw score: {report.roc_auc:.3f}")
    lines.append("")
    lines.extend(_format_per_category(report.per_category))
    lines.append("")
    lines.append("Worst 10 mismatches:")
    lines.append(
        f"  {'id':>4} {'category':<28} {'phenomenon':<22} "
        f"{'pred':>6} {'gold':>6} {'pred_label':<10} {'gold_label':<10}"
    )
    for r in report.worst:
        lines.append(
            f"  {r.pair_id:>4} {r.category:<28} {r.phenomenon:<22} "
            f"{r.predicted_score:>6.3f} {r.gold_score:>6.3f} "
            f"{r.predicted_label:<10} {r.gold_label:<10}"
        )
    return "\n".join(lines)


def _fmt_metric(value: float, width: int = 9) -> str:
    if value != value:  # nan
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.3f}"


def _gather_categories(reports: Mapping[str, Report | str]) -> list[str]:
    seen: set[str] = set()
    for entry in reports.values():
        if not isinstance(entry, str):
            seen.update(entry.per_category)
    return sorted(seen)


def format_comparison(reports: Mapping[str, Report | str]) -> str:
    """Side by side comparison of multiple backends with overall + per category breakdowns."""
    names = list(reports.keys())
    lines: list[str] = []
    lines.append("=== Overall ===")
    header_cells = [f"{n:>12}" for n in names]
    lines.append(f"  {'metric':<16}" + "".join(header_cells))
    for metric in ("macro_f1", "binary_f1", "roc_auc"):
        row_cells: list[str] = []
        for n in names:
            entry = reports[n]
            if not isinstance(entry, str):
                value = (
                    entry.binary.f1 if metric == "binary_f1"
                    else entry.roc_auc if metric == "roc_auc"
                    else entry.macro_f1
                )
                row_cells.append(_fmt_metric(value, width=12))
            else:
                row_cells.append(f"{'unavail':>12}")
        lines.append(f"  {metric:<16}" + "".join(row_cells))
    lines.append("")
    if any(not isinstance(entry, str) for entry in reports.values()):
        total_line = f"  {'n pairs':<16}" + "".join(
            f"{e.total:>12}" if not isinstance(e, str) else f"{'unavail':>12}"
            for e in reports.values()
        )
        lines.append(total_line)
    for unavail_name, entry in reports.items():
        if isinstance(entry, str):
            lines.append(f"  {unavail_name}: {entry}")
    lines.append("")
    categories = _gather_categories(reports)
    for metric_name, attr in (
        ("Macro F1 by category", "macro_f1"),
        ("Binary F1 by category", "binary_f1"),
        ("ROC AUC by category", "roc_auc"),
    ):
        lines.append(f"=== {metric_name} ===")
        lines.append(
            f"  {'category':<32} {'n':>4}" + "".join(f"{n:>12}" for n in names)
        )
        for category in categories:
            counts = [
                str(entry.per_category[category].count)
                if not isinstance(entry, str) and category in entry.per_category
                else "0"
                for entry in reports.values()
            ]
            n_col = max(counts, key=lambda c: int(c))
            row = f"  {category:<32} {n_col:>4}"
            for n in names:
                entry = reports[n]
                if not isinstance(entry, str) and category in entry.per_category:
                    row += _fmt_metric(getattr(entry.per_category[category], attr), width=12)
                else:
                    row += f"{'unavail':>12}"
            lines.append(row)
        lines.append("")
    return "\n".join(lines).rstrip()
