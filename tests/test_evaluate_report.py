"""Tests for evaluate_report formatting helpers."""
from __future__ import annotations

from pathlib import Path

from evaluate import compare_backends, evaluate, load_gold
from evaluate_report import format_comparison, format_report

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_pairs.json"


def test_format_report_includes_per_category_section() -> None:
    gold = load_gold(GOLD_PATH)
    report = evaluate(gold)
    text = format_report(report)
    assert "Per category:" in text
    for category in report.per_category:
        assert category in text


def test_format_report_includes_overall_metrics() -> None:
    gold = load_gold(GOLD_PATH)
    report = evaluate(gold)
    text = format_report(report)
    assert "Macro F1" in text
    assert "Binary view" in text
    assert "ROC AUC" in text
    assert "Confusion matrix" in text


def test_format_comparison_lists_each_backend_column() -> None:
    gold = load_gold(GOLD_PATH)
    reports = compare_backends(gold, names=("classical",))
    text = format_comparison(reports)
    assert "classical" in text
    assert "macro_f1" in text
    assert "Macro F1 by category" in text


def test_format_comparison_marks_unavailable_backend(monkeypatch) -> None:
    import evaluate as ev_mod
    real_get = ev_mod.get_backend

    def fake_get(name, corpus):
        if name == "gpt":
            raise RuntimeError("OPENAI_API_KEY missing")
        return real_get(name, corpus)

    monkeypatch.setattr(ev_mod, "get_backend", fake_get)
    gold = load_gold(GOLD_PATH)
    reports = compare_backends(gold, names=("classical", "gpt"))
    text = format_comparison(reports)
    assert "unavail" in text
    assert "OPENAI_API_KEY" in text
