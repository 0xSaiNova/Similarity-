"""Tests for the evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluate import (
    LABELS,
    BinaryMetrics,
    CategoryMetrics,
    GoldPair,
    LabelMetrics,
    PairResult,
    Report,
    _binary_metrics,
    _confusion,
    _per_category_metrics,
    _per_label_metrics,
    _roc_auc,
    compare_backends,
    evaluate,
    load_gold,
)

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_pairs.json"


def _valid_pair(**overrides) -> dict:
    base = {
        "id": 1,
        "phrase_a": "deploy the service",
        "phrase_b": "ship the service",
        "label": "MATCH",
        "gold_score": 0.95,
        "category": "easy_positive",
        "phenomenon": "synonym_substitution",
        "rationale": "deploy and ship are synonyms",
    }
    base.update(overrides)
    return base


def test_load_gold_accepts_well_formed_file(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([_valid_pair(), _valid_pair(id=2, label="NO_MATCH", gold_score=0.05)]))
    pairs = load_gold(path)
    assert len(pairs) == 2
    assert all(isinstance(p, GoldPair) for p in pairs)


def test_load_gold_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"id": 1}))
    with pytest.raises(ValueError, match="JSON list"):
        load_gold(path)


def test_load_gold_rejects_missing_field(tmp_path: Path) -> None:
    bad = _valid_pair()
    del bad["label"]
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([bad]))
    with pytest.raises(ValueError, match="missing field 'label'"):
        load_gold(path)


def test_load_gold_rejects_bad_label(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([_valid_pair(label="MAYBE")]))
    with pytest.raises(ValueError, match="'label' must be one of"):
        load_gold(path)


def test_load_gold_rejects_out_of_range_score(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([_valid_pair(gold_score=1.5)]))
    with pytest.raises(ValueError, match="must be in"):
        load_gold(path)


def test_load_gold_rejects_non_int_id(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([_valid_pair(id="one")]))
    with pytest.raises(ValueError, match="'id' must be int"):
        load_gold(path)


def _result(predicted: str, gold: str, pred_score: float = 0.5, gold_score: float = 0.5) -> PairResult:
    return PairResult(
        pair_id=0,
        category="test",
        phenomenon="test",
        predicted_score=pred_score,
        gold_score=gold_score,
        predicted_label=predicted,
        gold_label=gold,
    )


def test_per_label_metrics_perfect_predictions() -> None:
    results = [_result("MATCH", "MATCH"), _result("PARTIAL", "PARTIAL"), _result("NO_MATCH", "NO_MATCH")]
    metrics = _per_label_metrics(results)
    for label in LABELS:
        assert metrics[label].precision == 1.0
        assert metrics[label].recall == 1.0
        assert metrics[label].f1 == 1.0


def test_per_label_metrics_match_class_half_recall() -> None:
    # 1 true positive, 1 false negative, 0 false positives for MATCH
    results = [_result("MATCH", "MATCH"), _result("NO_MATCH", "MATCH")]
    metrics = _per_label_metrics(results)
    assert metrics["MATCH"].precision == 1.0
    assert metrics["MATCH"].recall == 0.5
    assert metrics["MATCH"].f1 == pytest.approx(2 / 3)


def test_per_label_metrics_zero_when_no_predictions() -> None:
    # nothing predicted as PARTIAL; precision and recall both zero, f1 zero
    results = [_result("MATCH", "PARTIAL")]
    metrics = _per_label_metrics(results)
    assert metrics["PARTIAL"].precision == 0.0
    assert metrics["PARTIAL"].recall == 0.0
    assert metrics["PARTIAL"].f1 == 0.0


def test_confusion_counts_correctly() -> None:
    results = [
        _result("MATCH", "MATCH"),
        _result("MATCH", "PARTIAL"),
        _result("NO_MATCH", "NO_MATCH"),
        _result("NO_MATCH", "NO_MATCH"),
    ]
    matrix = _confusion(results)
    assert matrix["MATCH"]["MATCH"] == 1
    assert matrix["PARTIAL"]["MATCH"] == 1
    assert matrix["NO_MATCH"]["NO_MATCH"] == 2
    assert matrix["MATCH"]["PARTIAL"] == 0


def test_evaluate_runs_end_to_end_on_gold_file() -> None:
    gold = load_gold(GOLD_PATH)
    report = evaluate(gold)
    assert isinstance(report, Report)
    assert report.total == len(gold)
    assert set(report.per_label.keys()) == set(LABELS)
    for m in report.per_label.values():
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1 <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0
    assert 0.0 <= report.mae <= 1.0
    assert sum(sum(row.values()) for row in report.confusion.values()) == report.total
    assert len(report.worst) == 10


def test_evaluate_macro_f1_is_mean_of_per_label() -> None:
    gold = load_gold(GOLD_PATH)
    report = evaluate(gold)
    expected = sum(m.f1 for m in report.per_label.values()) / len(report.per_label)
    assert report.macro_f1 == pytest.approx(expected)


def test_evaluate_worst_sorted_by_score_delta_desc() -> None:
    gold = load_gold(GOLD_PATH)
    report = evaluate(gold)
    deltas = [abs(r.predicted_score - r.gold_score) for r in report.worst]
    assert deltas == sorted(deltas, reverse=True)


def test_evaluate_empty_gold_raises() -> None:
    with pytest.raises(ValueError):
        evaluate([])


def test_binary_metrics_perfect_predictions() -> None:
    # 1 TP (MATCH/MATCH), 1 TP (PARTIAL/PARTIAL), 1 TN (NO_MATCH/NO_MATCH)
    results = [
        _result("MATCH", "MATCH"),
        _result("PARTIAL", "PARTIAL"),
        _result("NO_MATCH", "NO_MATCH"),
    ]
    b = _binary_metrics(results)
    assert b.precision == 1.0
    assert b.recall == 1.0
    assert b.f1 == 1.0


def test_binary_metrics_collapses_match_and_partial() -> None:
    # predicted PARTIAL but gold MATCH counts as TP under collapse
    results = [
        _result("PARTIAL", "MATCH"),
        _result("MATCH", "PARTIAL"),
        _result("NO_MATCH", "NO_MATCH"),
    ]
    b = _binary_metrics(results)
    assert b.precision == 1.0
    assert b.recall == 1.0
    assert b.f1 == 1.0


def test_binary_metrics_counts_false_positives_and_negatives() -> None:
    results = [
        _result("MATCH", "NO_MATCH"),     # FP
        _result("NO_MATCH", "PARTIAL"),   # FN
        _result("PARTIAL", "MATCH"),      # TP
        _result("NO_MATCH", "NO_MATCH"),  # TN
    ]
    b = _binary_metrics(results)
    assert b.precision == pytest.approx(1 / 2)
    assert b.recall == pytest.approx(1 / 2)
    assert b.f1 == pytest.approx(0.5)


def test_roc_auc_perfect_ranking() -> None:
    # candidates score higher than non-candidates
    results = [
        _result("MATCH", "MATCH", pred_score=0.9),
        _result("PARTIAL", "PARTIAL", pred_score=0.7),
        _result("NO_MATCH", "NO_MATCH", pred_score=0.2),
        _result("NO_MATCH", "NO_MATCH", pred_score=0.1),
    ]
    assert _roc_auc(results) == pytest.approx(1.0)


def test_roc_auc_random_ranking_near_half() -> None:
    # interleaved scores -> AUC around 0.5
    results = [
        _result("MATCH", "MATCH", pred_score=0.3),
        _result("NO_MATCH", "NO_MATCH", pred_score=0.7),
        _result("MATCH", "MATCH", pred_score=0.4),
        _result("NO_MATCH", "NO_MATCH", pred_score=0.6),
    ]
    assert _roc_auc(results) == pytest.approx(0.0)  # perfectly inverted is also informative


def test_roc_auc_single_class_returns_nan() -> None:
    import math
    results = [_result("MATCH", "MATCH"), _result("PARTIAL", "PARTIAL")]
    auc = _roc_auc(results)
    assert math.isnan(auc)


def test_evaluate_includes_binary_and_auc() -> None:
    gold = load_gold(GOLD_PATH)
    report = evaluate(gold)
    assert isinstance(report.binary, BinaryMetrics)
    assert 0.0 <= report.binary.precision <= 1.0
    assert 0.0 <= report.binary.recall <= 1.0
    assert 0.0 <= report.binary.f1 <= 1.0
    assert 0.0 <= report.roc_auc <= 1.0


def _result_in(category: str, pred: str, gold: str, *, pred_score: float = 0.5) -> PairResult:
    return PairResult(
        pair_id=1, category=category, phenomenon="x",
        predicted_score=pred_score, gold_score=0.5,
        predicted_label=pred, gold_label=gold,
    )


def test_per_category_metrics_groups_by_category() -> None:
    results = [
        _result_in("cat_a", "MATCH", "MATCH", pred_score=0.9),
        _result_in("cat_a", "NO_MATCH", "NO_MATCH", pred_score=0.1),
        _result_in("cat_b", "PARTIAL", "PARTIAL", pred_score=0.5),
    ]
    out = _per_category_metrics(results)
    assert set(out.keys()) == {"cat_a", "cat_b"}
    assert out["cat_a"].count == 2
    assert out["cat_b"].count == 1


def test_per_category_metrics_returns_category_metrics_with_all_fields() -> None:
    results = [
        _result_in("cat_a", "MATCH", "MATCH", pred_score=0.9),
        _result_in("cat_a", "NO_MATCH", "NO_MATCH", pred_score=0.1),
    ]
    out = _per_category_metrics(results)
    metrics = out["cat_a"]
    assert isinstance(metrics, CategoryMetrics)
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.binary_f1 <= 1.0


def test_evaluate_report_includes_per_category() -> None:
    gold = load_gold(GOLD_PATH)
    report = evaluate(gold)
    assert isinstance(report.per_category, dict)
    assert len(report.per_category) >= 1
    for cat_metrics in report.per_category.values():
        assert isinstance(cat_metrics, CategoryMetrics)
        assert cat_metrics.count >= 1


def test_compare_backends_returns_report_for_each_name() -> None:
    gold = load_gold(GOLD_PATH)
    out = compare_backends(gold, names=("classical",))
    assert set(out.keys()) == {"classical"}
    assert isinstance(out["classical"], Report)


def test_compare_backends_marks_unavailable_on_construction_error(monkeypatch) -> None:
    import evaluate as ev_mod
    real_get = ev_mod.get_backend
    def fake_get(name, corpus):
        if name == "use":
            raise ImportError("USE backend disabled for test")
        return real_get(name, corpus)
    monkeypatch.setattr(ev_mod, "get_backend", fake_get)
    gold = load_gold(GOLD_PATH)
    out = compare_backends(gold, names=("classical", "use"))
    assert isinstance(out["classical"], Report)
    assert isinstance(out["use"], str)
    assert "unavailable" in out["use"].lower()


def test_compare_backends_marks_unavailable_on_runtime_error(monkeypatch) -> None:
    import evaluate as ev_mod
    real_get = ev_mod.get_backend
    def fake_get(name, corpus):
        if name == "gpt":
            raise RuntimeError("OPENAI_API_KEY missing")
        return real_get(name, corpus)
    monkeypatch.setattr(ev_mod, "get_backend", fake_get)
    gold = load_gold(GOLD_PATH)
    out = compare_backends(gold, names=("gpt",))
    assert isinstance(out["gpt"], str)
    assert "OPENAI_API_KEY" in out["gpt"]
