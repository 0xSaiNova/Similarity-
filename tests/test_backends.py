"""Tests for backends: base interface, registry, classical equality regression."""
from __future__ import annotations

import pytest

from backends import Backend, BackendMatchResult, available, get_backend
from backends.classical import ClassicalBackend
from combiner import DEFAULT_PENALTIES, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, combine
from index import PhraseIndex
from matcher import compute_signals
from preprocess import build_phrase, detect_antonym_mismatch
from signals import detect_order_mismatch


CORPUS: list[str] = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "scale up the backend services",
    "scale down the backend services",
    "the system updates the cache",
    "the system does not update the cache",
    "the producer pushes messages to the consumer",
    "the consumer pushes messages to the producer",
]


def _direct_combine(corpus: list[str], a: str, b: str) -> tuple[float, str]:
    index = PhraseIndex(corpus)
    pa = build_phrase(a)
    pb = build_phrase(b)
    sigs = compute_signals(a, b, index, pa, pb)
    return combine(
        sigs,
        pa.has_negation != pb.has_negation,
        detect_antonym_mismatch(a, b),
        detect_order_mismatch(pa.tokens, pb.tokens),
        DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS, DEFAULT_PENALTIES,
    )


def test_available_lists_classical() -> None:
    assert "classical" in available()


def test_get_backend_returns_classical_instance() -> None:
    backend = get_backend("classical", CORPUS)
    assert isinstance(backend, ClassicalBackend)


def test_get_backend_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("does_not_exist", CORPUS)


def test_backend_requires_non_empty_corpus() -> None:
    with pytest.raises(ValueError):
        ClassicalBackend([])


def test_classical_thresholds_match_defaults() -> None:
    backend = ClassicalBackend(CORPUS)
    low, high = backend.thresholds
    assert low == DEFAULT_THRESHOLDS["low"]
    assert high == DEFAULT_THRESHOLDS["high"]


@pytest.mark.parametrize("pair", [
    ("the cat sat on the mat", "the cat sat on the mat"),
    ("scale up the backend services", "scale down the backend services"),
    ("the system updates the cache", "the system does not update the cache"),
    (
        "the producer pushes messages to the consumer",
        "the consumer pushes messages to the producer",
    ),
    ("the cat sat on the mat", "a dog ran in the park"),
    ("a dog ran in the park", "the cat sat on the mat"),
])
def test_classical_score_and_label_match_direct_engine(pair: tuple[str, str]) -> None:
    a, b = pair
    backend = ClassicalBackend(CORPUS)
    expected_score, expected_label = _direct_combine(CORPUS, a, b)
    score = backend.score_pair(a, b)
    assert score == expected_score
    assert backend.label(score) == expected_label


def test_classical_label_emits_expected_strings_around_thresholds() -> None:
    backend = ClassicalBackend(CORPUS)
    low = DEFAULT_THRESHOLDS["low"]
    high = DEFAULT_THRESHOLDS["high"]
    assert backend.label(0.0) == "NO_MATCH"
    assert backend.label(low - 1e-6) == "NO_MATCH"
    assert backend.label(low) == "PARTIAL"
    assert backend.label((low + high) / 2) == "PARTIAL"
    assert backend.label(high - 1e-6) == "PARTIAL"
    assert backend.label(high) == "MATCH"
    assert backend.label(1.0) == "MATCH"


def test_classical_score_with_explain_returns_score_and_six_signals() -> None:
    backend = ClassicalBackend(CORPUS)
    score, sigs = backend.score_with_explain(
        "the cat sat on the mat", "the cat sat on the mat",
    )
    assert 0.0 <= score <= 1.0
    assert sigs is not None
    assert set(sigs.keys()) == {"tfidf", "jaccard", "wordnet", "ngram", "order", "soft_overlap"}
    for v in sigs.values():
        assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9


def test_classical_score_with_explain_matches_score_pair() -> None:
    backend = ClassicalBackend(CORPUS)
    a, b = "the cat sat on the mat", "a dog ran in the park"
    only_score = backend.score_pair(a, b)
    score, _ = backend.score_with_explain(a, b)
    assert score == only_score


def test_classical_score_in_unit_interval() -> None:
    backend = ClassicalBackend(CORPUS)
    for a in CORPUS:
        for b in CORPUS:
            assert 0.0 <= backend.score_pair(a, b) <= 1.0


def test_classical_negation_demotes_score() -> None:
    backend = ClassicalBackend(CORPUS)
    same = backend.score_pair("the system updates the cache", "the system updates the cache")
    negated = backend.score_pair(
        "the system updates the cache", "the system does not update the cache",
    )
    assert negated < same


class _DummyBackend(Backend):
    """Minimal subclass used to exercise the base class defaults."""

    def score_pair(self, phrase_a: str, phrase_b: str) -> float:
        return 0.42

    @property
    def thresholds(self) -> tuple[float, float]:
        return 0.4, 0.7


def test_base_label_default_uses_thresholds() -> None:
    d = _DummyBackend(["x"])
    assert d.label(0.8) == "MATCH"
    assert d.label(0.5) == "PARTIAL"
    assert d.label(0.1) == "NO_MATCH"
    assert d.label(0.4) == "PARTIAL"
    assert d.label(0.7) == "MATCH"


def test_base_score_with_explain_default_returns_score_and_none() -> None:
    d = _DummyBackend(["x"])
    score, signals = d.score_with_explain("a", "b")
    assert score == 0.42
    assert signals is None


def test_backend_match_result_is_frozen_dataclass() -> None:
    r = BackendMatchResult(candidate="x", score=0.5, label="PARTIAL", signals=None)
    with pytest.raises(Exception):
        r.score = 0.9  # type: ignore[misc]


def test_use_backend_reads_thresholds_from_config_block(tmp_path) -> None:
    import json

    from backends.use import USE_HIGH, USE_LOW, UseBackend
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"use": {"thresholds": {"low": 0.42, "high": 0.81}}}))
    backend = UseBackend(CORPUS, config_path=cfg)
    assert backend.thresholds == (0.42, 0.81)
    # placeholder fallback only when block absent
    backend_fallback = UseBackend(CORPUS, config_path=tmp_path / "absent.json")
    assert backend_fallback.thresholds == (USE_LOW, USE_HIGH)


def test_gpt_backend_reads_thresholds_from_config_block(tmp_path, monkeypatch) -> None:
    import json

    from backends.gpt import GPT_HIGH, GPT_LOW, GptBackend
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"gpt": {"thresholds": {"low": 0.58, "high": 0.79}}}))
    backend = GptBackend(CORPUS, cache_path=tmp_path / "cache.json", config_path=cfg)
    assert backend.thresholds == (0.58, 0.79)
    backend_fallback = GptBackend(
        CORPUS, cache_path=tmp_path / "cache2.json", config_path=tmp_path / "absent.json",
    )
    assert backend_fallback.thresholds == (GPT_LOW, GPT_HIGH)


def test_classical_backend_falls_back_to_defaults_when_no_config(tmp_path) -> None:
    backend = ClassicalBackend(CORPUS, config_path=tmp_path / "absent.json")
    assert backend.thresholds == (DEFAULT_THRESHOLDS["low"], DEFAULT_THRESHOLDS["high"])


def test_classical_backend_loads_tuned_thresholds_from_classical_block(tmp_path) -> None:
    import json

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "classical": {
            "weights": DEFAULT_WEIGHTS,
            "thresholds": {"low": 0.33, "high": 0.66},
            "penalties": DEFAULT_PENALTIES,
        },
    }))
    backend = ClassicalBackend(CORPUS, config_path=cfg)
    assert backend.thresholds == (0.33, 0.66)


def test_classical_block_alone_does_not_change_embedding_backends(tmp_path) -> None:
    """Cross check: a tuned classical block must not flow into use/gpt thresholds."""
    import json

    from backends.use import USE_HIGH, USE_LOW, UseBackend
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "classical": {
            "weights": DEFAULT_WEIGHTS,
            "thresholds": {"low": 0.10, "high": 0.20},
            "penalties": DEFAULT_PENALTIES,
        },
    }))
    backend = UseBackend(CORPUS, config_path=cfg)
    assert backend.thresholds == (USE_LOW, USE_HIGH)
