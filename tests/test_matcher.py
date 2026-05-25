"""Tests for the compute_signals helper that lives in matcher.py."""
from __future__ import annotations

import pytest

from combiner import SIGNAL_NAMES
from index import PhraseIndex
from matcher import compute_signals


@pytest.fixture
def candidates() -> list[str]:
    return [
        "the cat sat on the mat",
        "a dog ran in the park",
        "birds fly south for winter",
        "the cat jumped over the fence",
    ]


def test_compute_signals_returns_all_six(candidates: list[str]) -> None:
    index = PhraseIndex(candidates)
    signals = compute_signals("the cat sat on the mat", "the cat sat on the mat", index)
    assert set(signals.keys()) == set(SIGNAL_NAMES)
    assert len(signals) == 6
    for value in signals.values():
        assert -1e-9 <= value <= 1.0 + 1e-9


def test_compute_signals_identical_phrase_is_high(candidates: list[str]) -> None:
    index = PhraseIndex(candidates)
    signals = compute_signals("the cat sat on the mat", "the cat sat on the mat", index)
    assert signals["tfidf"] == pytest.approx(1.0)
    assert signals["jaccard"] == pytest.approx(1.0)
    assert signals["order"] == pytest.approx(1.0)
