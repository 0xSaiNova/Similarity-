"""Tests for tune_backend.py: naturalistic subset filter + score scaffolding."""
from __future__ import annotations

from evaluate import GoldPair
from tune_backend import ADVERSARIAL_CATEGORIES, naturalistic_subset


def _gp(pair_id: int, category: str) -> GoldPair:
    return GoldPair(
        id=pair_id, phrase_a="x", phrase_b="y",
        label="MATCH", gold_score=0.5, category=category, phenomenon="p", rationale="",
    )


def test_adversarial_categories_lists_the_three_lexical_overlap_traps() -> None:
    assert set(ADVERSARIAL_CATEGORIES) == {
        "antonym", "negation", "high_overlap_different_meaning",
    }


def test_naturalistic_subset_drops_every_adversarial_category() -> None:
    gold = [
        _gp(1, "synonym_positive"),
        _gp(2, "antonym"),
        _gp(3, "easy_negative"),
        _gp(4, "negation"),
        _gp(5, "paraphrase_positive"),
        _gp(6, "high_overlap_different_meaning"),
        _gp(7, "partial"),
    ]
    keep = naturalistic_subset(gold)
    assert [p.id for p in keep] == [1, 3, 5, 7]
    assert all(p.category not in ADVERSARIAL_CATEGORIES for p in keep)


def test_naturalistic_subset_preserves_order() -> None:
    gold = [_gp(10, "synonym_positive"), _gp(11, "easy_positive"), _gp(12, "partial")]
    assert naturalistic_subset(gold) == gold


def test_naturalistic_subset_on_real_gold_drops_24_of_100() -> None:
    from pathlib import Path
    from evaluate import load_gold
    gold = load_gold(Path(__file__).resolve().parent.parent / "data" / "gold_pairs.json")
    keep = naturalistic_subset(gold)
    assert len(gold) == 100
    assert len(keep) == 100 - 8 - 8 - 8  # antonym + negation + high_overlap_different_meaning
