"""Combine per-signal scores into a final score + label."""
from __future__ import annotations

import json
from pathlib import Path

SIGNAL_NAMES: tuple[str, ...] = ("tfidf", "jaccard", "wordnet", "ngram", "order")

DEFAULT_WEIGHTS: dict[str, float] = {name: 0.2 for name in SIGNAL_NAMES}
DEFAULT_THRESHOLDS: dict[str, float] = {"low": 0.4, "high": 0.7}

NEGATION_PENALTY: float = 0.3
ANTONYM_PENALTY: float = 0.3


def _label_for(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "MATCH"
    if score >= thresholds["low"]:
        return "PARTIAL"
    return "NO_MATCH"


def combine(
    signals: dict[str, float],
    negation_mismatch: bool,
    antonym_mismatch: bool = False,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[float, str]:
    """Weighted sum of signals -> clamped [0,1] score + label; penalize negation/antonym mismatch."""
    w = DEFAULT_WEIGHTS if weights is None else weights
    t = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    score = sum(w[name] * signals[name] for name in SIGNAL_NAMES)
    if negation_mismatch:
        score *= NEGATION_PENALTY
    if antonym_mismatch:
        score *= ANTONYM_PENALTY
    score = max(0.0, min(1.0, score))
    return score, _label_for(score, t)


def load_config(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    """Read config.json or return defaults."""
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_WEIGHTS), dict(DEFAULT_THRESHOLDS)
    data = json.loads(p.read_text())
    weights = {name: float(data["weights"][name]) for name in SIGNAL_NAMES}
    thresholds = {
        "low": float(data["thresholds"]["low"]),
        "high": float(data["thresholds"]["high"]),
    }
    return weights, thresholds
