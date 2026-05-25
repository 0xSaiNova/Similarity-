"""Combine per-signal scores into a final score + label using OR-logic across two groups."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

SURFACE_SIGNALS: tuple[str, ...] = ("tfidf", "jaccard", "ngram", "order")
SEMANTIC_SIGNALS: tuple[str, ...] = ("wordnet", "soft_overlap")
SIGNAL_NAMES: tuple[str, ...] = SURFACE_SIGNALS + SEMANTIC_SIGNALS

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "surface": {name: 1.0 / len(SURFACE_SIGNALS) for name in SURFACE_SIGNALS},
    "semantic": {name: 1.0 / len(SEMANTIC_SIGNALS) for name in SEMANTIC_SIGNALS},
}
DEFAULT_THRESHOLDS: dict[str, float] = {"low": 0.4, "high": 0.7}
DEFAULT_PENALTIES: dict[str, float] = {"negation": 0.3, "antonym": 0.3, "order": 0.5}


def label_for(score: float, thresholds: dict[str, float]) -> str:
    """Map a score to MATCH / PARTIAL / NO_MATCH using the provided thresholds."""
    if score >= thresholds["high"]:
        return "MATCH"
    if score >= thresholds["low"]:
        return "PARTIAL"
    return "NO_MATCH"


def combine(
    signals: dict[str, float],
    negation_mismatch: bool,
    antonym_mismatch: bool = False,
    order_mismatch: bool = False,
    weights: dict[str, dict[str, float]] | None = None,
    thresholds: dict[str, float] | None = None,
    penalties: dict[str, float] | None = None,
) -> tuple[float, str]:
    """Max of surface vs semantic weighted sums, with negation/antonym/order gates."""
    w = DEFAULT_WEIGHTS if weights is None else weights
    t = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    p = DEFAULT_PENALTIES if penalties is None else penalties
    surface_score = sum(w["surface"][n] * signals[n] for n in SURFACE_SIGNALS)
    semantic_score = sum(w["semantic"][n] * signals[n] for n in SEMANTIC_SIGNALS)
    score = max(surface_score, semantic_score)
    if negation_mismatch:
        score *= p["negation"]
    if antonym_mismatch:
        score *= p["antonym"]
    if order_mismatch:
        score *= p["order"]
    score = max(0.0, min(1.0, score))
    return score, label_for(score, t)


WEIGHT_SUM_TOL: float = 1e-6


def _validate_weight_group(weights: dict[str, float], group: str) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(
            f"config weights['{group}'] must sum to 1.0 (got {total:.6f})"
        )


def load_backend_block(name: str, path: str | Path) -> dict | None:
    """Return data[name] from a per backend config.json, or None when not configured."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get(name)


def write_backend_block(name: str, block: dict, path: str | Path) -> None:
    """Merge block into config.json under the given backend key, preserving siblings."""
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[name] = block
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_config(
    path: str | Path,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    """Read the classical block from config.json or return defaults. Returns (weights, thresholds, penalties)."""
    block = load_backend_block("classical", path)
    if block is None:
        return deepcopy(DEFAULT_WEIGHTS), dict(DEFAULT_THRESHOLDS), dict(DEFAULT_PENALTIES)
    weights = {
        "surface": {name: float(block["weights"]["surface"][name]) for name in SURFACE_SIGNALS},
        "semantic": {name: float(block["weights"]["semantic"][name]) for name in SEMANTIC_SIGNALS},
    }
    _validate_weight_group(weights["surface"], "surface")
    _validate_weight_group(weights["semantic"], "semantic")
    thresholds = {
        "low": float(block["thresholds"]["low"]),
        "high": float(block["thresholds"]["high"]),
    }
    if "penalties" in block:
        penalties = {
            "negation": float(block["penalties"]["negation"]),
            "antonym": float(block["penalties"]["antonym"]),
            "order": float(block["penalties"]["order"]),
        }
    else:
        penalties = dict(DEFAULT_PENALTIES)
    return weights, thresholds, penalties


def load_backend_thresholds(
    name: str, path: str | Path, default: tuple[float, float],
) -> tuple[float, float]:
    """Read the (low, high) thresholds for a backend from config.json or fall back to default."""
    block = load_backend_block(name, path)
    if block is None or "thresholds" not in block:
        return default
    t = block["thresholds"]
    return float(t["low"]), float(t["high"])
