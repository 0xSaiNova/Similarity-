"""Tests for cli module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backends import BackendMatchResult
from cli import load_candidates, match


def test_load_candidates_reads_json_list(tmp_path: Path) -> None:
    path = tmp_path / "cands.json"
    path.write_text(json.dumps(["alpha", "beta", "gamma"]))
    assert load_candidates(path) == ["alpha", "beta", "gamma"]


def test_load_candidates_json_non_list_raises(tmp_path: Path) -> None:
    path = tmp_path / "cands.json"
    path.write_text(json.dumps({"phrase": "alpha"}))
    with pytest.raises(ValueError, match="must be a list of strings"):
        load_candidates(path)


def test_load_candidates_json_non_string_element_raises(tmp_path: Path) -> None:
    path = tmp_path / "cands.json"
    path.write_text(json.dumps(["alpha", 42, "gamma"]))
    with pytest.raises(ValueError, match="must be a list of strings"):
        load_candidates(path)


def test_load_candidates_reads_txt_one_per_line(tmp_path: Path) -> None:
    path = tmp_path / "cands.txt"
    path.write_text("the cat sat on the mat\na dog ran in the park\n")
    assert load_candidates(path) == [
        "the cat sat on the mat",
        "a dog ran in the park",
    ]


def test_load_candidates_txt_strips_blank_and_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "cands.txt"
    path.write_text("  alpha  \n\n\tbeta\n   \n")
    assert load_candidates(path) == ["alpha", "beta"]


def test_load_candidates_json_malformed_raises(tmp_path: Path) -> None:
    path = tmp_path / "cands.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_candidates(path)


def test_match_wrapper_returns_ranked_results() -> None:
    candidates = [
        "the cat sat on the mat",
        "a dog ran in the park",
        "birds fly south for winter",
        "the cat jumped over the fence",
    ]
    results = match("a cat on a mat", candidates, k=4)
    assert len(results) == 4
    assert all(isinstance(r, BackendMatchResult) for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].candidate == "the cat sat on the mat"
    assert results[0].label == "MATCH"


def test_match_wrapper_respects_k() -> None:
    candidates = ["alpha one", "beta two", "gamma three", "delta four"]
    results = match("alpha one", candidates, k=2)
    assert len(results) == 2
