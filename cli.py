"""Command-line entry point: query a phrase against a candidate list."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from backends import BackendMatchResult, available, get_backend


def load_candidates(path: str | Path) -> list[str]:
    """Read candidates from a file. JSON list if .json, else one phrase per line."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError(f"{path}: JSON candidates must be a list of strings")
        return data
    return [line.strip() for line in text.splitlines() if line.strip()]


def match(
    query: str,
    candidates: Sequence[str],
    backend_name: str = "classical",
    k: int = 20,
) -> list[BackendMatchResult]:
    """Score every candidate via the selected backend, return top-k by score desc."""
    backend = get_backend(backend_name, list(candidates))
    scored: list[BackendMatchResult] = []
    for cand in candidates:
        score, signals = backend.score_with_explain(query, cand)
        scored.append(BackendMatchResult(
            candidate=cand,
            score=score,
            label=backend.label(score),
            signals=signals,
        ))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:k]


def format_results(results: Sequence[BackendMatchResult]) -> str:
    """Render results as a fixed-width table; include the signal breakdown when provided."""
    if not results:
        return "(no candidates)"
    show_breakdown = results[0].signals is not None
    lines: list[str] = []
    if show_breakdown:
        header = (
            f"{'rank':>4}  {'score':>6}  {'label':<8}  "
            f"{'tfidf':>6} {'jacc':>6} {'wnet':>6} {'ngrm':>6} {'ordr':>6} {'soft':>6}  "
            f"candidate"
        )
    else:
        header = f"{'rank':>4}  {'score':>6}  {'label':<8}  candidate"
    lines.append(header)
    lines.append("-" * len(header))
    for rank, r in enumerate(results, start=1):
        if show_breakdown:
            s = r.signals or {}
            lines.append(
                f"{rank:>4}  {r.score:>6.3f}  {r.label:<8}  "
                f"{s['tfidf']:>6.3f} {s['jaccard']:>6.3f} {s['wordnet']:>6.3f} "
                f"{s['ngram']:>6.3f} {s['order']:>6.3f} {s['soft_overlap']:>6.3f}  "
                f"{r.candidate}"
            )
        else:
            lines.append(f"{rank:>4}  {r.score:>6.3f}  {r.label:<8}  {r.candidate}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidates against a query phrase.",
    )
    parser.add_argument("query", help="query phrase")
    parser.add_argument(
        "candidates",
        help="path to candidates file (one phrase per line, or a .json list of strings)",
    )
    parser.add_argument("--k", type=int, default=20, help="number of top candidates to score")
    parser.add_argument(
        "--backend",
        default="classical",
        choices=available(),
        help="similarity backend to use",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    candidates = load_candidates(args.candidates)
    if not candidates:
        print(f"error: no candidates loaded from {args.candidates}", file=sys.stderr)
        return 1
    results = match(args.query, candidates, backend_name=args.backend, k=args.k)
    print(format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
