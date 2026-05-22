"""End-to-end pipeline for Type 3 (missing tool) hallucinations.

Loads triples, injects a rule-based 'missing tool' span into each, writes to data/type3.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import Triple, read_jsonl, write_jsonl
from src.inject_type3 import inject_type3, uncovered_actions


def load_triples(path: Path | str) -> list[Triple]:
    rows = read_jsonl(path)
    return [
        Triple(
            id=r["id"], user=r["user"], tool_call=r.get("tool_call", ""),
            tool_output_raw=r["tool_output_raw"], tool_output=r["tool_output"],
            assistant=r["assistant"], tools_available=r.get("tools_available", []),
        )
        for r in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/triples.jsonl")
    parser.add_argument("--out", default="data/type3.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    triples = load_triples(args.triples)
    print(f"Triples: {len(triples)}")

    rng = random.Random(args.seed)
    injected: list[dict] = []
    n_skipped = 0
    action_counts: dict[str, int] = {}
    n_uncovered_actions: dict[int, int] = {}

    for t in triples:
        n_uncov = len(uncovered_actions(t.tools_available))
        n_uncovered_actions[n_uncov] = n_uncovered_actions.get(n_uncov, 0) + 1
        result = inject_type3(t, rng)
        if result is None:
            n_skipped += 1
            continue
        new_answer, span = result
        action_counts[span["field"]] = action_counts.get(span["field"], 0) + 1
        injected.append({
            "id": t.id,
            "query": t.user,
            "context": t.tool_output_raw,
            "tools_available": t.tools_available,
            "output": new_answer,
            "original_output": t.assistant,
            "hallucination_labels": [span],
        })

    print(f"Injected: {len(injected)} ({100 * len(injected) / len(triples):.1f}%)")
    print(f"Skipped:  {n_skipped}")
    print()
    print("Distribution of action categories used:")
    for k, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {k}")
    print()
    print("Distribution of #uncovered-actions per source:")
    for k in sorted(n_uncovered_actions):
        print(f"  {k:2d} uncovered: {n_uncovered_actions[k]} sources")

    write_jsonl(injected, args.out)
    print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
