"""Smoke-test LLM-based Type 3 injection on N samples."""
from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import Triple, read_jsonl
from src.llm_type3 import make_record, propose_type3


def load_triples(path: str) -> list[Triple]:
    rows = read_jsonl(path)
    return [Triple(id=r["id"], user=r["user"], tool_call=r.get("tool_call", ""),
                   tool_output_raw=r["tool_output_raw"], tool_output=r["tool_output"],
                   assistant=r["assistant"], tools_available=r.get("tools_available", []))
            for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/triples.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); return 1

    triples = load_triples(args.triples)
    rng = random.Random(args.seed)
    rng.shuffle(triples)
    sample = triples[: args.n]

    print(f"Testing LLM Type 3 on {len(sample)} samples\n")

    results: list[tuple[Triple, dict]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(propose_type3, t, api_key): t for t in sample}
        for fut in as_completed(futures):
            t = futures[fut]
            prop = fut.result()
            if prop is None:
                results.append((t, None))
                continue
            rec = make_record(t, prop)
            results.append((t, rec))

    valid = [r for r in results if r[1] is not None]
    print(f"Generated: {len(valid)}/{len(sample)}\n")
    for i, (t, rec) in enumerate(results):
        if rec is None:
            print(f"--- {i+1}. {t.id} — FAILED")
            continue
        l = rec["hallucination_labels"][0]
        out = rec["output"]
        pre = out[max(0, l["start"]-150):l["start"]]
        print(f"--- {i+1}. {t.id}  action={l['field']}")
        tool_names = [tt.get("name", "?") for tt in t.tools_available]
        print(f"  tools: {tool_names}")
        print(f"  reason: {l.get('llm_reason')}")
        print(f"  ...{pre}")
        print(f"  >>> {l['text']} <<<")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
