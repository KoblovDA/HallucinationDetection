"""Smoke-test LLM-based Type 1 injection on 10 samples that the rule-based injector skipped."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import read_jsonl
from src.llm_inject import OpenRouterError, propose_swap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/triples.jsonl")
    parser.add_argument("--type1", default="data/type1.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: set OPENROUTER_API_KEY in env first")
        return 1

    triples = read_jsonl(args.triples)
    type1_ids = {row["id"] for row in read_jsonl(args.type1)}
    uncovered = [t for t in triples if t["id"] not in type1_ids]
    print(f"Triples: {len(triples)}, type1 covered: {len(type1_ids)}, uncovered: {len(uncovered)}")

    rng = random.Random(args.seed)
    rng.shuffle(uncovered)
    sample = uncovered[: args.n]
    print(f"Testing LLM injection on {len(sample)} uncovered samples\n")

    results = []
    for i, t in enumerate(sample):
        print(f"--- Sample {i + 1}/{len(sample)}: {t['id']} ---")
        try:
            swap = propose_swap(
                tool_output=t["tool_output_raw"],
                user_query=t["user"],
                answer=t["assistant"],
            )
        except OpenRouterError as e:
            print(f"  API ERROR: {e}\n")
            results.append({"id": t["id"], "ok": False, "error": str(e)})
            continue
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  PARSE ERROR: {e}\n")
            results.append({"id": t["id"], "ok": False, "error": f"parse: {e}"})
            continue

        orig = swap.original_substring
        new = swap.new_substring
        answer = t["assistant"]

        ok = True
        reasons = []
        if not orig:
            ok = False
            reasons.append("empty original")
        elif orig not in answer:
            ok = False
            reasons.append(f"original not in answer: {orig!r}")
        if not new:
            ok = False
            reasons.append("empty new")
        if new == orig:
            ok = False
            reasons.append("new == original")
        # word-boundary check: ensure original_substring doesn't slice across a word
        if ok:
            idx = answer.find(orig)
            left_char = answer[idx - 1] if idx > 0 else ""
            right_char = answer[idx + len(orig)] if idx + len(orig) < len(answer) else ""
            if left_char.isalpha() and orig[0].isalpha():
                ok = False
                reasons.append(f"original starts mid-word (left='{left_char}')")
            if right_char.isalpha() and orig[-1].isalpha():
                ok = False
                reasons.append(f"original ends mid-word (right='{right_char}')")

        print(f"  proposed: {orig!r} -> {new!r}")
        print(f"  reason: {swap.reason}")
        if ok:
            idx = answer.find(orig)
            new_answer = answer[:idx] + new + answer[idx + len(orig):]
            pre = new_answer[max(0, idx - 60): idx]
            post = new_answer[idx + len(new): idx + len(new) + 60]
            print(f"  context: ...{pre}[[{new}]]{post}...")
            print(f"  VALID")
        else:
            print(f"  INVALID: {', '.join(reasons)}")
        print()

        results.append({
            "id": t["id"],
            "ok": ok,
            "original_substring": orig,
            "new_substring": new,
            "reason": swap.reason,
            "validation_errors": reasons,
        })
        time.sleep(0.5)

    n_valid = sum(1 for r in results if r.get("ok"))
    print(f"\n=== Summary: {n_valid}/{len(results)} valid LLM proposals ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
