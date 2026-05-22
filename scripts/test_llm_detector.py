"""Smoke-test LLM detector on N random samples from combined_test, optionally with few-shot."""
from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import read_jsonl
from src.llm_detector import detect_one, pick_few_shot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  default="data/combined_test.jsonl")
    parser.add_argument("--train", default="data/combined_train.jsonl")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--few-shot", action="store_true", default=True,
                        help="include few-shot examples (default: on)")
    parser.add_argument("--no-few-shot", dest="few_shot", action="store_false")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); return 1

    test_samples = read_jsonl(args.test)
    rng = random.Random(args.seed)
    rng.shuffle(test_samples)
    sample = test_samples[: args.n]
    print(f"Testing LLM detector on {len(sample)} samples (from combined_test).")

    few_shot_examples: list[dict] = []
    if args.few_shot:
        train = read_jsonl(args.train)
        few_shot_examples = pick_few_shot(train, seed=args.seed)
        print(f"Using few-shot ({len(few_shot_examples)} examples):")
        for ex in few_shot_examples:
            kind = ex["hallucination_labels"][0].get("type", "?") if ex["hallucination_labels"] else "clean"
            print(f"  - {ex['id']}  [{kind}]  ctx_len={len(ex['context'])}, ans_len={len(ex['output'])}")
    else:
        print("Few-shot DISABLED.")
    print()

    results: list[tuple[dict, list, list, str | None]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(detect_one, s, api_key, few_shot=few_shot_examples or None): s
                   for s in sample}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                located, not_found = fut.result()
                results.append((s, located, not_found, None))
            except Exception as e:
                results.append((s, [], [], str(e)[:200]))

    n_ok = sum(1 for r in results if r[3] is None)
    n_not_found = sum(len(r[2]) for r in results)
    print(f"OK: {n_ok}/{len(results)}, unlocatable substrings reported by LLM: {n_not_found}\n")

    # Per-type stats
    from collections import Counter
    by_kind: dict[str, list[tuple[dict, list, list]]] = {}
    for s, loc, nf, err in results:
        if err is not None: continue
        kind = s["hallucination_labels"][0].get("type", "?") if s["hallucination_labels"] else "clean"
        by_kind.setdefault(kind, []).append((s, loc, nf))

    # Sample-level qualitative
    for s, loc, nf, err in results:
        gold = s["hallucination_labels"]
        kind = "CLEAN" if not gold else gold[0].get("type", "?")
        print(f"--- {s['id']}  [{kind}]")
        if err:
            print(f"  ERROR: {err}"); print(); continue
        for g in gold:
            print(f"  GOLD: [{g['start']}..{g['end']}] {s['output'][g['start']:g['end']]!r}")
        if not loc:
            print("  PRED: (none)")
        for p in loc:
            print(f"  PRED: [{p.start}..{p.end}] {p.text!r}")
        if nf:
            print(f"  NOT FOUND: {nf}")
        print()

    # Quick numeric summary
    def stats(items):
        n = len(items)
        if not n: return None, None, None, None
        hallu = [(s,l) for s,l,_ in items if s["hallucination_labels"]]
        clean = [(s,l) for s,l,_ in items if not s["hallucination_labels"]]
        recall_hits = sum(1 for s,l in hallu if l)
        fp_clean = sum(1 for s,l in clean if l)
        return len(hallu), recall_hits, len(clean), fp_clean

    print("\n=== Summary by type ===")
    for kind in ["Type1_Contradiction", "Type2_Overgeneration", "Type3_MissingTool", "clean"]:
        items = by_kind.get(kind, [])
        h_n, h_hit, c_n, c_fp = stats(items)
        if h_n is None: continue
        print(f"  {kind:25s}  hallu detected: {h_hit}/{h_n}  | clean FPs: {c_fp}/{c_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
