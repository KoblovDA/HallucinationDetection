"""Run LLM-based Type 1 injection on triples that the rule-based injector skipped.

Reads `data/triples.jsonl` (full) and `data/type1.jsonl` (rule-based), finds uncovered ones,
calls OpenRouter for each, validates the swap, and writes the union to `data/type1_full.jsonl`.

Saves progress incrementally to `data/type1_llm_progress.jsonl` so a crash mid-way is recoverable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import read_jsonl, write_jsonl
from src.llm_inject import OpenRouterError, propose_swap


def _validate(answer: str, orig: str, new: str) -> tuple[bool, str]:
    if not orig:
        return False, "empty_original"
    if orig not in answer:
        return False, "original_not_in_answer"
    if not new:
        return False, "empty_new"
    if new == orig:
        return False, "identity"
    idx = answer.find(orig)
    left = answer[idx - 1] if idx > 0 else ""
    right = answer[idx + len(orig)] if idx + len(orig) < len(answer) else ""
    if left.isalpha() and orig[0].isalpha():
        return False, "mid_word_left"
    if right.isalpha() and orig[-1].isalpha():
        return False, "mid_word_right"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/triples.jsonl")
    parser.add_argument("--rule-based", default="data/type1.jsonl")
    parser.add_argument("--out", default="data/type1_full.jsonl")
    parser.add_argument("--progress", default="data/type1_llm_progress.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="optional cap on number of LLM calls")
    parser.add_argument("--workers", type=int, default=10, help="number of parallel API calls")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: set OPENROUTER_API_KEY env var", file=sys.stderr)
        return 1

    triples = read_jsonl(args.triples)
    rule_based = read_jsonl(args.rule_based)
    covered_ids = {row["id"] for row in rule_based}
    uncovered = [t for t in triples if t["id"] not in covered_ids]
    print(f"Triples: {len(triples)}, rule-based covered: {len(covered_ids)}, uncovered: {len(uncovered)}")

    # Resume from progress file if it exists
    progress_path = Path(args.progress)
    already_attempted: set[str] = set()
    resumed_records: list[dict] = []
    if progress_path.exists():
        for row in read_jsonl(progress_path):
            already_attempted.add(row["id"])
            resumed_records.append(row)
        print(f"Resuming: {len(already_attempted)} already attempted")

    todo = [t for t in uncovered if t["id"] not in already_attempted]
    if args.limit:
        todo = todo[: args.limit]
    print(f"To process: {len(todo)}\n")

    new_records: list[dict] = []
    n_valid = 0
    n_invalid = 0
    n_api_err = 0

    progress_f = open(progress_path, "a")
    progress_lock = threading.Lock()

    def _write_progress(record: dict) -> None:
        with progress_lock:
            progress_f.write(json.dumps(record) + "\n")
            progress_f.flush()

    def _process_one(t: dict) -> dict:
        try:
            swap = propose_swap(
                tool_output=t["tool_output_raw"],
                user_query=t["user"],
                answer=t["assistant"],
            )
        except OpenRouterError as e:
            return {"id": t["id"], "status": "api_error", "error": str(e)[:200]}
        except (ValueError, json.JSONDecodeError) as e:
            return {"id": t["id"], "status": "parse_error", "error": str(e)[:200]}

        orig = swap.original_substring
        new = swap.new_substring
        answer = t["assistant"]
        ok, reason = _validate(answer, orig, new)
        if not ok:
            return {"id": t["id"], "status": "invalid", "reason": reason,
                    "original": orig, "new": new}

        idx = answer.find(orig)
        new_answer = answer[:idx] + new + answer[idx + len(orig):]
        return {
            "id": t["id"],
            "status": "valid",
            "query": t["user"],
            "context": t["tool_output_raw"],
            "output": new_answer,
            "original_output": answer,
            "hallucination_labels": [
                {
                    "start": idx,
                    "end": idx + len(new),
                    "text": new,
                    "original_text": orig,
                    "field": "",
                    "type": "Type1_Contradiction",
                    "strategy": "llm",
                    "llm_reason": swap.reason,
                }
            ],
        }

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, t): t for t in todo}
            done = 0
            for fut in as_completed(futures):
                result = fut.result()
                done += 1
                status = result["status"]
                if status == "valid":
                    n_valid += 1
                    record = {k: v for k, v in result.items() if k != "status"}
                    new_records.append(record)
                elif status == "api_error":
                    n_api_err += 1
                else:
                    n_invalid += 1
                _write_progress(result)
                if done % 10 == 0:
                    print(f"  [{done}/{len(todo)}] valid={n_valid} invalid={n_invalid} api_err={n_api_err}", flush=True)
    finally:
        progress_f.close()

    print(f"\nDone: valid={n_valid}, invalid={n_invalid}, api_err={n_api_err}")

    # also recover valid records from previous resumed runs
    for row in resumed_records:
        if row.get("status") == "valid":
            rec = {k: v for k, v in row.items() if k not in ("status",)}
            new_records.append(rec)

    # union with rule-based
    print(f"Writing merged dataset to {args.out}")
    merged = rule_based + new_records
    write_jsonl(merged, args.out)
    print(f"  → total {len(merged)} samples ({len(rule_based)} rule-based + {len(new_records)} LLM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
