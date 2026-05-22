"""Build train/val/test augmented Type 1 dataset.

Splits source triples by id (50 val / 150 test / rest train) and produces:
  - data/train.jsonl: up to N variants per train source (rule-based + LLM-fill)
  - data/val.jsonl:   1 variant per val source
  - data/test.jsonl:  1 variant per test source

Saves progress incrementally so a crash mid-way is recoverable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import (
    Variant,
    llm_propose_excluding,
    select_diverse,
    variant_to_record,
)
from src.data import Triple, read_jsonl, write_jsonl
from src.injection import collect_all_swaps
from src.pools import build_cross_sample_pool


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


def split_triples(triples: list[Triple], n_val: int, n_test: int, seed: int
                  ) -> tuple[list[Triple], list[Triple], list[Triple]]:
    rng = random.Random(seed)
    ids = [t.id for t in triples]
    rng.shuffle(ids)
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train_ids = set(ids[n_test + n_val :])
    by_id = {t.id: t for t in triples}
    return [by_id[i] for i in train_ids], [by_id[i] for i in val_ids], [by_id[i] for i in test_ids]


def process_source(triple: Triple, target_n: int,
                   cross_pool: dict, api_key: str, seed: int) -> list[Variant]:
    rng = random.Random(seed)
    rule_candidates = collect_all_swaps(triple, rng, cross_pool)
    variants = select_diverse(rule_candidates, target_n, rng)

    seen_spans: set[tuple[int, int, str]] = {
        (v.span.start, v.span.end, v.span.text) for v in variants
    }
    consecutive_failures = 0
    while len(variants) < target_n and consecutive_failures < 2:
        excluded = [v.span.original_text for v in variants]
        new_v = llm_propose_excluding(triple, excluded, api_key=api_key, temperature=0.9)
        if new_v is None:
            consecutive_failures += 1
            continue
        key = (new_v.span.start, new_v.span.end, new_v.span.text)
        if key in seen_spans:
            consecutive_failures += 1
            continue
        seen_spans.add(key)
        variants.append(new_v)
        consecutive_failures = 0
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/triples.jsonl")
    parser.add_argument("--out-train", default="data/train.jsonl")
    parser.add_argument("--out-val", default="data/val.jsonl")
    parser.add_argument("--out-test", default="data/test.jsonl")
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--n-train-variants", type=int, default=5)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", default="data/augment_progress.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        return 1

    print(f"Loading triples from {args.triples}")
    triples = load_triples(args.triples)
    print(f"  → {len(triples)} triples")

    print("Building cross-sample pool")
    cross_pool = build_cross_sample_pool(triples)

    print(f"Splitting (seed={args.seed}): val={args.n_val}, test={args.n_test}, rest=train")
    train, val, test = split_triples(triples, args.n_val, args.n_test, args.seed)
    print(f"  → train={len(train)}, val={len(val)}, test={len(test)}")

    # All sources to process: each labeled with split + how many variants needed
    jobs: list[tuple[Triple, str, int]] = (
        [(t, "train", args.n_train_variants) for t in train]
        + [(t, "val", 1) for t in val]
        + [(t, "test", 1) for t in test]
    )

    # Resume: skip jobs already in progress file
    progress_path = Path(args.progress)
    already_done: set[str] = set()
    resumed: dict[str, list[dict]] = {}
    if progress_path.exists():
        for line in open(progress_path):
            if not line.strip(): continue
            rec = json.loads(line)
            sid = rec["source_id"]
            already_done.add(sid)
            resumed.setdefault(sid, []).append(rec)
        print(f"Resuming: {len(already_done)} sources already processed")

    todo = [(t, split, n) for (t, split, n) in jobs if t.id not in already_done]
    print(f"To process: {len(todo)}")

    progress_f = open(progress_path, "a")
    progress_lock = threading.Lock()

    def _write_records(records: list[dict]) -> None:
        with progress_lock:
            for r in records:
                progress_f.write(json.dumps(r) + "\n")
            progress_f.flush()

    def _worker(triple: Triple, split: str, target_n: int) -> list[dict]:
        seed = hash((triple.id, args.seed)) & 0xFFFFFFFF
        variants = process_source(triple, target_n, cross_pool, api_key, seed)
        records = []
        for i, v in enumerate(variants):
            rec = variant_to_record(triple, v, i)
            rec["split"] = split
            records.append(rec)
        _write_records(records)
        return records

    new_records: list[dict] = []
    n_processed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t, sp, n): (t, sp) for (t, sp, n) in todo}
            for fut in as_completed(futures):
                t, sp = futures[fut]
                try:
                    recs = fut.result()
                    new_records.extend(recs)
                    n_processed += 1
                    if n_processed % 25 == 0:
                        print(f"  [{n_processed}/{len(todo)}] sources processed, {len(new_records)} variants so far", flush=True)
                except Exception as e:
                    print(f"  ! source {t.id} failed: {e}", file=sys.stderr)
    finally:
        progress_f.close()

    # combine new + resumed
    all_records = new_records.copy()
    for sid, recs in resumed.items():
        all_records.extend(recs)

    # bucket by split
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in all_records:
        sp = r.get("split", "train")
        by_split.setdefault(sp, []).append(r)

    for sp, path in [("train", args.out_train), ("val", args.out_val), ("test", args.out_test)]:
        # strip internal split field before writing
        cleaned = [{k: v for k, v in r.items() if k != "split"} for r in by_split.get(sp, [])]
        write_jsonl(cleaned, path)
        print(f"  {sp}: {len(cleaned)} samples → {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
