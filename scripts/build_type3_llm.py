"""Build augmented Type 3 dataset entirely via LLM.

Splits source triples 50 val / 150 test / rest train (same seed/split as Type 1).
For train: N LLM calls per source with action-diversity (excluding previously used categories).
For val/test: 1 LLM call per source.

Saves progress incrementally to `data/type3_llm_progress.jsonl`.
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

from src.data import Triple, read_jsonl, write_jsonl
from src.llm_type3 import make_record, propose_type3


def load_triples(path: str) -> list[Triple]:
    rows = read_jsonl(path)
    return [Triple(id=r["id"], user=r["user"], tool_call=r.get("tool_call", ""),
                   tool_output_raw=r["tool_output_raw"], tool_output=r["tool_output"],
                   assistant=r["assistant"], tools_available=r.get("tools_available", []))
            for r in rows]


def split_triples(triples: list[Triple], n_val: int, n_test: int, seed: int):
    import random
    rng = random.Random(seed)
    ids = [t.id for t in triples]
    rng.shuffle(ids)
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    by_id = {t.id: t for t in triples}
    return (
        [by_id[i] for i in ids[n_test + n_val :]],
        [by_id[i] for i in val_ids],
        [by_id[i] for i in test_ids],
    )


def process_source(triple: Triple, target_n: int, api_key: str) -> list[dict]:
    """Sequentially call LLM target_n times, excluding previously chosen actions."""
    records: list[dict] = []
    used_actions: list[str] = []
    consecutive_failures = 0
    for variant_idx in range(target_n):
        prop = propose_type3(
            triple, api_key=api_key,
            exclude_actions=used_actions,
            temperature=0.9 if variant_idx > 0 else 0.7,
        )
        if prop is None:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                break
            continue
        consecutive_failures = 0
        rec = make_record(triple, prop)
        rec["id"] = f"{triple.id}_v{variant_idx}"
        rec["source_id"] = triple.id
        records.append(rec)
        if prop.action:
            used_actions.append(prop.action)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/triples.jsonl")
    parser.add_argument("--out-train", default="data/type3_train.jsonl")
    parser.add_argument("--out-val", default="data/type3_val.jsonl")
    parser.add_argument("--out-test", default="data/type3_test.jsonl")
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--n-train-variants", type=int, default=3)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", default="data/type3_llm_progress.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    triples = load_triples(args.triples)
    print(f"Triples: {len(triples)}")

    train, val, test = split_triples(triples, args.n_val, args.n_test, args.seed)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    jobs: list[tuple[Triple, str, int]] = (
        [(t, "train", args.n_train_variants) for t in train]
        + [(t, "val", 1) for t in val]
        + [(t, "test", 1) for t in test]
    )

    # resume
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

    todo = [(t, sp, n) for (t, sp, n) in jobs if t.id not in already_done]
    print(f"To process: {len(todo)}")

    progress_f = open(progress_path, "a")
    progress_lock = threading.Lock()

    def _write(recs: list[dict]) -> None:
        with progress_lock:
            for r in recs:
                progress_f.write(json.dumps(r) + "\n")
            progress_f.flush()

    def _worker(triple: Triple, split: str, target_n: int) -> list[dict]:
        recs = process_source(triple, target_n, api_key)
        for r in recs:
            r["split"] = split
        _write(recs)
        return recs

    new_records: list[dict] = []
    n_processed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t, sp, n): (t, sp) for (t, sp, n) in todo}
            for fut in as_completed(futures):
                t, sp = futures[fut]
                try:
                    new_records.extend(fut.result())
                    n_processed += 1
                    if n_processed % 25 == 0:
                        print(f"  [{n_processed}/{len(todo)}] {len(new_records)} variants so far", flush=True)
                except Exception as e:
                    print(f"  ! source {t.id} failed: {e}", file=sys.stderr)
    finally:
        progress_f.close()

    # merge with resumed
    all_records = new_records.copy()
    for sid, recs in resumed.items():
        all_records.extend(recs)

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in all_records:
        by_split.setdefault(r.get("split", "train"), []).append(r)

    for sp, path in [("train", args.out_train), ("val", args.out_val), ("test", args.out_test)]:
        cleaned = [{k: v for k, v in r.items() if k != "split"} for r in by_split.get(sp, [])]
        write_jsonl(cleaned, path)
        print(f"  {sp}: {len(cleaned)} samples → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
