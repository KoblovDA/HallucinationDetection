"""Add clean (non-hallucinated) samples to each split, and produce both per-type
balanced files and a single combined train/val/test.

Output files:
  data/type1_train_balanced.jsonl, _val_, _test_  (Type 1 + clean per source)
  data/type2_train_balanced.jsonl, _val_, _test_
  data/type3_train_balanced.jsonl, _val_, _test_
  data/combined_train.jsonl, combined_val.jsonl, combined_test.jsonl
    (all three hallucination types + clean, one clean per source)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import read_jsonl, write_jsonl


def build_clean(triple: dict, split: str) -> dict:
    return {
        "id": f"{triple['id']}_clean",
        "source_id": triple["id"],
        "query": triple["user"],
        "context": triple["tool_output_raw"],
        "output": triple["assistant"],
        "original_output": triple["assistant"],
        "tools_available": triple.get("tools_available", []),
        "hallucination_labels": [],
        "split": split,
    }


def ensure_tools_available(record: dict, triples_by_id: dict[str, dict]) -> dict:
    """Augment a hallucinated record with tools_available looked up from the source triple."""
    if "tools_available" in record:
        return record
    sid = record.get("source_id")
    if sid and sid in triples_by_id:
        record["tools_available"] = triples_by_id[sid].get("tools_available", [])
    else:
        record["tools_available"] = []
    return record


def split_source_ids(triples: list[dict], n_val: int, n_test: int, seed: int
                     ) -> tuple[set[str], set[str], set[str]]:
    """Reproduces the split used by build_augmented.py / build_type2.py / build_type3_llm.py."""
    rng = random.Random(seed)
    ids = [t["id"] for t in triples]
    rng.shuffle(ids)
    test = set(ids[:n_test])
    val = set(ids[n_test : n_test + n_val])
    train = set(ids[n_test + n_val :])
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/triples.jsonl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    triples = read_jsonl(args.triples)
    triples_by_id = {t["id"]: t for t in triples}
    print(f"Triples: {len(triples)}")

    train_ids, val_ids, test_ids = split_source_ids(triples, args.n_val, args.n_test, args.seed)
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    type_files = {
        1: {"train": "train.jsonl",      "val": "val.jsonl",      "test": "test.jsonl"},
        2: {"train": "type2_train.jsonl","val": "type2_val.jsonl","test": "type2_test.jsonl"},
        3: {"train": "type3_train.jsonl","val": "type3_val.jsonl","test": "type3_test.jsonl"},
    }

    # collect per-(type, split) records (just hallucinated). Prefix IDs with type to avoid
    # collisions when merging (different per-type files can use the same `{source}_v0` IDs).
    per_type: dict[tuple[int, str], list[dict]] = {}
    for t in (1, 2, 3):
        for split in ("train", "val", "test"):
            path = data_dir / type_files[t][split]
            if not path.exists():
                print(f"  ! missing {path}")
                continue
            recs = read_jsonl(path)
            for r in recs:
                r["id"] = f"t{t}_{r['id']}"
                r["split"] = split
                ensure_tools_available(r, triples_by_id)
            per_type[(t, split)] = recs

    # build per-type balanced files (hallucinated + clean per source in this split)
    def clean_for(split: str) -> list[dict]:
        sids = {"train": train_ids, "val": val_ids, "test": test_ids}[split]
        return [build_clean(triples_by_id[sid], split) for sid in sids]

    for t in (1, 2, 3):
        for split in ("train", "val", "test"):
            hallucinated = per_type.get((t, split), [])
            clean = clean_for(split)
            merged = hallucinated + clean
            out = data_dir / f"type{t}_{split}_balanced.jsonl"
            # strip the temp "split" key
            cleaned = [{k: v for k, v in r.items() if k != "split"} for r in merged]
            write_jsonl(cleaned, out)
            print(f"  type{t}_{split}_balanced: {len(hallucinated)} hallu + {len(clean)} clean = {len(merged)} → {out}")

    # combined splits (one clean per source, plus all hallucinated types)
    for split in ("train", "val", "test"):
        all_hallu = (per_type.get((1, split), [])
                     + per_type.get((2, split), [])
                     + per_type.get((3, split), []))
        clean = clean_for(split)
        merged = all_hallu + clean
        out = data_dir / f"combined_{split}.jsonl"
        cleaned = [{k: v for k, v in r.items() if k != "split"} for r in merged]
        write_jsonl(cleaned, out)
        h, c = len(all_hallu), len(clean)
        pct = 100 * c / (h + c) if (h + c) else 0.0
        print(f"  combined_{split}: {h} hallu + {c} clean = {h + c} ({pct:.1f}% clean) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
