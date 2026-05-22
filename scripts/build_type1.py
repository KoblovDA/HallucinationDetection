"""End-to-end pipeline: download ToolACE → extract triples → inject Type 1 → write RAGTruth JSONL."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import download_toolace, extract_triples, write_jsonl
from src.injection import inject
from src.pools import build_cross_sample_pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/type1.jsonl")
    parser.add_argument("--triples-out", default="data/triples.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Downloading ToolACE...")
    toolace_path = download_toolace()
    print(f"  → {toolace_path}")

    print("Extracting triples...")
    triples = extract_triples(toolace_path)
    print(f"  → {len(triples)} triples")

    print("Writing triples to disk...")
    write_jsonl((t.to_json() for t in triples), args.triples_out)
    print(f"  → {args.triples_out}")

    print("Building cross-sample pool...")
    cross_pool = build_cross_sample_pool(triples)
    print(f"  → {len(cross_pool)} (tool, field) keys")

    print("Injecting Type 1 hallucinations...")
    injected = []
    n_skipped = 0
    strategy_counts: dict[str, int] = {}
    for t in triples:
        result = inject(t, rng, cross_pool=cross_pool)
        if result is None:
            n_skipped += 1
            continue
        corrupted, spans = result
        for s in spans:
            strategy_counts[s.strategy] = strategy_counts.get(s.strategy, 0) + 1
        injected.append({
            "id": t.id,
            "query": t.user,
            "context": t.tool_output_raw,
            "output": corrupted,
            "original_output": t.assistant,
            "hallucination_labels": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "original_text": s.original_text,
                    "field": s.field,
                    "type": "Type1_Contradiction",
                    "strategy": s.strategy,
                }
                for s in spans
            ],
        })

    print(f"  → {len(injected)} injected ({n_skipped} skipped, {100 * len(injected) / len(triples):.1f}% coverage)")
    print("\nStrategy breakdown:")
    for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:5d}  {s}")

    write_jsonl(injected, args.out)
    print(f"\nDataset written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
