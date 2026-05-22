"""Local eval of LettuceDetect on all three test sets.

Usage:
  python scripts/eval_lettucedetect.py \
    --type1 data/test.jsonl --type2 data/type2_test.jsonl --type3 data/type3_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.lettucedetect_runner import DEFAULT_MODEL, LettuceDetectRunner
from src.data import read_jsonl
from src.evaluation import example_metrics, span_metrics


def evaluate(name: str, samples: list[dict], runner: LettuceDetectRunner) -> dict:
    print(f"\n=== {name} ({len(samples)} samples) ===", flush=True)
    preds = runner.predict_many(samples)
    micro, macro = span_metrics(samples, preds)
    ex = example_metrics(samples, preds)
    print(f"  span micro P/R/F1: {micro.precision:.3f} / {micro.recall:.3f} / {micro.f1:.3f}")
    print(f"  span macro F1:     {macro.f1:.3f}")
    print(f"  example P/R/F1:    {ex.precision:.3f} / {ex.recall:.3f} / {ex.f1:.3f}")
    return {
        "name": name,
        "n": len(samples),
        "span_micro": micro.as_dict(),
        "span_macro_f1": macro.f1,
        "example": ex.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type1", default="data/test.jsonl")
    parser.add_argument("--type2", default="data/type2_test.jsonl")
    parser.add_argument("--type3", default="data/type3_test.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="data/results_lettucedetect.json")
    args = parser.parse_args()

    print(f"Loading LettuceDetect: {args.model}")
    runner = LettuceDetectRunner(model_path=args.model)

    results = {}
    for label, path in [("Type 1 (Hallucination)", args.type1),
                        ("Type 2 (Overgeneration)", args.type2),
                        ("Type 3 (Missing Tool)", args.type3)]:
        if not Path(path).exists():
            print(f"  ! missing {path}, skipping", file=sys.stderr)
            continue
        samples = read_jsonl(path)
        results[label] = evaluate(label, samples, runner)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
