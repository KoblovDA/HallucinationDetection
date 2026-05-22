"""Full eval of LLM detector on combined_test with few-shot from combined_train.

Outputs:
  data/results_llm_detector.json  — per-split metrics + raw predictions
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

from src.data import read_jsonl
from src.evaluation import example_metrics, span_metrics
from src.llm_detector import detect_one, pick_few_shot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  default="data/combined_test.jsonl")
    parser.add_argument("--train", default="data/combined_train.jsonl")
    parser.add_argument("--out",   default="data/results_llm_detector.json")
    parser.add_argument("--progress", default="data/llm_detector_progress.jsonl")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); return 1

    samples = read_jsonl(args.test)
    train = read_jsonl(args.train)
    few_shot = pick_few_shot(train, seed=args.seed)
    print(f"Few-shot ({len(few_shot)}):")
    for ex in few_shot:
        kind = ex["hallucination_labels"][0].get("type", "?") if ex["hallucination_labels"] else "clean"
        print(f"  - {ex['id']}  [{kind}]")
    print(f"\nEvaluating {len(samples)} samples with {args.workers} workers…\n")

    # Resume
    progress_path = Path(args.progress)
    done: dict[str, list[dict]] = {}
    if progress_path.exists():
        for line in open(progress_path):
            rec = json.loads(line)
            done[rec["id"]] = rec["pred_spans"]
        print(f"Resumed: {len(done)} samples already predicted")

    todo = [s for s in samples if s["id"] not in done]
    progress_f = open(progress_path, "a")
    lock = threading.Lock()

    def _write(sid: str, spans: list[dict]) -> None:
        with lock:
            progress_f.write(json.dumps({"id": sid, "pred_spans": spans}) + "\n")
            progress_f.flush()

    def _worker(s: dict) -> tuple[str, list[dict]]:
        located, _ = detect_one(s, api_key=api_key, few_shot=few_shot)
        spans = [{"start": p.start, "end": p.end, "text": p.text} for p in located]
        _write(s["id"], spans)
        return s["id"], spans

    n_processed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, s): s for s in todo}
            for fut in as_completed(futures):
                try:
                    sid, spans = fut.result()
                    done[sid] = spans
                    n_processed += 1
                    if n_processed % 25 == 0:
                        print(f"  [{n_processed}/{len(todo)}]", flush=True)
                except Exception as e:
                    s = futures[fut]
                    print(f"  ! {s['id']} failed: {str(e)[:200]}")
                    done[s["id"]] = []
    finally:
        progress_f.close()

    # Align preds order to samples order
    pred_spans = [done.get(s["id"], []) for s in samples]

    # Metrics per subset
    def filter_subset(label: str):
        sel_s, sel_p = [], []
        for s, p in zip(samples, pred_spans):
            if not s["hallucination_labels"]:
                sel_s.append(s); sel_p.append(p); continue
            if s["hallucination_labels"][0].get("type") == label:
                sel_s.append(s); sel_p.append(p)
        return sel_s, sel_p

    subsets = {
        "Combined":          (samples, pred_spans),
        "Type 1 + clean":    filter_subset("Type1_Contradiction"),
        "Type 2 + clean":    filter_subset("Type2_Overgeneration"),
        "Type 3 + clean":    filter_subset("Type3_MissingTool"),
    }

    out = {}
    print("\n=== Results ===")
    for name, (s, p) in subsets.items():
        micro, macro = span_metrics(s, p)
        ex = example_metrics(s, p)
        out[name] = {
            "n": len(s),
            "span_micro": micro.as_dict(),
            "span_macro_f1": macro.f1,
            "example": ex.as_dict(),
        }
        print(f"  {name:25s} (N={len(s):3d}): "
              f"span P/R/F1 = {micro.precision:.3f}/{micro.recall:.3f}/{micro.f1:.3f} | "
              f"macro F1 = {macro.f1:.3f} | "
              f"ex P/R/F1 = {ex.precision:.3f}/{ex.recall:.3f}/{ex.f1:.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
