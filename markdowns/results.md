# Results

All numbers are on **`data/combined_test.jsonl`** — 599 samples: 150 Type 1 + 150 Type 2 + 149 Type 3 + 150 clean (25% clean). Per-type rows are filtered subsets: `{Type N hallucinated} ∪ {all clean}`, so each per-type row uses 299–300 samples.

Metrics:
- **Span micro P/R/F1** — character-overlap RAGTruth-style: P = overlap_chars / pred_chars, R = overlap_chars / gold_chars, F1 = harmonic mean.
- **Span macro F1** — per-sample F1 averaged across the split (clean correctly predicted as empty counts as 1.0).
- **Example P/R/F1** — binary "has any hallucination span" at the sample level.

## Main table (span micro F1, the primary metric)

| Split | N | LettuceDetect baseline | LLM-as-judge (Qwen3-235B) | Δ |
|---|---|---|---|---|
| **Combined** | **599** | **0.660** | **0.859** | **+0.199** |
| Type 1 + clean | 300 | 0.137 | 0.315 | +0.178 |
| Type 2 + clean | 300 | 0.726 | 0.844 | +0.118 |
| Type 3 + clean | 299 | 0.597 | 0.792 | +0.195 |

## Full breakdown

### LettuceDetect baseline

Off-the-shelf `KRLabsOrg/lettucedect-large-modernbert-en-v1` (ModernBERT-large token classifier trained on RAGTruth). Inference in Colab on T4 GPU.

| Split | N | Span P | Span R | Span F1 | Span macro F1 | Ex P | Ex R | Ex F1 |
|---|---|---|---|---|---|---|---|---|
| Combined | 599 | 0.515 | 0.918 | 0.660 | 0.629 | 0.871 | 0.902 | 0.886 |
| Type 1 + clean | 300 | 0.077 | 0.669 | 0.137 | 0.440 | 0.661 | 0.780 | 0.716 |
| Type 2 + clean | 300 | 0.570 | 0.998 | 0.726 | 0.746 | 0.714 | 1.000 | 0.833 |
| Type 3 + clean | 299 | 0.461 | 0.847 | 0.597 | 0.671 | 0.697 | 0.926 | 0.795 |

Reproducibility: [`notebooks/lettucedetect_baseline.ipynb`](notebooks/lettucedetect_baseline.ipynb) (uses `KRLabsOrg/lettucedect-large-modernbert-en-v1`; needs combined_test.jsonl in `/content/`).

### LLM-as-judge detector (Qwen3-235B via OpenRouter)

`qwen/qwen3-235b-a22b-2507` with custom prompt (3-type taxonomy + TIGHT SPANS rule for value-level contradictions) + 4 few-shot examples from train (1 per type + 1 clean). Inference: parallel via OpenRouter API.

| Split | N | Span P | Span R | Span F1 | Span macro F1 | Ex P | Ex R | Ex F1 |
|---|---|---|---|---|---|---|---|---|
| Combined | 599 | 0.758 | 0.992 | 0.859 | 0.827 | 0.901 | 0.991 | 0.944 |
| Type 1 + clean | 300 | 0.191 | 0.900 | 0.315 | 0.688 | 0.749 | 0.973 | 0.846 |
| Type 2 + clean | 300 | 0.732 | 0.996 | 0.844 | 0.824 | 0.754 | 1.000 | 0.860 |
| Type 3 + clean | 299 | 0.655 | 1.000 | 0.792 | 0.815 | 0.753 | 1.000 | 0.859 |

Cost: ~$0.90 per full pass. Wall-clock: ~3 minutes with 15 parallel workers. Reproducibility: [`scripts/eval_llm_detector.py`](scripts/eval_llm_detector.py).

## Notes

- **Example precision = 1.0 in earlier reports was misleading.** Until we added clean samples to test, every test sample had a hallucination → P trivially = 1.0. With 25% clean in combined_test, P/R/F1 became informative. Recomputed LettuceDetect numbers above are with clean.
- **Type 1 micro F1 is suppressed by labeling incompleteness, not detector failure.** Type 1 hallucinated samples have short (median 9-char) gold spans. The same answers sometimes contain pre-existing Type-3-style phrases ("Would you like me to …") from the original ToolACE generation — these are real Type-3 hallucinations we never labeled. The LLM detector flags them, dragging char-level precision down. Macro F1 = 0.688 is the more representative number for Type 1 detector quality.
- **LLM detector got access to `tools_available` in its prompt; LettuceDetect did not.** Adding the available-tool list to LettuceDetect's input might lift its Type 3 numbers; we did not run that ablation.
- **ID collisions bug**: before fix, per-type variants used non-unique IDs like `{source}_v0` that collided across types. The LettuceDetect numbers are not affected (its eval did not look up by ID), but the first LLM detector pass was — fixed by prefixing IDs with `t1_` / `t2_` / `t3_` in `combine_splits.py`.

## TODO

- LookBackLens baseline (second baseline required by the assignment).
- Fine-tune ModernBERT on `combined_train` and compare against the LLM detector.
- Possible ensemble: LLM detector + a structural JSON-value matcher tightening Type 1 precision.
