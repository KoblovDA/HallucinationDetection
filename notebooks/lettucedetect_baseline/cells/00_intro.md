# LettuceDetect baseline — Hallucination Detection in Tool Calling

Evaluates the off-the-shelf `KRLabsOrg/lettucedect-large-modernbert-en-v1` checkpoint on three test splits we built from ToolACE, each corresponding to one hallucination type:

- **Type 1** — *Hallucination*: answer contradicts tool output.
- **Type 2** — *Overgeneration*: answer adds facts not in tool output.
- **Type 3** — *Missing tool*: answer proposes an action requiring an unavailable tool.

For each split we report span-level (micro + macro F1) and example-level (P/R/F1) metrics.

**Requires a GPU runtime (T4 / P100 / L4).** Inference on ~150 samples takes ~30 s on T4.

Upload the test files (`test.jsonl`, `type2_test.jsonl`, `type3_test.jsonl`) as a Kaggle dataset and point `DATA_DIR` to its mount path below.
