# Fine-tune ModernBERT — Hallucination Detection in Tool Calling

Token classifier on top of either:
- `KRLabsOrg/lettucedect-large-modernbert-en-v1` — already fine-tuned on RAGTruth (recommended; transfer learning).
- `answerdotai/ModernBERT-large` — base model from scratch.

Input: `[CLS] context [SEP] question [SEP] answer [SEP]`. Per-token binary labels (1 = hallucinated). Context/question tokens are masked out of the loss (label=-100).

Upload `combined_train.jsonl`, `combined_val.jsonl`, `combined_test.jsonl` (drop in `/content/` or run upload cell). **Requires GPU**.