# LookBackLens baseline — Hallucination Detection in Tool Calling

Implements the LookBackLens method (Chuang et al., 2024) using `meta-llama/Llama-3.2-3B-Instruct` as the backbone (~3.2B params, fits on a Kaggle P100/T4 in fp16).

Pipeline:
1. For each train sample, do one forward pass with output_attentions=True. Extract per-token lookback ratios over the answer span.
2. For each hallucinated train sample, the gold span is the **positive** training example. Add `K=3` random non-overlapping windows of similar size as **negative** training examples. For clean train samples, only negatives.
3. Train a logistic regression on these `L*H`-dim feature vectors.
4. For test, do one forward pass per sample, slide a fixed-size window across the answer tokens, classify each window. Merge consecutive positive windows into char spans.
5. Evaluate against gold using the same span-overlap metrics we use for LettuceDetect / LLM detector.

Upload `combined_train.jsonl`, `combined_val.jsonl`, `combined_test.jsonl` to `/content/` (Colab) or as a Kaggle dataset under `/kaggle/input/halluc-toolace/`.

Requires GPU and a HF token with access to Llama-3.2 (add as Kaggle / Colab secret `HF_TOKEN`).