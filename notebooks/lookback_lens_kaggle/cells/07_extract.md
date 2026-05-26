## Extract features

For TRAIN: one positive feature vector per hallucinated span + up to `N_NEG_PER_SAMPLE` random negative windows of similar size. For clean train samples, only negatives.

For VAL/TEST: only sliding-window features (used both for tuning threshold on val and for final predictions).

~2-3 hours on Kaggle P100 / T4 for train + val + test. Progress is saved to `/content/lookback_progress.jsonl` so reruns resume.