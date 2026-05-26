from tqdm.auto import tqdm

train_samples = read_jsonl(DATA_DIR / "combined_train.jsonl")
val_samples   = read_jsonl(DATA_DIR / "combined_val.jsonl")
test_samples  = read_jsonl(DATA_DIR / "combined_test.jsonl")
print(f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

# TRAIN: sliding window over every answer; label each window 1 if it overlaps ANY gold span, 0 otherwise.
train_X, train_y = [], []
for s in tqdm(train_samples, desc="train extract"):
    out = extract_lookback_ratios(model, tokenizer, s, device=device)
    feats, idx_lists = sliding_windows(out.ratios, WINDOW, STRIDE)
    # gold token indices for this sample
    gold_idxs = set()
    for span in s["hallucination_labels"]:
        gold_idxs.update(_tok_overlap(out.answer_token_offsets, int(span["start"]), int(span["end"])))
    for feat, idxs in zip(feats, idx_lists):
        train_X.append(feat)
        train_y.append(1 if any(i in gold_idxs for i in idxs) else 0)

train_X = np.stack(train_X, axis=0).astype(np.float32)
train_y = np.array(train_y, dtype=np.int64)
print(f"\nTRAIN windows: {train_X.shape}  pos={int((train_y==1).sum())} neg={int((train_y==0).sum())}")
