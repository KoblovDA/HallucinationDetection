from tqdm.auto import tqdm

train_samples = read_jsonl(DATA_DIR / "combined_train.jsonl")
val_samples   = read_jsonl(DATA_DIR / "combined_val.jsonl")
test_samples  = read_jsonl(DATA_DIR / "combined_test.jsonl")
print(f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

rng = np.random.default_rng(SEED)

# --- TRAIN features: positives (gold spans) + negatives (random non-gold windows)
train_X, train_y, train_meta = [], [], []
for s in tqdm(train_samples, desc="train extract"):
    out = extract_lookback_ratios(model, tokenizer, s, device=device)
    gold_idxs = []
    for span in s["hallucination_labels"]:
        gold_idxs.extend(_tok_overlap(out.answer_token_offsets, int(span["start"]), int(span["end"])))
    gold_idxs = sorted(set(gold_idxs))
    # positive
    if gold_idxs:
        train_X.append(span_feature(out.ratios, gold_idxs)); train_y.append(1)
        train_meta.append({"id": s["id"], "type": "pos", "n_tokens": len(gold_idxs)})
    # negatives
    neg_window = max(2, len(gold_idxs) if gold_idxs else WINDOW)
    for chunk in random_negative_chunks(out.answer_token_offsets, set(gold_idxs),
                                        neg_window, N_NEG_PER_SAMPLE, rng):
        train_X.append(span_feature(out.ratios, chunk)); train_y.append(0)
        train_meta.append({"id": s["id"], "type": "neg", "n_tokens": neg_window})

train_X = np.stack(train_X, axis=0).astype(np.float32)
train_y = np.array(train_y, dtype=np.int64)
print(f"\nTRAIN features: {train_X.shape}  pos={int((train_y==1).sum())} neg={int((train_y==0).sum())}")
