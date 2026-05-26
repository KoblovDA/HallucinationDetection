import pandas as pd

def predict_spans(s):
    out = extract_lookback_ratios(model, tokenizer, s, device=device)
    feats, idx_lists = sliding_windows(out.ratios, WINDOW, STRIDE)
    if feats.shape[0] == 0:
        return []
    scores = clf.predict_proba(feats.astype(np.float32))[:, 1]
    return windows_to_spans(out.answer_token_offsets, idx_lists, scores, THRESHOLD)

test_preds = [predict_spans(s) for s in tqdm(test_samples, desc="test inference")]

def filter_subset(label):
    sel_s, sel_p = [], []
    for s, p in zip(test_samples, test_preds):
        if not s["hallucination_labels"]:
            sel_s.append(s); sel_p.append(p); continue
        if s["hallucination_labels"][0].get("type") == label:
            sel_s.append(s); sel_p.append(p)
    return sel_s, sel_p

subsets = {
    "Combined":           (test_samples, test_preds),
    "Type 1 + clean":     filter_subset("Type1_Contradiction"),
    "Type 2 + clean":     filter_subset("Type2_Overgeneration"),
    "Type 3 + clean":     filter_subset("Type3_MissingTool"),
}
rows = []
for name, (s, p) in subsets.items():
    sp, sr, sf = span_micro_f1(s, p)
    macro = span_macro_f1(s, p)
    ep, er, ef = example_f1(s, p)
    rows.append({"Split": name, "N": len(s),
                 "Span P": round(sp, 3), "Span R": round(sr, 3), "Span F1": round(sf, 3),
                 "Span macro F1": round(macro, 3),
                 "Ex P": round(ep, 3), "Ex R": round(er, 3), "Ex F1": round(ef, 3)})
    print(f"{name} (N={len(s)}): span P/R/F1 = {sp:.3f}/{sr:.3f}/{sf:.3f} | macro F1 = {macro:.3f} "
          f"| ex P/R/F1 = {ep:.3f}/{er:.3f}/{ef:.3f}")
pd.DataFrame(rows)
