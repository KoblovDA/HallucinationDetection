from tqdm.auto import tqdm
import pandas as pd

THRESHOLD = 0.5
model.eval()

def predict_spans(sample):
    e = encode_sample(sample, tokenizer, MAX_LENGTH)
    inputs = {
        "input_ids":      torch.tensor([e.input_ids],      dtype=torch.long, device=device),
        "attention_mask": torch.tensor([e.attention_mask], dtype=torch.long, device=device),
    }
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits[0], dim=-1)[:, 1]
    a_probs = probs[e.answer_token_indices].tolist()
    text = sample["output"]
    spans = []; cs = ce = None; cmax = 0.0
    for (a, b), p in zip(e.answer_token_offsets, a_probs):
        if a == b: continue
        if p > THRESHOLD:
            if cs is None: cs = a
            ce = b; cmax = max(cmax, p)
        else:
            if cs is not None:
                spans.append({"start": cs, "end": ce, "text": text[cs:ce], "confidence": cmax})
                cs = ce = None; cmax = 0.0
    if cs is not None:
        spans.append({"start": cs, "end": ce, "text": text[cs:ce], "confidence": cmax})
    return spans

test_preds = [predict_spans(s) for s in tqdm(test_samples, desc="Inference (combined_test)")]

def filter_subset(label):
    s, p = [], []
    for x, y in zip(test_samples, test_preds):
        if not x["hallucination_labels"]:
            s.append(x); p.append(y); continue
        if x["hallucination_labels"][0].get("type") == label:
            s.append(x); p.append(y)
    return s, p

subsets = {
    "Combined (all + clean)":      (test_samples, test_preds),
    "Type 1 + clean":              filter_subset("Type1_Contradiction"),
    "Type 2 + clean":              filter_subset("Type2_Overgeneration"),
    "Type 3 + clean":              filter_subset("Type3_MissingTool"),
}
rows = []
for name, (s, p) in subsets.items():
    sp, sr, sf = span_micro_f1(s, p)
    ep, er, ef = example_f1(s, p)
    rows.append({"Split": name, "N": len(s),
                 "Span P": round(sp, 3), "Span R": round(sr, 3), "Span F1": round(sf, 3),
                 "Ex P": round(ep, 3), "Ex R": round(er, 3), "Ex F1": round(ef, 3)})
    print(f"{name} (N={len(s)}): span P/R/F1 = {sp:.3f}/{sr:.3f}/{sf:.3f} | ex P/R/F1 = {ep:.3f}/{er:.3f}/{ef:.3f}")
pd.DataFrame(rows)
