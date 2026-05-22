from tqdm.auto import tqdm

def predict(sample):
    raw = detector.predict(
        context=[sample["context"]],
        question=sample["query"],
        answer=sample["output"],
        output_format="spans",
    )
    return [
        {"start": int(s["start"]), "end": int(s["end"]),
         "text": s.get("text", sample["output"][int(s["start"]):int(s["end"])]),
         "confidence": float(s.get("confidence", 0.0))}
        for s in raw
    ]

# Inference once on combined_test (599 samples = 150 T1 + 150 T2 + 149 T3 + 150 clean).
samples_all = read_jsonl(DATA_DIR / "combined_test.jsonl")
preds_all = [predict(s) for s in tqdm(samples_all, desc="Inference (combined_test)")]

def filter_subset(type_label):
    """Return (samples, preds) restricted to samples of `type_label` plus all clean samples."""
    sel_s, sel_p = [], []
    for s, p in zip(samples_all, preds_all):
        if not s["hallucination_labels"]:
            sel_s.append(s); sel_p.append(p)
            continue
        if s["hallucination_labels"][0].get("type") == type_label:
            sel_s.append(s); sel_p.append(p)
    return sel_s, sel_p

subsets = {
    "Combined (all types + clean)":   (samples_all, preds_all),
    "Type 1 + clean":                 filter_subset("Type1_Contradiction"),
    "Type 2 + clean":                 filter_subset("Type2_Overgeneration"),
    "Type 3 + clean":                 filter_subset("Type3_MissingTool"),
}

results = {}
for name, (s, p) in subsets.items():
    span_micro, span_macro_f1 = span_metrics(s, p)
    ex = example_metrics(s, p)
    results[name] = {
        "n": len(s),
        "span_micro": span_micro,
        "span_macro_f1": span_macro_f1,
        "example": ex,
        "preds": p,
        "samples": s,
    }
    print(f"\n{name} (N={len(s)}): "
          f"span micro P/R/F1 = {span_micro.precision:.3f} / {span_micro.recall:.3f} / {span_micro.f1:.3f} | "
          f"span macro F1 = {span_macro_f1:.3f} | "
          f"example P/R/F1 = {ex.precision:.3f} / {ex.recall:.3f} / {ex.f1:.3f}")
