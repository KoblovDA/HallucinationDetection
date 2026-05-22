def show(sample, preds, n=3, width=120):
    out = sample["output"]
    gold = sample["hallucination_labels"]
    print(f"--- {sample.get('id', '?')} ---")
    for g in gold:
        s, e = g["start"], g["end"]
        print(f"  GOLD: [{s}..{e}] {out[s:e]!r}")
    if preds:
        for p in preds:
            s, e = p["start"], p["end"]
            print(f"  PRED: [{s}..{e}] {out[s:e]!r}  (conf={p.get('confidence', 0):.2f})")
    else:
        print(f"  PRED: (none)")
    print()

for name, r in results.items():
    print(f"\n=== {name} ===")
    for i in range(min(3, r["n"])):
        show(r["samples"][i], r["preds"][i])
