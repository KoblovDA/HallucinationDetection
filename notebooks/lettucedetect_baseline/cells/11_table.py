import pandas as pd

rows = []
for name, r in results.items():
    rows.append({
        "Split": name,
        "N": r["n"],
        "Span micro P": round(r["span_micro"].precision, 3),
        "Span micro R": round(r["span_micro"].recall, 3),
        "Span micro F1": round(r["span_micro"].f1, 3),
        "Span macro F1": round(r["span_macro_f1"], 3),
        "Ex P": round(r["example"].precision, 3),
        "Ex R": round(r["example"].recall, 3),
        "Ex F1": round(r["example"].f1, 3),
    })
df = pd.DataFrame(rows)
df
