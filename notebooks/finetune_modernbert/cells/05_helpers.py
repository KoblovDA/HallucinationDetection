import json
from dataclasses import dataclass

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def char_spans_to_token_labels(answer_offsets, spans):
    labels = [0] * len(answer_offsets)
    for span in spans:
        s, e = int(span["start"]), int(span["end"])
        for i, (a, b) in enumerate(answer_offsets):
            if a == b: continue
            if a < e and b > s:
                labels[i] = 1
    return labels

@dataclass
class Encoded:
    input_ids: list
    attention_mask: list
    labels: list
    answer_token_offsets: list
    answer_token_indices: list

def encode_sample(sample, tokenizer, max_length=4096):
    enc_ctx = tokenizer(sample["context"],  add_special_tokens=False, truncation=False)
    enc_q   = tokenizer(sample["query"],    add_special_tokens=False, truncation=False)
    enc_a   = tokenizer(sample["output"],   add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    ids = [cls_id] + enc_ctx["input_ids"] + [sep_id] + enc_q["input_ids"] + [sep_id]
    answer_start = len(ids)
    ids += enc_a["input_ids"]
    answer_end = len(ids)
    ids += [sep_id]
    if len(ids) > max_length:
        excess = len(ids) - max_length
        ctx_len = len(enc_ctx["input_ids"])
        keep = max(0, ctx_len - excess)
        new_ctx = enc_ctx["input_ids"][:keep]
        ids = [cls_id] + new_ctx + [sep_id] + enc_q["input_ids"] + [sep_id] + enc_a["input_ids"] + [sep_id]
        answer_start = 1 + len(new_ctx) + 1 + len(enc_q["input_ids"]) + 1
        answer_end = answer_start + len(enc_a["input_ids"])
    attn = [1] * len(ids)
    labels = [-100] * len(ids)
    a_labels = char_spans_to_token_labels(enc_a["offset_mapping"], sample["hallucination_labels"])
    for i, lab in enumerate(a_labels):
        labels[answer_start + i] = lab
    return Encoded(ids, attn, labels, enc_a["offset_mapping"],
                   list(range(answer_start, answer_end)))

# Metrics
def _char_set(spans):
    out = set()
    for s in spans:
        out.update(range(int(s["start"]), int(s["end"])))
    return out

def span_micro_f1(samples, pred_spans):
    o = p = g = 0
    for s, ps in zip(samples, pred_spans):
        gs = _char_set(s["hallucination_labels"]); pset = _char_set(ps)
        o += len(gs & pset); p += len(pset); g += len(gs)
    pr = o/p if p else 0.0; re = o/g if g else 0.0
    f1 = 2*pr*re/(pr+re) if (pr+re) else 0.0
    return pr, re, f1

def example_f1(samples, pred_spans):
    tp = fp = fn = 0
    for s, ps in zip(samples, pred_spans):
        gold = bool(s["hallucination_labels"]); pred = bool(ps)
        if gold and pred: tp += 1
        elif gold: fn += 1
        elif pred: fp += 1
    pr = tp/(tp+fp) if (tp+fp) else 0.0
    re = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*pr*re/(pr+re) if (pr+re) else 0.0
    return pr, re, f1

train_samples = read_jsonl(DATA_DIR / "combined_train.jsonl")
val_samples   = read_jsonl(DATA_DIR / "combined_val.jsonl")
test_samples  = read_jsonl(DATA_DIR / "combined_test.jsonl")

# Oversample clean samples in train (clean = empty hallucination_labels).
if CLEAN_OVERSAMPLE > 1:
    clean = [s for s in train_samples if not s["hallucination_labels"]]
    hallu = [s for s in train_samples if s["hallucination_labels"]]
    train_samples = hallu + clean * CLEAN_OVERSAMPLE
    pct = 100 * len(clean) * CLEAN_OVERSAMPLE / len(train_samples)
    print(f"Oversampled clean × {CLEAN_OVERSAMPLE}: train = {len(hallu)} hallu + {len(clean)*CLEAN_OVERSAMPLE} clean ({pct:.0f}% clean)")
print(f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
