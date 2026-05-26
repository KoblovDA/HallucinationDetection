import json
from dataclasses import dataclass

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

# ----- metrics -----
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

def span_macro_f1(samples, pred_spans):
    fs = []
    for s, ps in zip(samples, pred_spans):
        gs = _char_set(s["hallucination_labels"]); pset = _char_set(ps)
        o = len(gs & pset)
        if not pset and not gs:
            fs.append(1.0); continue
        pr = o/len(pset) if pset else 0.0
        re = o/len(gs) if gs else 0.0
        fs.append(2*pr*re/(pr+re) if (pr+re) else 0.0)
    return sum(fs)/len(fs) if fs else 0.0

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

# ----- LookBackLens core: forward pass + per-token lookback ratios -----
import numpy as np
import torch

@dataclass
class ExtractionOut:
    ratios: np.ndarray
    answer_token_offsets: np.ndarray
    answer_text: str

@torch.no_grad()
def extract_lookback_ratios(model, tokenizer, sample, device="cuda", max_length=MAX_LENGTH):
    context_block = f"{sample['context']}\n\nQuestion: {sample['query']}\n\nAnswer: "
    enc_ctx = tokenizer(context_block, add_special_tokens=True, return_tensors="pt")
    enc_ans = tokenizer(sample["output"], add_special_tokens=False, return_tensors="pt",
                        return_offsets_mapping=True)
    input_ids = torch.cat([enc_ctx["input_ids"], enc_ans["input_ids"]], dim=1)
    if input_ids.size(1) > max_length:
        excess = input_ids.size(1) - max_length
        ctx_len = enc_ctx["input_ids"].size(1)
        keep = max(0, ctx_len - excess)
        input_ids = torch.cat([enc_ctx["input_ids"][:, :keep], enc_ans["input_ids"]], dim=1)
    attention_mask = torch.ones_like(input_ids)
    context_end = input_ids.size(1) - enc_ans["input_ids"].size(1)
    num_ans = enc_ans["input_ids"].size(1)

    input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_attentions=True, use_cache=False)
    L = len(out.attentions); H = out.attentions[0].size(1)

    tril = torch.tril(torch.ones(num_ans, num_ans, device=device, dtype=torch.float16), diagonal=-1)
    counts = torch.arange(num_ans, device=device, dtype=torch.float16).clamp(min=1)

    ratios = torch.empty(L, H, num_ans, device=device, dtype=torch.float16)
    for l, attn in enumerate(out.attentions):
        a = attn[0].to(torch.float16)
        ans_rows = a[:, context_end:context_end + num_ans, :]
        att_ctx = ans_rows[:, :, :context_end].mean(dim=-1)
        a2a = ans_rows[:, :, context_end:context_end + num_ans]
        att_new = (a2a * tril).sum(dim=-1) / counts
        att_new[:, 0] = 0
        ratios[l] = att_ctx / (att_ctx + att_new + 1e-6)

    return ExtractionOut(
        ratios=ratios.cpu().numpy(),
        answer_token_offsets=enc_ans["offset_mapping"][0].numpy(),
        answer_text=sample["output"],
    )

def _tok_overlap(offsets, s, e):
    return [i for i, (a, b) in enumerate(offsets) if a < e and b > s and a != b]

def span_feature(ratios, idxs):
    if not idxs:
        return np.zeros(ratios.shape[0] * ratios.shape[1], dtype=np.float16)
    return ratios[:, :, idxs].mean(axis=-1).reshape(-1).astype(np.float16)

def random_negative_chunks(offsets, gold_idxs, window, n_chunks, rng):
    n = offsets.shape[0]
    if n < window: return []
    starts = [s for s in range(0, n - window + 1)
              if not any(i in gold_idxs for i in range(s, s + window))]
    if not starts: return []
    rng.shuffle(starts)
    chosen, used = [], set()
    for s in starts:
        if any(i in used for i in range(s, s + window)): continue
        chosen.append(list(range(s, s + window)))
        used.update(range(s, s + window))
        if len(chosen) >= n_chunks: break
    return chosen

def sliding_windows(ratios, window, stride):
    n = ratios.shape[-1]
    if n == 0: return np.zeros((0, ratios.shape[0]*ratios.shape[1]), dtype=np.float16), []
    if n < window: window = n
    feats, idx_lists = [], []
    for s in range(0, n - window + 1, stride):
        idxs = list(range(s, s + window))
        feats.append(span_feature(ratios, idxs)); idx_lists.append(idxs)
    return np.stack(feats), idx_lists

def windows_to_spans(offsets, idx_lists, scores, threshold):
    if not idx_lists: return []
    n = offsets.shape[0]
    token_score = np.zeros(n, dtype=np.float32)
    for idxs, s in zip(idx_lists, scores):
        for i in idxs:
            if s > token_score[i]: token_score[i] = s
    flags = token_score > threshold
    spans, cs, ce, cm = [], None, None, 0.0
    for i, (a, b) in enumerate(offsets):
        if a == b: continue
        if flags[i]:
            if cs is None: cs = int(a)
            ce = int(b); cm = max(cm, float(token_score[i]))
        else:
            if cs is not None:
                spans.append({"start": cs, "end": ce, "confidence": cm})
                cs = ce = None; cm = 0.0
    if cs is not None:
        spans.append({"start": cs, "end": ce, "confidence": cm})
    return spans
