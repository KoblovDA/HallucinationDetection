"""LookBackLens baseline (Chuang et al., 2024).

Per-token lookback ratio:
    LR^{l,h}_t = A_ctx / (A_ctx + A_new)
where A_ctx = mean attention from answer-token t back to context tokens,
and A_new = mean attention from t to previously-generated answer tokens.

Sliding window setup (paper §2.2):
- Fixed-size chunk window over answer tokens.
- Train chunk label = 1 if any token in the chunk overlaps a gold hallucinated span, else 0.
- Logistic regression on the L*H mean lookback features per chunk.
- At inference, slide the same window over each test answer; consecutive positive chunks become
  a single char span in the answer text.

Backbone is any causal HuggingFace transformer that supports `output_attentions=True`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExtractionOut:
    ratios: np.ndarray             # [L, H, num_answer_tokens]  float16
    answer_token_offsets: np.ndarray  # [num_answer_tokens, 2]
    answer_text: str


@torch.no_grad()
def extract_lookback_ratios(model, tokenizer, sample: dict, device: str = "cuda",
                            max_length: int = 4096) -> ExtractionOut:
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


def tok_overlap(offsets: np.ndarray, start_char: int, end_char: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if a < end_char and b > start_char and a != b]


def chunk_feature(ratios: np.ndarray, idxs: list[int]) -> np.ndarray:
    if not idxs:
        return np.zeros(ratios.shape[0] * ratios.shape[1], dtype=np.float16)
    return ratios[:, :, idxs].mean(axis=-1).reshape(-1).astype(np.float16)


def sliding_windows(ratios: np.ndarray, window: int, stride: int
                    ) -> tuple[np.ndarray, list[list[int]]]:
    n = ratios.shape[-1]
    if n == 0:
        return np.zeros((0, ratios.shape[0] * ratios.shape[1]), dtype=np.float16), []
    if n < window:
        window = n
    feats, idx_lists = [], []
    for s in range(0, n - window + 1, stride):
        idxs = list(range(s, s + window))
        feats.append(chunk_feature(ratios, idxs))
        idx_lists.append(idxs)
    return np.stack(feats, axis=0), idx_lists


def merge_positive_windows(offsets: np.ndarray, idx_lists: list[list[int]],
                           scores: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Paper-style aggregation: merge consecutive positive chunks into one char span."""
    spans: list[dict] = []
    if not idx_lists:
        return spans
    cur_start = cur_end = None
    cur_max = 0.0
    for idxs, sc in zip(idx_lists, scores):
        if sc > threshold:
            first_tok, last_tok = idxs[0], idxs[-1]
            chunk_start = int(offsets[first_tok, 0])
            chunk_end = int(offsets[last_tok, 1])
            if cur_start is None:
                cur_start = chunk_start
                cur_end = chunk_end
            else:
                # Adjacent / overlapping with current span?
                if chunk_start <= cur_end:
                    cur_end = max(cur_end, chunk_end)
                else:
                    spans.append({"start": cur_start, "end": cur_end, "confidence": cur_max})
                    cur_start = chunk_start; cur_end = chunk_end; cur_max = 0.0
            cur_max = max(cur_max, float(sc))
        else:
            if cur_start is not None:
                spans.append({"start": cur_start, "end": cur_end, "confidence": cur_max})
                cur_start = cur_end = None
                cur_max = 0.0
    if cur_start is not None:
        spans.append({"start": cur_start, "end": cur_end, "confidence": cur_max})
    return spans
