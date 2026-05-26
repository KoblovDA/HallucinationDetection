"""LookBackLens baseline (Chuang et al., 2024).

The attention-map signal for each answer token is:
    LR^{l,h}_t = A_ctx / (A_ctx + A_new)
where A_ctx = mean attention from answer-token t to context tokens (positions < context_end),
and A_new = mean attention from t to previously-generated answer tokens (positions in [context_end, t)).

For each predefined span we average LR^{l,h} across the span tokens to obtain a feature vector
of length L*H. A linear classifier is trained on these features. At inference time we slide a
fixed-size window across the answer and classify each window; consecutive positive windows are
merged into char spans.

Backbone-LM is arbitrary (any causal HuggingFace transformer with `output_attentions=True`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ExtractionOut:
    """Per-token features for one sample."""
    ratios: np.ndarray             # [L, H, num_answer_tokens]  float16
    answer_token_offsets: np.ndarray  # [num_answer_tokens, 2]  char offsets in `output`
    answer_text: str


@torch.no_grad()
def extract_lookback_ratios(model, tokenizer, sample: dict, device: str = "cuda",
                            max_length: int = 4096) -> ExtractionOut:
    """Run one forward pass and return per-token lookback ratios over the answer span.

    Input format follows LettuceDetect: ``context [SEP] question`` as the "context" segment,
    then the answer concatenated. Special tokens for the boundary are model-specific; we use the
    tokenizer's pad/sep/eos in a generic way. For decoder-only LMs without a SEP we just join with
    newlines, which is fine for the lookback ratio (it's all "context" relative to the answer).
    """
    context_block = f"{sample['context']}\n\nQuestion: {sample['query']}\n\nAnswer: "
    enc_ctx = tokenizer(context_block, add_special_tokens=True, return_tensors="pt")
    enc_ans = tokenizer(sample["output"], add_special_tokens=False, return_tensors="pt",
                        return_offsets_mapping=True)

    input_ids = torch.cat([enc_ctx["input_ids"], enc_ans["input_ids"]], dim=1)
    attention_mask = torch.ones_like(input_ids)

    # Truncate context if too long; never truncate answer.
    if input_ids.size(1) > max_length:
        excess = input_ids.size(1) - max_length
        ctx_len = enc_ctx["input_ids"].size(1)
        keep = max(0, ctx_len - excess)
        input_ids = torch.cat([enc_ctx["input_ids"][:, :keep], enc_ans["input_ids"]], dim=1)
        attention_mask = torch.ones_like(input_ids)

    context_end = input_ids.size(1) - enc_ans["input_ids"].size(1)
    answer_len = enc_ans["input_ids"].size(1)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_attentions=True, use_cache=False)
    # attentions: tuple of L tensors, each [1, H, seq, seq]
    L = len(outputs.attentions)
    H = outputs.attentions[0].size(1)
    num_ans = answer_len

    # Pre-compute the lower-triangular mask for "previously generated" within the answer.
    tril_mask = torch.tril(torch.ones(num_ans, num_ans, device=device, dtype=torch.float16),
                           diagonal=-1)
    counts = torch.arange(num_ans, device=device, dtype=torch.float16).clamp(min=1)

    ratios_per_layer = torch.empty(L, H, num_ans, device=device, dtype=torch.float16)
    for l, attn in enumerate(outputs.attentions):
        a = attn[0].to(torch.float16)  # [H, seq, seq]
        # Rows for answer tokens
        answer_rows = a[:, context_end:context_end + num_ans, :]            # [H, num_ans, seq]
        att_ctx = answer_rows[:, :, :context_end].mean(dim=-1)              # [H, num_ans]
        ans_to_ans = answer_rows[:, :, context_end:context_end + num_ans]   # [H, num_ans, num_ans]
        masked = ans_to_ans * tril_mask                                     # broadcast over H
        att_new = masked.sum(dim=-1) / counts                                # [H, num_ans]
        att_new[:, 0] = 0
        ratios_per_layer[l] = att_ctx / (att_ctx + att_new + 1e-6)

    return ExtractionOut(
        ratios=ratios_per_layer.cpu().numpy(),
        answer_token_offsets=enc_ans["offset_mapping"][0].numpy(),
        answer_text=sample["output"],
    )


def _token_indices_overlapping(offsets: np.ndarray, start_char: int, end_char: int) -> list[int]:
    """Return indices of tokens whose char-offsets overlap [start_char, end_char)."""
    idxs: list[int] = []
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if a < end_char and b > start_char:
            idxs.append(i)
    return idxs


def span_feature(ratios: np.ndarray, token_indices: list[int]) -> np.ndarray:
    """Average lookback ratios across the given answer token indices.
    Returns a flat feature vector of length L*H (float16)."""
    if not token_indices:
        return np.zeros(ratios.shape[0] * ratios.shape[1], dtype=np.float16)
    slc = ratios[:, :, token_indices]              # [L, H, k]
    mean = slc.mean(axis=-1)                       # [L, H]
    return mean.reshape(-1).astype(np.float16)


def gold_token_indices(sample: dict, offsets: np.ndarray) -> list[int]:
    """Token indices covered by any gold hallucination_label span."""
    out: list[int] = []
    for span in sample.get("hallucination_labels", []):
        out.extend(_token_indices_overlapping(
            offsets, int(span["start"]), int(span["end"])))
    return sorted(set(out))


def random_negative_chunks(offsets: np.ndarray, gold_idxs: set[int], window: int,
                           n_chunks: int, rng: np.random.Generator) -> list[list[int]]:
    """Pick `n_chunks` random non-overlapping windows of size `window` from answer tokens
    that do not overlap any gold token."""
    n = offsets.shape[0]
    valid_starts = []
    for s in range(0, n - window + 1):
        if not any(i in gold_idxs for i in range(s, s + window)):
            valid_starts.append(s)
    if not valid_starts:
        return []
    rng.shuffle(valid_starts)
    chosen: list[list[int]] = []
    used = set()
    for s in valid_starts:
        if any(i in used for i in range(s, s + window)):
            continue
        chosen.append(list(range(s, s + window)))
        used.update(range(s, s + window))
        if len(chosen) >= n_chunks:
            break
    return chosen


def sliding_window_features(ratios: np.ndarray, window: int, stride: int
                            ) -> tuple[np.ndarray, list[list[int]]]:
    """Compute features for sliding windows over answer tokens.

    Returns (features [num_windows, L*H], token_index_lists per window).
    """
    n = ratios.shape[-1]
    if n < window:
        if n == 0:
            return np.zeros((0, ratios.shape[0] * ratios.shape[1]), dtype=np.float16), []
        window = n
    feats: list[np.ndarray] = []
    idx_lists: list[list[int]] = []
    for s in range(0, n - window + 1, stride):
        idxs = list(range(s, s + window))
        feats.append(span_feature(ratios, idxs))
        idx_lists.append(idxs)
    return np.stack(feats, axis=0), idx_lists


def windows_to_spans(answer_token_offsets: np.ndarray, idx_lists: list[list[int]],
                     scores: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Merge consecutive positive windows into character spans of the answer text."""
    if len(idx_lists) == 0:
        return []
    # For each token, take the MAX score across all windows containing it.
    n_tokens = answer_token_offsets.shape[0]
    token_score = np.zeros(n_tokens, dtype=np.float32)
    token_seen = np.zeros(n_tokens, dtype=bool)
    for window_idxs, s in zip(idx_lists, scores):
        for i in window_idxs:
            if not token_seen[i] or token_score[i] < s:
                token_score[i] = s
            token_seen[i] = True
    flags = token_score > threshold

    spans: list[dict] = []
    cur_start = cur_end = None
    cur_max = 0.0
    for i, (a, b) in enumerate(answer_token_offsets):
        if a == b:
            continue
        if flags[i]:
            if cur_start is None:
                cur_start = int(a)
            cur_end = int(b)
            cur_max = max(cur_max, float(token_score[i]))
        else:
            if cur_start is not None:
                spans.append({"start": cur_start, "end": cur_end, "confidence": cur_max})
                cur_start = cur_end = None
                cur_max = 0.0
    if cur_start is not None:
        spans.append({"start": cur_start, "end": cur_end, "confidence": cur_max})
    return spans
