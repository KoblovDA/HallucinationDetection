"""Fine-tune a token classifier (ModernBERT) on our combined_train.jsonl.

Architecture follows LettuceDetect:
  - Input: tokenized `[CLS] context [SEP] question [SEP] answer [SEP]`
  - Labels: per-token 0 (supported) / 1 (hallucinated). context/question tokens get -100 (ignored).
  - Loss: standard CrossEntropyLoss over 2 classes per token.

Hallucination spans in our dataset are at the CHARACTER level on the `output` field. We use the
tokenizer's `offset_mapping` to convert char-spans → token labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def char_spans_to_token_labels(answer_offsets: list[tuple[int, int]],
                               spans: list[dict]) -> list[int]:
    """Each answer-token gets label 1 iff it overlaps any hallucination span (char range)."""
    labels = [0] * len(answer_offsets)
    for span in spans:
        s, e = int(span["start"]), int(span["end"])
        for i, (a, b) in enumerate(answer_offsets):
            if a == b:        # special token
                continue
            if a < e and b > s:
                labels[i] = 1
    return labels


@dataclass
class EncodedSample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]            # -100 for non-answer tokens; 0/1 for answer tokens
    answer_token_offsets: list[tuple[int, int]]   # char offsets of answer tokens (for inference)
    answer_token_indices: list[int]               # indices into input_ids that are answer tokens


def encode_sample(sample: dict, tokenizer, max_length: int = 4096) -> EncodedSample:
    """Encode `[CLS] context [SEP] question [SEP] answer [SEP]` with per-answer-token labels.

    LettuceDetect's input ordering is context first, then question, then answer.
    """
    context = sample["context"]
    question = sample["query"]
    answer = sample["output"]

    # We need to know where the answer starts in the concatenated input.
    cls = tokenizer.cls_token or ""
    sep = tokenizer.sep_token or ""

    # Tokenize each segment WITH offset_mapping for the answer (we need its char→token mapping).
    enc_context = tokenizer(context, add_special_tokens=False, truncation=False)
    enc_question = tokenizer(question, add_special_tokens=False, truncation=False)
    enc_answer = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True,
                           truncation=False)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Compose: [CLS] context [SEP] question [SEP] answer [SEP]
    ids = [cls_id]
    ids += enc_context["input_ids"]
    ids += [sep_id]
    ids += enc_question["input_ids"]
    ids += [sep_id]
    answer_start = len(ids)
    ids += enc_answer["input_ids"]
    answer_end = len(ids)
    ids += [sep_id]

    # Truncate if too long: drop excess context tokens (preserve answer).
    if len(ids) > max_length:
        excess = len(ids) - max_length
        ctx_len = len(enc_context["input_ids"])
        keep = max(0, ctx_len - excess)
        new_context_ids = enc_context["input_ids"][:keep]
        ids = ([cls_id] + new_context_ids + [sep_id]
               + enc_question["input_ids"] + [sep_id]
               + enc_answer["input_ids"] + [sep_id])
        answer_start = 1 + len(new_context_ids) + 1 + len(enc_question["input_ids"]) + 1
        answer_end = answer_start + len(enc_answer["input_ids"])

    attention_mask = [1] * len(ids)

    # Build per-token labels: -100 everywhere, 0/1 for answer tokens.
    labels = [-100] * len(ids)
    answer_offsets = enc_answer["offset_mapping"]
    answer_labels = char_spans_to_token_labels(answer_offsets, sample["hallucination_labels"])
    for i, lab in enumerate(answer_labels):
        labels[answer_start + i] = lab

    answer_token_indices = list(range(answer_start, answer_end))
    return EncodedSample(
        input_ids=ids,
        attention_mask=attention_mask,
        labels=labels,
        answer_token_offsets=answer_offsets,
        answer_token_indices=answer_token_indices,
    )


class HallucinationDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = 4096):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, Any]:
        enc = encode_sample(self.samples[idx], self.tokenizer, self.max_length)
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": enc.labels,
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    def pad(seq, value):
        return seq + [value] * (max_len - len(seq))
    return {
        "input_ids":      torch.tensor([pad(b["input_ids"], pad_token_id) for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0)         for b in batch], dtype=torch.long),
        "labels":         torch.tensor([pad(b["labels"], -100)              for b in batch], dtype=torch.long),
    }


def predict_spans(model, tokenizer, sample: dict, max_length: int = 4096,
                  threshold: float = 0.5, device: str = "cuda") -> list[dict]:
    """Run the trained classifier on one sample and decode char-spans from token probabilities."""
    enc = encode_sample(sample, tokenizer, max_length)
    inputs = {
        "input_ids":      torch.tensor([enc.input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([enc.attention_mask], dtype=torch.long, device=device),
    }
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0]                        # [seq_len, 2]
    probs = torch.softmax(logits, dim=-1)[:, 1]   # P(hallucinated) per token
    answer_probs = probs[enc.answer_token_indices].tolist()

    # Aggregate consecutive answer-tokens with prob > threshold into char spans.
    spans: list[dict] = []
    cur_start = cur_end = None
    cur_max = 0.0
    answer_text = sample["output"]
    for offset, p in zip(enc.answer_token_offsets, answer_probs):
        a, b = offset
        if a == b:
            continue
        if p > threshold:
            if cur_start is None:
                cur_start = a
            cur_end = b
            cur_max = max(cur_max, p)
        else:
            if cur_start is not None:
                spans.append({
                    "start": cur_start, "end": cur_end,
                    "text": answer_text[cur_start:cur_end],
                    "confidence": cur_max,
                })
                cur_start = cur_end = None
                cur_max = 0.0
    if cur_start is not None:
        spans.append({
            "start": cur_start, "end": cur_end,
            "text": answer_text[cur_start:cur_end],
            "confidence": cur_max,
        })
    return spans
