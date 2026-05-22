"""Span-level and example-level metrics for hallucination detection.

Span-level (RAGTruth-style): character-overlap precision/recall/F1.
  - micro: sum of overlap chars across all samples / sum of (gold or predicted) chars
  - macro: average per-sample F1

Example-level: binary 'has any hallucination span' precision/recall/F1.
  - In our test sets, every sample has a hallucination → gold=1 always.
  - We still compute (precision/recall/F1 by treating "detector found ≥1 span" as positive prediction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1}


def _char_set(spans: Iterable[dict]) -> set[int]:
    chars: set[int] = set()
    for s in spans:
        chars.update(range(int(s["start"]), int(s["end"])))
    return chars


def per_sample_span_prf(gold_spans: list[dict], pred_spans: list[dict]) -> tuple[int, int, int]:
    """Return (overlap_chars, pred_chars, gold_chars) for one sample."""
    gold = _char_set(gold_spans)
    pred = _char_set(pred_spans)
    return len(gold & pred), len(pred), len(gold)


def span_metrics(samples: list[dict], pred_spans_per_sample: list[list[dict]],
                 ) -> tuple[Metrics, Metrics]:
    """Compute micro and macro span-level metrics across the dataset.

    Each `samples[i]` is expected to have `hallucination_labels: list[span]`.
    `pred_spans_per_sample[i]` is the model's predicted spans for that sample.
    """
    assert len(samples) == len(pred_spans_per_sample)
    micro_overlap = 0
    micro_pred = 0
    micro_gold = 0
    macro_f1s: list[float] = []
    for sample, preds in zip(samples, pred_spans_per_sample):
        gold = sample["hallucination_labels"]
        overlap, pred_n, gold_n = per_sample_span_prf(gold, preds)
        micro_overlap += overlap
        micro_pred += pred_n
        micro_gold += gold_n
        if pred_n == 0 and gold_n == 0:
            macro_f1s.append(1.0)
            continue
        p = overlap / pred_n if pred_n else 0.0
        r = overlap / gold_n if gold_n else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        macro_f1s.append(f1)

    p_mi = micro_overlap / micro_pred if micro_pred else 0.0
    r_mi = micro_overlap / micro_gold if micro_gold else 0.0
    f_mi = 2 * p_mi * r_mi / (p_mi + r_mi) if (p_mi + r_mi) > 0 else 0.0
    f_ma = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    return Metrics(p_mi, r_mi, f_mi), Metrics(0.0, 0.0, f_ma)


def example_metrics(samples: list[dict], pred_spans_per_sample: list[list[dict]],
                    ) -> Metrics:
    """Binary 'has hallucination' precision/recall/F1.
    gold = 1 iff sample has ≥1 hallucination_label
    pred = 1 iff model predicted ≥1 span"""
    tp = fp = fn = tn = 0
    for sample, preds in zip(samples, pred_spans_per_sample):
        gold = 1 if sample["hallucination_labels"] else 0
        pred = 1 if preds else 0
        if gold == 1 and pred == 1: tp += 1
        elif gold == 1 and pred == 0: fn += 1
        elif gold == 0 and pred == 1: fp += 1
        else: tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return Metrics(p, r, f1)
