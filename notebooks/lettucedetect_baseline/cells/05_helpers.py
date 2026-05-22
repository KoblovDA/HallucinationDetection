import json
from dataclasses import dataclass


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _char_set(spans):
    chars = set()
    for s in spans:
        chars.update(range(int(s["start"]), int(s["end"])))
    return chars


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float


def span_metrics(samples, pred_spans):
    micro_overlap = micro_pred = micro_gold = 0
    macro_f1 = []
    for sample, preds in zip(samples, pred_spans):
        gold = _char_set(sample["hallucination_labels"])
        pred = _char_set(preds)
        overlap = len(gold & pred)
        micro_overlap += overlap
        micro_pred += len(pred)
        micro_gold += len(gold)
        if not pred and not gold:
            macro_f1.append(1.0); continue
        p = overlap / len(pred) if pred else 0.0
        r = overlap / len(gold) if gold else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        macro_f1.append(f1)
    p_mi = micro_overlap / micro_pred if micro_pred else 0.0
    r_mi = micro_overlap / micro_gold if micro_gold else 0.0
    f_mi = 2 * p_mi * r_mi / (p_mi + r_mi) if (p_mi + r_mi) > 0 else 0.0
    f_ma = sum(macro_f1) / len(macro_f1) if macro_f1 else 0.0
    return Metrics(p_mi, r_mi, f_mi), f_ma


def example_metrics(samples, pred_spans):
    tp = fp = fn = 0
    for sample, preds in zip(samples, pred_spans):
        gold = 1 if sample["hallucination_labels"] else 0
        pred = 1 if preds else 0
        if gold == 1 and pred == 1: tp += 1
        elif gold == 1 and pred == 0: fn += 1
        elif gold == 0 and pred == 1: fp += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return Metrics(p, r, f1)
