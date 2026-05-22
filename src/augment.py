"""Augmentation pipeline: produce N diverse hallucinations per source triple.

Strategy:
1. Collect all valid rule-based swaps for the source (every applicable strategy on every leaf value).
2. Pick up to `n` candidates with diversity in strategies (round-robin) and fields.
3. If fewer than `n` are available, call the LLM with high temperature, excluding already-used
   `original_substring`s, until we reach `n` or hit a per-source LLM call cap.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .data import Triple
from .injection import Span, collect_all_swaps
from .llm_inject import OpenRouterError, USER_PROMPT_TEMPLATE, SYSTEM_PROMPT, call_openrouter, _extract_json


@dataclass
class Variant:
    output: str
    span: Span


def select_diverse(candidates: list[tuple[str, Span]], n: int,
                   rng: random.Random) -> list[Variant]:
    """Round-robin pick up to n candidates, preferring strategy diversity, then field diversity."""
    if not candidates:
        return []
    by_strategy: dict[str, list[tuple[str, Span]]] = defaultdict(list)
    for c in candidates:
        by_strategy[c[1].strategy].append(c)
    for items in by_strategy.values():
        rng.shuffle(items)
    strategies = list(by_strategy.keys())
    rng.shuffle(strategies)
    chosen: list[Variant] = []
    used_fields: set[str] = set()
    # First pass: one per strategy, preferring unseen fields
    while len(chosen) < n and any(by_strategy[s] for s in strategies):
        progressed = False
        for s in strategies:
            if not by_strategy[s] or len(chosen) >= n:
                continue
            # prefer a candidate with an unused field
            chosen_idx = None
            for i, c in enumerate(by_strategy[s]):
                if c[1].field not in used_fields:
                    chosen_idx = i
                    break
            if chosen_idx is None:
                chosen_idx = 0
            text, span = by_strategy[s].pop(chosen_idx)
            chosen.append(Variant(output=text, span=span))
            used_fields.add(span.field)
            progressed = True
        if not progressed:
            break
    return chosen[:n]


def _validate_swap(answer: str, orig: str, new: str) -> bool:
    if not orig or not new or orig == new:
        return False
    if orig not in answer:
        return False
    idx = answer.find(orig)
    left = answer[idx - 1] if idx > 0 else ""
    right = answer[idx + len(orig)] if idx + len(orig) < len(answer) else ""
    if left.isalpha() and orig[0].isalpha():
        return False
    if right.isalpha() and orig[-1].isalpha():
        return False
    return True


def llm_propose_excluding(triple: Triple, exclude_originals: list[str],
                          api_key: str, temperature: float = 0.9,
                          model: str = "qwen/qwen3-235b-a22b-2507") -> Variant | None:
    extra = ""
    if exclude_originals:
        exc = ", ".join(repr(e) for e in exclude_originals)
        extra = f"\n\nIMPORTANT: choose a DIFFERENT fact than these already-used spans: {exc}"
    user_msg = USER_PROMPT_TEMPLATE.format(
        tool_output=triple.tool_output_raw,
        user_query=triple.user,
        answer=triple.assistant,
    ) + extra
    try:
        raw = call_openrouter(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_msg}],
            model=model, api_key=api_key, temperature=temperature,
        )
        parsed = _extract_json(raw)
    except (OpenRouterError, ValueError, json.JSONDecodeError):
        return None

    orig = parsed.get("original_substring", "")
    new = parsed.get("new_substring", "")
    reason = parsed.get("reason", "")
    if not _validate_swap(triple.assistant, orig, new):
        return None
    idx = triple.assistant.find(orig)
    new_answer = triple.assistant[:idx] + new + triple.assistant[idx + len(orig):]
    span = Span(start=idx, end=idx + len(new), text=new,
                original_text=orig, field="", strategy="llm")
    return Variant(output=new_answer, span=span)


def variant_to_record(triple: Triple, variant: Variant, variant_idx: int) -> dict[str, Any]:
    return {
        "id": f"{triple.id}_v{variant_idx}",
        "source_id": triple.id,
        "query": triple.user,
        "context": triple.tool_output_raw,
        "output": variant.output,
        "original_output": triple.assistant,
        "hallucination_labels": [
            {
                "start": variant.span.start,
                "end": variant.span.end,
                "text": variant.span.text,
                "original_text": variant.span.original_text,
                "field": variant.span.field,
                "type": "Type1_Contradiction",
                "strategy": variant.span.strategy,
            }
        ],
    }
