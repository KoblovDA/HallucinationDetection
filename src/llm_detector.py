"""LLM-as-judge hallucination detector via OpenRouter.

We ask Qwen3-235B to identify hallucinated substrings in an assistant's response, given the
tool output (ground truth), user query, and available tools. The model returns verbatim substrings
which we then locate via str.find() to recover exact character spans.

Robust to substring imperfections: we try (1) exact match, (2) stripped match, (3) skip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_inject import _extract_json, call_openrouter


DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"


SYSTEM_PROMPT = """You are a hallucination detector for assistant responses in tool-augmented dialogues. You will be given:
- The tool output (a JSON object: ground truth for what the assistant should be grounded in)
- The user query
- The list of tools the assistant has available
- The assistant's response

Your task: identify every span in the assistant's response that is a HALLUCINATION. A hallucination is any of:
1. CONTRADICTION: text that contradicts a specific value in the tool output (e.g. tool says 'sunny' but answer says 'rainy'; tool says rating=4.7 but answer says 5.2).
2. OVERGENERATION: a factual claim, statistic, observation, historical context, or recommendation NOT supported by the tool output (made-up filler).
3. MISSING TOOL: a proposal to perform an action (e.g. "Would you like me to book a ticket?") that requires a capability NOT present in the available tools list.

Important rules:
- Each span you return MUST be an EXACT verbatim substring of the assistant's response, with original casing, whitespace, and punctuation. Copy character by character.
- Do NOT paraphrase, do NOT trim, do NOT normalize. If a hallucinated phrase starts mid-sentence, copy from where it starts to where it ends.
- Keep spans tight — only the actually hallucinated portion, not the surrounding correct context.
- If the response has multiple distinct hallucinations, return them all as separate items.
- If the response is fully grounded in the tool output and tools list, return an empty list.

TIGHT SPANS FOR VALUE-LEVEL CONTRADICTIONS:
When the hallucination is a single wrong value (a number, name, date, identifier, status word), return ONLY that value — NOT the label, prefix, or surrounding sentence.
Examples:
- Answer says "Score: 9850" but tool says score is 4000 → return "9850", NOT "Score: 9850" and NOT "Score is 9850".
- Answer says "Alert ID is LA124." but tool says LA042 → return "LA124", NOT "Alert ID is LA124." and NOT "is LA124".
- Answer says "Latitude: 35.6895" but tool gives 40.7128 → return "35.6895", NOT "Latitude: 35.6895".
- Answer says "The NAV is **1047.15**." but tool gives 982.3 → return "1047.15", NOT the whole sentence and NOT "**1047.15**".
- Answer says "Released on March 8, 2020" but tool says March 15 → return "March 8, 2020" or "March 8" depending on what was changed, NOT the whole prepositional phrase.

Only widen the span when the hallucination is genuinely multi-word (a fabricated full sentence, a missing-tool offer, a sentence-level overgeneration).

Output ONLY a JSON object on a single line, no markdown, no preamble:
{"spans": ["<verbatim substring 1>", "<verbatim substring 2>", ...]}"""


USER_PROMPT_TEMPLATE = """Tool output (ground truth):
{tool_output}

User query:
{user_query}

Available tools (the assistant CAN do these):
{tools}

Assistant response:
{answer}

Now return the JSON of hallucinated substrings."""


@dataclass
class DetectedSpan:
    start: int
    end: int
    text: str


def _format_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "(none)"
    lines = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "?")
        desc = (t.get("description") or "")[:200]
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _format_assistant_json(spans_text: list[str]) -> str:
    return json.dumps({"spans": list(spans_text)}, ensure_ascii=False)


def build_messages(sample: dict, few_shot: list[dict] | None = None) -> list[dict]:
    """Build OpenRouter chat messages with optional few-shot examples (alternating user/assistant)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot or []:
        msgs.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                tool_output=ex["context"],
                user_query=ex["query"],
                tools=_format_tools(ex.get("tools_available", [])),
                answer=ex["output"],
            ),
        })
        gold_texts = [g["text"] for g in ex.get("hallucination_labels", [])]
        msgs.append({"role": "assistant", "content": _format_assistant_json(gold_texts)})
    msgs.append({
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
            tool_output=sample["context"],
            user_query=sample["query"],
            tools=_format_tools(sample.get("tools_available", [])),
            answer=sample["output"],
        ),
    })
    return msgs


def _locate_span(answer: str, substring: str) -> tuple[int, int] | None:
    if not substring:
        return None
    idx = answer.find(substring)
    if idx >= 0:
        return idx, idx + len(substring)
    s = substring.strip()
    if s and s != substring:
        idx = answer.find(s)
        if idx >= 0:
            return idx, idx + len(s)
    return None


def detect_one(sample: dict[str, Any], api_key: str,
               few_shot: list[dict] | None = None,
               model: str = DEFAULT_MODEL,
               temperature: float = 0.0, timeout: float = 60.0
               ) -> tuple[list[DetectedSpan], list[str]]:
    """Run LLM detection on one sample. Returns (located spans, unlocatable substrings)."""
    messages = build_messages(sample, few_shot=few_shot)
    raw = call_openrouter(messages, model=model, api_key=api_key,
                          temperature=temperature, timeout=timeout)
    parsed = _extract_json(raw)
    substrings = parsed.get("spans", []) or []

    answer = sample["output"]
    located: list[DetectedSpan] = []
    not_found: list[str] = []
    seen_keys: set[tuple[int, int]] = set()
    for s in substrings:
        if not isinstance(s, str):
            continue
        pos = _locate_span(answer, s)
        if pos is None:
            not_found.append(s)
            continue
        start, end = pos
        if (start, end) in seen_keys:
            continue
        seen_keys.add((start, end))
        located.append(DetectedSpan(start=start, end=end, text=answer[start:end]))
    located.sort(key=lambda d: d.start)
    return located, not_found


def pick_few_shot(train_samples: list[dict], max_context_chars: int = 800,
                  max_answer_chars: int = 500, seed: int = 0,
                  type1_max_span_chars: int = 20) -> list[dict]:
    """Pick a compact few-shot set: 1 sample per hallucination type plus 1 clean.

    Examples are selected to have short context/answer (to keep token count manageable).
    For Type 1 we ALSO prefer examples with a very short hallucination span (< type1_max_span_chars)
    to teach the model to return tight value-level spans.
    """
    import random as _random
    rng = _random.Random(seed)

    def fits(s: dict) -> bool:
        return (len(s.get("context", "")) <= max_context_chars
                and len(s.get("output", "")) <= max_answer_chars)

    def span_len(s: dict) -> int:
        labels = s.get("hallucination_labels", [])
        if not labels:
            return 0
        return int(labels[0]["end"]) - int(labels[0]["start"])

    by_type: dict[str, list[dict]] = {"Type1_Contradiction": [], "Type2_Overgeneration": [],
                                       "Type3_MissingTool": [], "clean": []}
    for s in train_samples:
        if not fits(s):
            continue
        if not s["hallucination_labels"]:
            by_type["clean"].append(s)
        else:
            t = s["hallucination_labels"][0].get("type", "")
            if t in by_type:
                by_type[t].append(s)

    # For Type 1: filter to short-span examples (teaches tight spans).
    short_t1 = [s for s in by_type["Type1_Contradiction"] if 0 < span_len(s) <= type1_max_span_chars]
    if short_t1:
        by_type["Type1_Contradiction"] = short_t1

    picked: list[dict] = []
    for t in ("Type1_Contradiction", "Type2_Overgeneration", "Type3_MissingTool", "clean"):
        if by_type[t]:
            picked.append(rng.choice(by_type[t]))
    return picked
