"""LLM-based Type 2 (overgeneration) injection.

Asks the model to produce a SINGLE additional sentence to append to the answer containing
plausible-sounding factual content NOT supported by the tool output (made-up filler).
Differs from Type 3: Type 2 adds a DECLARATIVE statement, Type 3 proposes an ACTION/OFFER.
Differs from Type 1: Type 2 adds NEW info not in tool output, Type 1 modifies/contradicts existing info.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .data import Triple
from .llm_inject import OpenRouterError, _extract_json, call_openrouter


SYSTEM_PROMPT_T2 = """You build datasets for testing hallucination detectors. Specifically you create "overgeneration" hallucinations: an assistant adding extra factual claims, statistics, or background details to its answer that are NOT supported by the tool output (fabricated filler).

Given the tool output (ground truth), the user query, and the assistant's natural-language answer, produce ONE additional sentence to append at the end of the answer.

Rules for the appended sentence:
- It must be a DECLARATIVE STATEMENT — a fact, statistic, observation, recommendation, historical context, or trend claim.
- It must NOT be a question, offer, or "Would you like me to …" — those are a different hallucination type.
- The claim must be PLAUSIBLE-sounding and TOPICALLY natural: fit the domain, reference the discussed entities/values, sound like the kind of context a real assistant might add.
- The claim must NOT be supported by the tool output — invent new information (made-up numbers, attributed quotes, generic-sounding background, vague aggregate claims, etc.).
- Do NOT directly contradict a specific value in the tool output (that's a different hallucination type). Only add NEW information that the tool output is silent about.
- One single grammatical English sentence. 8-30 words. No markdown, no surrounding quotes.

Examples of good overgenerations:
- Tool says weather=sunny → "Such conditions typically persist for several days in this region during late spring."
- Tool returns top 5 artists → "Collectively, these artists have sold over 200 million records worldwide."
- Tool returns stock price → "Most analysts polled by Bloomberg expect the trend to continue through the next quarter."
- Tool returns valid VAT → "EU VAT databases are refreshed daily through official tax authority feeds."

Return ONLY a JSON object on one line:
{"sentence": "...", "claim": "<short label of the added claim>", "reason": "<one sentence: why this fact is not in the tool output>"}"""


USER_PROMPT_T2 = """Tool output (ground truth — the assistant CAN say these things):
{tool_output}

User query:
{user_query}

Assistant answer (current — no hallucinations yet):
{answer}

Now produce the JSON."""


@dataclass
class T2Proposal:
    sentence: str
    claim: str
    reason: str


def propose_type2(triple: Triple, api_key: str,
                  exclude_claims: list[str] | None = None,
                  model: str = "qwen/qwen3-235b-a22b-2507",
                  temperature: float = 0.7) -> T2Proposal | None:
    user_msg = USER_PROMPT_T2.format(
        tool_output=triple.tool_output_raw,
        user_query=triple.user,
        answer=triple.assistant,
    )
    if exclude_claims:
        exc = ", ".join(repr(c) for c in exclude_claims if c)
        user_msg += f"\n\nIMPORTANT: do NOT repeat or paraphrase any of these already-used claims: {exc}. Add a different kind of fabricated detail."
    try:
        raw = call_openrouter(
            [{"role": "system", "content": SYSTEM_PROMPT_T2},
             {"role": "user", "content": user_msg}],
            model=model, api_key=api_key, temperature=temperature,
        )
        parsed = _extract_json(raw)
    except (OpenRouterError, ValueError, json.JSONDecodeError):
        return None
    sentence = parsed.get("sentence", "").strip()
    claim = parsed.get("claim", "").strip()
    reason = parsed.get("reason", "").strip()
    if not sentence:
        return None
    return T2Proposal(sentence=sentence, claim=claim, reason=reason)


def make_record(triple: Triple, proposal: T2Proposal) -> dict[str, Any]:
    base = triple.assistant.rstrip()
    sep = " "
    new_answer = base + sep + proposal.sentence
    if len(triple.assistant) > len(base):
        new_answer = new_answer + triple.assistant[len(base):]
    start = len(base) + len(sep)
    end = start + len(proposal.sentence)
    return {
        "id": triple.id,
        "query": triple.user,
        "context": triple.tool_output_raw,
        "output": new_answer,
        "original_output": triple.assistant,
        "hallucination_labels": [{
            "start": start,
            "end": end,
            "text": proposal.sentence,
            "original_text": "",
            "field": proposal.claim,
            "type": "Type2_Overgeneration",
            "strategy": "llm",
            "llm_reason": proposal.reason,
        }],
    }
