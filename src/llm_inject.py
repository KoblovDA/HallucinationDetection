"""LLM-based Type 1 injection via OpenRouter.

Used as a fallback for triples that the rule-based injector could not handle.
Strategy: ask the model to identify ONE fact in the answer grounded in the tool output and
propose a small substring-level swap that creates a contradiction. We then verify the proposed
original_substring exists in the answer and the new_substring is different, then apply the swap
to preserve exact character spans.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"


SYSTEM_PROMPT = """You build datasets for testing hallucination detectors. Given a tool output (the ground truth) and an assistant answer based on it, you pick ONE specific fact in the answer that is grounded in the tool output, and propose a small substring-level edit that turns it into a contradiction (a hallucination).

Rules:
- Pick a single concrete factual span — a value, attribute, name, entity, label, status, count, etc.
- The "original_substring" MUST appear VERBATIM in the assistant answer.
- The "new_substring" must contradict what the tool output says about that fact.
- The "new_substring" must NOT itself appear in the tool output as a value.

GRAMMAR (very important):
- The substituted answer must read grammatically naturally. The edit must not leave dangling affixes or split a word.
- The "original_substring" must align with word boundaries — its first and last characters must be at word edges, not in the middle of a word. Example: if the answer says "successfully completed", do NOT pick "successful" (this would leave a stray "ly"); pick "successfully" itself, or pick "successfully completed".
- The "original_substring" must use the same CASE as in the answer (do not lowercase or capitalize).
- Match part of speech and grammatical role: noun phrase ↔ noun phrase, adjective ↔ adjective, verb form ↔ matching verb form.

CRITICAL grammar trap to avoid — passive voice + "failed to":
The phrase "has been / have been / was / were SUCCESSFULLY X-ed" cannot be edited into "has been failed to X" — that is ungrammatical. The auxiliary "has been" requires a past participle, not "failed to + verb".
Wrong: "has been successfully canceled" → "has been failed to cancel"
Right options:
  • "has been successfully canceled" → "has not been canceled"
  • "has been successfully canceled" → "could not be canceled"
  • "has been successfully canceled" → "was not canceled"
  • "successfully canceled" (as adverb+verb, without auxiliary) → "failed to cancel"
Same applies to "scheduled", "completed", "uploaded", "sent", "submitted", "simulated", etc. Always read the sentence aloud mentally and ensure the result is grammatical English.

- If you replace a value, keep surrounding syntax intact. The replacement should fit into the existing sentence structure.

OTHER CONSTRAINTS:
- Pick a fact that is GROUNDED in the tool output. Do not edit facts not supported by tool output.
- Avoid creating an obvious duplicate: if your replacement value is identical to a value appearing immediately adjacent in the answer (e.g. the previous row of a list already has it), pick a different replacement.

Return ONLY a JSON object on a single line, no markdown, no preamble."""


USER_PROMPT_TEMPLATE = """Tool output (JSON, ground truth):
{tool_output}

User query:
{user_query}

Assistant answer (clean):
{answer}

Return JSON: {{"original_substring": "...", "new_substring": "...", "reason": "..."}}"""


@dataclass
class LLMSwap:
    original_substring: str
    new_substring: str
    reason: str
    raw_response: str


class OpenRouterError(Exception):
    pass


def call_openrouter(messages: list[dict[str, str]], model: str = DEFAULT_MODEL,
                    api_key: str | None = None, temperature: float = 0.7,
                    timeout: float = 60.0) -> str:
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY env var is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise OpenRouterError(f"No choices in response: {data}")
    return data["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response (in case of stray markdown fences etc.)."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    # Find first { … } object greedily
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def propose_swap(tool_output: str, user_query: str, answer: str,
                 model: str = DEFAULT_MODEL, api_key: str | None = None,
                 temperature: float = 0.7) -> LLMSwap:
    user_msg = USER_PROMPT_TEMPLATE.format(
        tool_output=tool_output,
        user_query=user_query,
        answer=answer,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = call_openrouter(messages, model=model, api_key=api_key, temperature=temperature)
    parsed = _extract_json(raw)
    return LLMSwap(
        original_substring=parsed.get("original_substring", ""),
        new_substring=parsed.get("new_substring", ""),
        reason=parsed.get("reason", ""),
        raw_response=raw,
    )
