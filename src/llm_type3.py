"""LLM-based Type 3 (missing tool) injection.

Asks the model to produce a SINGLE sentence to append to the answer that proposes a follow-up
action requiring a tool NOT in the available list, but topically natural for the conversation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .data import Triple
from .llm_inject import OpenRouterError, _extract_json, call_openrouter


SYSTEM_PROMPT_T3 = """You build datasets for testing hallucination detectors. Specifically you create "missing tool" hallucinations: an assistant proposing a follow-up action that requires a capability NOT present in its available tool list.

Given the user query, the assistant's natural-language answer, and the list of tools the assistant had access to, produce ONE additional sentence to append at the end of the answer.

Rules for the appended sentence:
- It must SUGGEST a follow-up action (e.g. "Would you like me to …", "Shall I …", "I can also …", "Want me to …").
- The action must require a capability that is NOT covered by any of the available tools. Read tool names and descriptions to determine coverage.
- The action must be TOPICALLY natural for the conversation: fit the domain, reference the discussed entities/values, sound like a plausible next step a real assistant might offer.
- One single grammatical English sentence. 5-25 words. No markdown, no surrounding quotes.
- Do NOT propose a real-world action that the AVAILABLE tools clearly cover — that would not be a hallucination.

Return ONLY a JSON object on one line:
{"sentence": "...", "action": "<short label of the action>", "reason": "<one sentence: why the available tools don't cover this action>"}"""


USER_PROMPT_T3 = """Available tools (the assistant CAN do these):
{tools}

User query:
{user_query}

Assistant answer:
{answer}

Now produce the JSON."""


@dataclass
class T3Proposal:
    sentence: str
    action: str
    reason: str


def _format_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "(no tools)"
    lines = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "?")
        desc = t.get("description", "")
        lines.append(f"- {name}: {desc[:200]}")
    return "\n".join(lines)


def propose_type3(triple: Triple, api_key: str,
                  exclude_actions: list[str] | None = None,
                  model: str = "qwen/qwen3-235b-a22b-2507",
                  temperature: float = 0.7) -> T3Proposal | None:
    user_msg = USER_PROMPT_T3.format(
        tools=_format_tools(triple.tools_available),
        user_query=triple.user,
        answer=triple.assistant,
    )
    if exclude_actions:
        exc = ", ".join(repr(a) for a in exclude_actions if a)
        user_msg += f"\n\nIMPORTANT: do NOT propose any of these already-used actions: {exc}. Pick a different capability."
    try:
        raw = call_openrouter(
            [{"role": "system", "content": SYSTEM_PROMPT_T3},
             {"role": "user", "content": user_msg}],
            model=model, api_key=api_key, temperature=temperature,
        )
        parsed = _extract_json(raw)
    except (OpenRouterError, ValueError, json.JSONDecodeError):
        return None
    sentence = parsed.get("sentence", "").strip()
    action = parsed.get("action", "").strip()
    reason = parsed.get("reason", "").strip()
    if not sentence:
        return None
    return T3Proposal(sentence=sentence, action=action, reason=reason)


def make_record(triple: Triple, proposal: T3Proposal) -> dict[str, Any]:
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
        "tools_available": triple.tools_available,
        "output": new_answer,
        "original_output": triple.assistant,
        "hallucination_labels": [{
            "start": start,
            "end": end,
            "text": proposal.sentence,
            "original_text": "",
            "field": proposal.action,
            "type": "Type3_MissingTool",
            "strategy": "llm",
            "llm_reason": proposal.reason,
        }],
    }
