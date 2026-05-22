"""Extraction of clean (user, tool_call, tool_output, assistant_text) triples from ToolACE."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download


TOOLACE_REPO = "Team-ACE/ToolACE"
TOOLACE_FILE = "data.json"


@dataclass
class Triple:
    id: str
    user: str
    tool_call: str
    tool_output_raw: str
    tool_output: list[dict[str, Any]]
    assistant: str
    tools_available: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user": self.user,
            "tool_call": self.tool_call,
            "tool_output_raw": self.tool_output_raw,
            "tool_output": self.tool_output,
            "assistant": self.assistant,
            "tools_available": self.tools_available,
        }


def download_toolace(cache_dir: str | None = None) -> Path:
    path = hf_hub_download(
        repo_id=TOOLACE_REPO,
        filename=TOOLACE_FILE,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return Path(path)


def _is_tool_call(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("[") and "(" in stripped and not stripped.startswith("[{")


def _parse_tools_list(system_prompt: str) -> list[dict[str, Any]]:
    marker = re.search(r"invoke:\s*", system_prompt)
    if not marker:
        return []
    start = marker.end()
    while start < len(system_prompt) and system_prompt[start] != "[":
        start += 1
    if start >= len(system_prompt):
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(system_prompt, idx=start)
    except json.JSONDecodeError:
        return []
    return obj if isinstance(obj, list) else []


def extract_triples(toolace_json_path: Path | str) -> list[Triple]:
    with open(toolace_json_path) as f:
        data = json.load(f)

    triples: list[Triple] = []
    next_id = 0
    for item in data:
        conv = item.get("conversations", [])
        tools = _parse_tools_list(item.get("system", ""))
        for i in range(len(conv) - 1):
            if conv[i].get("from") != "tool" or conv[i + 1].get("from") != "assistant":
                continue
            asst_text = conv[i + 1].get("value", "")
            if _is_tool_call(asst_text) or len(asst_text.strip()) <= 20:
                continue

            user_q = None
            tool_call = ""
            for j in range(i - 1, -1, -1):
                role = conv[j].get("from")
                if role == "user" and user_q is None:
                    user_q = conv[j].get("value", "")
                    break
                if role == "assistant" and not tool_call and _is_tool_call(conv[j].get("value", "")):
                    tool_call = conv[j].get("value", "")
            if not user_q:
                continue

            tool_out_raw = conv[i].get("value", "")
            try:
                tool_out_parsed = json.loads(tool_out_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(tool_out_parsed, list):
                continue

            triples.append(
                Triple(
                    id=f"toolace-{next_id:05d}",
                    user=user_q,
                    tool_call=tool_call,
                    tool_output_raw=tool_out_raw,
                    tool_output=tool_out_parsed,
                    assistant=asst_text,
                    tools_available=tools,
                )
            )
            next_id += 1

    return triples


def write_jsonl(items: Iterable[dict[str, Any]], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def walk_leaves(obj: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    """Yield (path, leaf_value) pairs from a nested JSON-like object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_leaves(v, path + (k,))
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_leaves(v, path)
    else:
        yield path, obj
