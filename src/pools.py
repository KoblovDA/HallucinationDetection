"""Build in-sample and cross-sample value pools per (tool_name, field_name)."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .data import Triple, walk_leaves


DIRTY_FIELDS: frozenset[str] = frozenset({
    "name", "title", "id", "description", "text", "symbol",
    "code", "condition", "type", "status", "category",
})


def is_dirty_field(field_name: str) -> bool:
    return field_name.lower() in DIRTY_FIELDS


def build_cross_sample_pool(triples: list[Triple]) -> dict[tuple[str, str], set[str]]:
    """For each (tool_name, last_key), aggregate all string leaf values seen across the dataset."""
    pool: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in triples:
        for entry in t.tool_output:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            tool_name = entry["name"]
            results = entry.get("results", entry)
            for path, value in walk_leaves(results):
                if not path:
                    continue
                field = path[-1]
                if isinstance(value, str) and 0 < len(value) < 80:
                    pool[(tool_name, field)].add(value)
    return pool


def in_sample_pool(tool_output: list[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    """Pool restricted to the current sample's tool_output (catches list-internal values)."""
    pool: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in tool_output:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        tool_name = entry["name"]
        results = entry.get("results", entry)
        for path, value in walk_leaves(results):
            if not path:
                continue
            field = path[-1]
            if isinstance(value, str) and 0 < len(value) < 80:
                pool[(tool_name, field)].add(value)
    return pool
