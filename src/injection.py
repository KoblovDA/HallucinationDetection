"""Type 1 injection strategies for ToolACE.

Each strategy attempts to find one swap candidate in the assistant answer that can be replaced
with a contradicting value, and returns (mutated_text, span_dict) on success.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .data import Triple, walk_leaves
from .dates import find_in_text as find_date_in_text, parse_iso
from .pools import in_sample_pool, is_dirty_field


@dataclass
class Span:
    start: int
    end: int
    text: str             # the new (hallucinated) substring inserted into the answer
    original_text: str    # what was there before
    field: str            # last JSON key
    strategy: str         # which strategy produced it


def _swap_substring(text: str, old: str, new: str, occurrence: int = 0) -> tuple[str, int] | None:
    """Replace the (occurrence-th) occurrence of `old` in `text` with `new`. Returns (new_text, start) or None."""
    start = -1
    pos = 0
    for _ in range(occurrence + 1):
        start = text.find(old, pos)
        if start < 0:
            return None
        pos = start + 1
    return text[:start] + new + text[start + len(old):], start


def _word_bounded_find(text: str, token: str) -> int:
    """Return the index of the first occurrence of `token` in `text` such that the surrounding
    characters are not digits or '.' (so '12' won't match inside '120' or '12.5')."""
    for m in re.finditer(re.escape(token), text):
        i, j = m.start(), m.end()
        left_ok = i == 0 or (not text[i - 1].isdigit() and text[i - 1] != ".")
        right_ok = j == len(text) or (not text[j].isdigit() and text[j] != ".")
        if left_ok and right_ok:
            return i
    return -1


def _try_int_swap(value: int, answer: str, field: str, rng: random.Random) -> tuple[str, Span] | None:
    candidates = [str(value), f"{value:,}"] if abs(value) >= 1000 else [str(value)]
    chosen = None
    for c in candidates:
        if _word_bounded_find(answer, c) >= 0:
            chosen = c
            break
    if chosen is None:
        return None

    for _ in range(30):
        if abs(value) < 10:
            new = value + rng.choice([-5, -3, -2, 2, 3, 5, 7])
        elif abs(value) < 100:
            new = value + rng.choice([-50, -30, 20, 30, 50, 70])
        else:
            factor = rng.uniform(2.5, 5.0) if rng.random() < 0.5 else rng.uniform(0.1, 0.4)
            new = int(value * factor)
        if new != value:
            break
    else:
        return None
    new_s = f"{new:,}" if "," in chosen else str(new)
    if new_s == chosen:
        return None
    swapped = _swap_substring(answer, chosen, new_s)
    if not swapped:
        return None
    new_text, start = swapped
    return new_text, Span(start=start, end=start + len(new_s), text=new_s,
                          original_text=chosen, field=field, strategy="type_based_int")


def _try_float_swap(value: float, answer: str, field: str, rng: random.Random) -> tuple[str, Span] | None:
    # Try representations in priority order (most specific first, to avoid e.g. "0.00" matching inside "0.002").
    raw = repr(value).rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    candidates = [str(value), raw, f"{value:.4f}", f"{value:.3f}", f"{value:.2f}", f"{value:.1f}"]
    seen = set()
    ordered: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    chosen = None
    for c in ordered:
        # require c to be a "word"-bounded match: not just a substring of a longer numeric token
        for m in re.finditer(re.escape(c), answer):
            i, j = m.start(), m.end()
            if (i == 0 or not answer[i - 1].isdigit() and answer[i - 1] != ".") and \
               (j == len(answer) or not answer[j].isdigit() and answer[j] != "."):
                chosen = c
                break
        if chosen is not None:
            break
    if chosen is None:
        return None

    decimals = len(chosen.split(".")[1]) if "." in chosen else 0
    for _ in range(30):
        factor = rng.uniform(2.5, 5.0) if rng.random() < 0.5 else rng.uniform(0.1, 0.4)
        new = round(value * factor, decimals) if decimals else int(round(value * factor))
        new_s = f"{new:.{decimals}f}" if decimals else str(new)
        if new_s != chosen:
            break
    else:
        return None

    swapped = _swap_substring(answer, chosen, new_s)
    if not swapped:
        return None
    new_text, start = swapped
    return new_text, Span(start=start, end=start + len(new_s), text=new_s,
                          original_text=chosen, field=field, strategy="type_based_float")


def _try_date_swap(value: str, answer: str, field: str, rng: random.Random) -> tuple[str, Span] | None:
    d = parse_iso(value)
    if d is None:
        return None
    hit = find_date_in_text(d, answer)
    if hit is None:
        return None
    fmt, idx, rendered = hit
    # shift by 30..365 days, random sign
    shift = rng.randint(30, 365) * rng.choice([-1, 1])
    try:
        new_d = d + timedelta(days=shift)
    except OverflowError:
        return None
    new_rendered = fmt(new_d)
    if new_rendered == rendered:
        return None
    new_text = answer[:idx] + new_rendered + answer[idx + len(rendered):]
    return new_text, Span(start=idx, end=idx + len(new_rendered), text=new_rendered,
                          original_text=rendered, field=field, strategy=f"type_based_date[{fmt.name}]")


def _try_url_swap(value: str, answer: str, field: str, rng: random.Random) -> tuple[str, Span] | None:
    if value not in answer:
        return None
    # Modify final path segment or v= param
    new = value
    m = re.search(r"v=([\w-]+)", value)
    if m:
        new_id = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(len(m.group(1))))
        new = value[:m.start(1)] + new_id + value[m.end(1):]
    elif "/" in value:
        idx = value.rfind("/")
        tail = value[idx + 1:] or "page"
        new_tail = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(max(len(tail), 6)))
        new = value[:idx + 1] + new_tail
    if new == value:
        return None
    swapped = _swap_substring(answer, value, new)
    if not swapped:
        return None
    new_text, start = swapped
    return new_text, Span(start=start, end=start + len(new), text=new,
                          original_text=value, field=field, strategy="type_based_url")


def _try_pool_swap(value: str, answer: str, field: str, pool: set[str], rng: random.Random,
                   strategy: str = "in_sample_pool") -> tuple[str, Span] | None:
    if is_dirty_field(field):
        return None
    idx = answer.find(value)
    if idx < 0:
        return None
    alternatives = [x for x in pool if x != value and x]
    rng.shuffle(alternatives)
    # Prefer alternatives that don't already appear immediately adjacent (avoid "React, React"-style duplicates).
    for new in alternatives:
        new_text = answer[:idx] + new + answer[idx + len(value):]
        window = new_text[max(0, idx - len(new) - 4): idx + 2 * len(new) + 4]
        if window.count(new) <= 1:
            return new_text, Span(start=idx, end=idx + len(new), text=new,
                                  original_text=value, field=field, strategy=strategy)
    if alternatives:
        new = alternatives[0]
        new_text = answer[:idx] + new + answer[idx + len(value):]
        return new_text, Span(start=idx, end=idx + len(new), text=new,
                              original_text=value, field=field, strategy=strategy)
    return None


def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


# Bool rewrite rules: list of (compiled_regex_for_finding, replacement_string).
# Replacement substitutes the entire match; the span returned points at the new text.
_BOOL_RULES_TRUE: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "is_correct": [(re.compile(r"\(Correct\)"), "(Incorrect)")],
    "iscorrect":  [(re.compile(r"\(Correct\)"), "(Incorrect)")],
    "valid":      [(re.compile(r"\bis valid\b"), "is invalid"),
                   (re.compile(r"\bvalidated\b"), "invalidated")],
    "isvalid":    [(re.compile(r"\bis valid\b"), "is invalid")],
    "is_valid":   [(re.compile(r"\bis valid\b"), "is invalid")],
    "exists":     [(re.compile(r"\bdoes indeed exist\b"), "does not exist"),
                   (re.compile(r"\bdoes exist\b"), "does not exist")],
    "success":    [(re.compile(r"\bhas been successfully\b"), "has failed to be"),
                   (re.compile(r"\bsuccessfully\b"), "unsuccessfully")],
    "stock":          [(re.compile(r"\*\*In Stock:?\*\*:?\s*Yes\b"), "**In Stock:** No")],
    "free_shipping":  [(re.compile(r"\*\*Free Shipping\*\*:?\s*Yes\b"), "**Free Shipping**: No")],
    "prime_eligible": [(re.compile(r"\*\*Prime Eligible\*\*:?\s*Yes\b"), "**Prime Eligible**: No")],
}
_BOOL_RULES_FALSE: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "valid":     [(re.compile(r"\bis not valid\b"), "is valid"),
                  (re.compile(r"\bis invalid\b"), "is valid"),
                  (re.compile(r"\binvalid\b"), "valid")],
    "isvalid":   [(re.compile(r"\bis not valid\b"), "is valid"),
                  (re.compile(r"\bis invalid\b"), "is valid")],
    "is_valid":  [(re.compile(r"\bis not valid\b"), "is valid"),
                  (re.compile(r"\bis invalid\b"), "is valid")],
    "stock":          [(re.compile(r"\*\*In Stock:?\*\*:?\s*No\b"), "**In Stock:** Yes")],
    "free_shipping":  [(re.compile(r"\*\*Free Shipping\*\*:?\s*No\b"), "**Free Shipping**: Yes")],
    "prime_eligible": [(re.compile(r"\*\*Prime Eligible\*\*:?\s*No\b"), "**Prime Eligible**: Yes")],
}


def _try_bool_swap(value: bool, answer: str, field: str) -> tuple[str, Span] | None:
    rules_map = _BOOL_RULES_TRUE if value else _BOOL_RULES_FALSE
    rules = rules_map.get(field.lower())
    if not rules:
        return None
    for pat, new_str in rules:
        m = pat.search(answer)
        if m is None:
            continue
        start, end = m.span()
        original = answer[start:end]
        if original == new_str:
            continue
        new_text = answer[:start] + new_str + answer[end:]
        return new_text, Span(start=start, end=start + len(new_str), text=new_str,
                              original_text=original, field=field, strategy="type_based_bool")
    return None


def collect_all_swaps(triple: Triple, rng: random.Random,
                      cross_pool: dict[tuple[str, str], set[str]] | None = None
                      ) -> list[tuple[str, Span]]:
    """Try every applicable strategy on every leaf value in tool_output.

    Returns a list of (corrupted_answer, span) — one entry per distinct valid swap candidate.
    De-duplicates by (start, end, text). Caller is responsible for picking among them.
    """
    answer = triple.assistant
    sample_pool = in_sample_pool(triple.tool_output)
    cross_pool = cross_pool or {}
    candidates: list[tuple[str, Span]] = []
    seen: set[tuple[int, int, str]] = set()

    for entry in triple.tool_output:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        tool_name = entry["name"]
        results_node = entry.get("results", entry)
        for path, value in walk_leaves(results_node):
            if not path:
                continue
            field = path[-1]
            attempts: list[tuple[str, Span]] = []

            if isinstance(value, bool):
                r = _try_bool_swap(value, answer, field)
                if r is not None: attempts.append(r)
            elif isinstance(value, int):
                r = _try_int_swap(value, answer, field, rng)
                if r is not None: attempts.append(r)
            elif isinstance(value, float):
                r = _try_float_swap(value, answer, field, rng)
                if r is not None: attempts.append(r)
            elif isinstance(value, str):
                if _is_url(value):
                    r = _try_url_swap(value, answer, field, rng)
                    if r is not None: attempts.append(r)
                elif parse_iso(value) is not None:
                    r = _try_date_swap(value, answer, field, rng)
                    if r is not None: attempts.append(r)
                else:
                    r = _try_pool_swap(value, answer, field,
                                       sample_pool.get((tool_name, field), set()),
                                       rng, strategy="in_sample_pool")
                    if r is not None: attempts.append(r)
                    r = _try_pool_swap(value, answer, field,
                                       cross_pool.get((tool_name, field), set()),
                                       rng, strategy="cross_sample_pool")
                    if r is not None: attempts.append(r)

            for new_text, span in attempts:
                key = (span.start, span.end, span.text)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((new_text, span))
    return candidates


def inject(triple: Triple, rng: random.Random,
           cross_pool: dict[tuple[str, str], set[str]] | None = None) -> tuple[str, list[Span]] | None:
    """Try strategies in priority order, return (corrupted_answer, [span]) or None if nothing applies.

    `cross_pool` is a global `(tool_name, field) → {values}` pool built across the whole dataset;
    used as a last-resort fallback for categorical strings when in-sample pool is exhausted.
    """
    answer = triple.assistant
    sample_pool = in_sample_pool(triple.tool_output)
    cross_pool = cross_pool or {}

    # Gather candidate leaf values (tool_name, last_field, value)
    candidates: list[tuple[str, str, Any]] = []
    for entry in triple.tool_output:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        tool_name = entry["name"]
        results = entry.get("results", entry)
        for path, value in walk_leaves(results):
            if not path:
                continue
            field = path[-1]
            candidates.append((tool_name, field, value))

    rng.shuffle(candidates)

    for tool_name, field, value in candidates:
        # type-based, in priority of robustness
        if isinstance(value, bool):
            r = _try_bool_swap(value, answer, field)
            if r is not None:
                return r[0], [r[1]]
            continue
        if isinstance(value, int):
            r = _try_int_swap(value, answer, field, rng)
            if r is not None:
                return r[0], [r[1]]
        elif isinstance(value, float):
            r = _try_float_swap(value, answer, field, rng)
            if r is not None:
                return r[0], [r[1]]
        elif isinstance(value, str):
            if _is_url(value):
                r = _try_url_swap(value, answer, field, rng)
                if r is not None:
                    return r[0], [r[1]]
                continue
            # ISO date?
            if parse_iso(value) is not None:
                r = _try_date_swap(value, answer, field, rng)
                if r is not None:
                    return r[0], [r[1]]
                continue
            # categorical string in clean field — try in-sample pool, then cross-sample pool
            r = _try_pool_swap(value, answer, field, sample_pool.get((tool_name, field), set()),
                               rng, strategy="in_sample_pool")
            if r is not None:
                return r[0], [r[1]]
            r = _try_pool_swap(value, answer, field, cross_pool.get((tool_name, field), set()),
                               rng, strategy="cross_sample_pool")
            if r is not None:
                return r[0], [r[1]]

    return None
