"""ISO date detection plus natural-language reformat patterns observed in ToolACE answers.

Supports finding an ISO date in an answer in several common rendered forms,
and producing a swapped date in the same surface format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta


ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?")

MONTH_LONG = ["", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
MONTH_SHORT = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_iso(value: str) -> date | None:
    m = ISO_RE.match(value)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _ordinal_suffix(d: int) -> str:
    if 11 <= d % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")


@dataclass(frozen=True)
class DateFormat:
    """A rendered form of a date and the inverse — how to render any other date in the same form."""
    name: str
    render: callable  # date -> str

    def __call__(self, d: date) -> str:
        return self.render(d)


def _render_iso(d: date) -> str:
    return d.isoformat()


def _render_month_d_yyyy(d: date) -> str:
    return f"{MONTH_LONG[d.month]} {d.day}, {d.year}"


def _render_month_dth_yyyy(d: date) -> str:
    return f"{MONTH_LONG[d.month]} {d.day}{_ordinal_suffix(d.day)}, {d.year}"


def _render_short_month_d_yyyy(d: date) -> str:
    return f"{MONTH_SHORT[d.month]} {d.day}, {d.year}"


def _render_month_d_no_year(d: date) -> str:
    return f"{MONTH_LONG[d.month]} {d.day}"


def _render_month_dth_no_year(d: date) -> str:
    return f"{MONTH_LONG[d.month]} {d.day}{_ordinal_suffix(d.day)}"


def _render_month_yyyy(d: date) -> str:
    return f"{MONTH_LONG[d.month]} {d.year}"


def _render_dth_month_yyyy(d: date) -> str:
    return f"{d.day}{_ordinal_suffix(d.day)} {MONTH_LONG[d.month]} {d.year}"


# Order matters: longest/most specific first so we prefer e.g. "January 15, 2024" over "January 15".
FORMATS: list[DateFormat] = [
    DateFormat("ISO", _render_iso),
    DateFormat("Month Dth, YYYY", _render_month_dth_yyyy),
    DateFormat("Month D, YYYY", _render_month_d_yyyy),
    DateFormat("ShortMonth D, YYYY", _render_short_month_d_yyyy),
    DateFormat("Dth Month YYYY", _render_dth_month_yyyy),
    DateFormat("Month Dth", _render_month_dth_no_year),
    DateFormat("Month D", _render_month_d_no_year),
    DateFormat("Month YYYY", _render_month_yyyy),
]


def find_in_text(d: date, text: str) -> tuple[DateFormat, int, str] | None:
    """Return (format, char_offset, rendered_string) of the first matching rendering of d in text.

    Searches formats in priority order; returns the first occurrence found.
    """
    for fmt in FORMATS:
        rendered = fmt(d)
        idx = text.find(rendered)
        if idx >= 0:
            return fmt, idx, rendered
    return None
