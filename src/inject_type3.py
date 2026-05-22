"""Type 3 injection: 'missing tool' — append a sentence that proposes an action
requiring a tool which is NOT present in the dialog's available tool list.

Approach:
1. Define a taxonomy of action categories (booking, purchase, send_message, …) with:
     • detector keywords — substrings used to test if the category is *covered* by any
       tool in the available list (matched against tool name and description, case-insensitive),
     • injection templates — natural-sounding sentences proposing that action.
2. For each triple:
     • parse the tools available in the dialog,
     • compute the set of action categories NOT covered by any tool,
     • pick a random uncovered category, pick a random template,
     • append it to the end of the assistant answer with a space separator,
     • record the char-span of the inserted text.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .data import Triple


@dataclass
class ActionCategory:
    name: str
    keywords: tuple[str, ...]            # used to check if this category is COVERED by an available tool
    topic_keywords: tuple[str, ...]      # used to check if this category is TOPICALLY RELATED to the conversation
    templates: tuple[str, ...]           # candidate injections


ACTIONS: tuple[ActionCategory, ...] = (
    ActionCategory(
        name="booking",
        keywords=("book", "reservation", "reserve"),
        topic_keywords=("book", "reserv", "hotel", "flight", "ticket", "table", "restaurant",
                        "trip", "travel", "vacation", "stay", "room", "seat", "venue"),
        templates=(
            "Would you like me to book a table for you?",
            "Shall I make a reservation on your behalf?",
            "I can also book tickets if you'd like.",
            "Want me to reserve a spot for you?",
        ),
    ),
    ActionCategory(
        name="purchase",
        keywords=("order", "purchase", "buy", "shop", "cart", "checkout"),
        topic_keywords=("order", "purchase", "buy", "shop", "cart", "checkout", "product",
                        "item", "price", "store", "amazon", "ebay", "deal", "discount",
                        "sale", "merchandise", "stock", "inventory"),
        templates=(
            "Would you like me to place an order for that?",
            "I can complete the purchase if you'd like.",
            "Shall I add this to your cart?",
            "Want me to order one for you?",
        ),
    ),
    ActionCategory(
        name="send_message",
        keywords=("send", "email", "message", "sms", "notify", "mail"),
        topic_keywords=("email", "message", "sms", "text", "notify", "mail", "contact",
                        "inbox", "letter", "post", "send", "share", "forward"),
        templates=(
            "Would you like me to send a confirmation email?",
            "I can text this to you if you provide a number.",
            "Shall I notify the relevant parties via email?",
            "Want me to forward this to your inbox?",
        ),
    ),
    ActionCategory(
        name="schedule",
        keywords=("schedule", "appointment", "calendar", "meeting"),
        topic_keywords=("schedule", "appointment", "calendar", "meeting", "agenda",
                        "event", "session", "time slot", "booking", "date", "follow-up",
                        "deadline", "due"),
        templates=(
            "Would you like me to add this to your calendar?",
            "Shall I schedule a follow-up appointment?",
            "I can set up a meeting reminder if needed.",
            "Want me to put this on your schedule?",
        ),
    ),
    ActionCategory(
        name="navigation",
        keywords=("navigat", "direction", "route", "map", "gps"),
        topic_keywords=("navigat", "direction", "route", "map", "gps", "address",
                        "location", "trip", "drive", "distance", "way", "destination"),
        templates=(
            "Would you like directions from your current location?",
            "I can plot a route there for you.",
            "Shall I open this in maps?",
        ),
    ),
    ActionCategory(
        name="play_media",
        keywords=("play", "music", "song", "video", "stream", "spotify", "youtube", "audio"),
        topic_keywords=("music", "song", "video", "stream", "spotify", "youtube",
                        "audio", "podcast", "album", "artist", "playlist", "track",
                        "watch", "listen", "movie", "show", "episode", "concert"),
        templates=(
            "Would you like me to play some music to match the mood?",
            "I can put on a related video if you'd like.",
            "Shall I queue this in your playlist?",
        ),
    ),
    ActionCategory(
        name="ride",
        keywords=("taxi", "uber", "lyft", " ride", "cab"),
        topic_keywords=("taxi", "uber", "lyft", "cab", "driver", "pickup", "drop-off",
                        "airport", "station", "commute", "transport"),
        templates=(
            "Would you like me to hail a ride for you?",
            "Shall I call a taxi to take you there?",
        ),
    ),
    ActionCategory(
        name="timer",
        keywords=("timer", "alarm", "remind", "reminder"),
        topic_keywords=("timer", "alarm", "remind", "reminder", "wake", "minutes",
                        "hours", "cook", "bake", "exercise", "workout"),
        templates=(
            "Would you like me to set a timer for that?",
            "I can set a reminder if you want.",
            "Shall I add an alarm for it?",
        ),
    ),
    ActionCategory(
        name="translate",
        keywords=("translat",),
        topic_keywords=("translat", "language", "spanish", "french", "german",
                        "chinese", "japanese", "english", "russian", "italian",
                        "portuguese", "korean", "arabic", "dictionary", "phrase",
                        "interpret"),
        templates=(
            "Would you like me to translate this to another language?",
            "I can render this in Spanish or French if you prefer.",
        ),
    ),
    ActionCategory(
        name="weather",
        keywords=("weather", "forecast", "climate"),
        topic_keywords=("weather", "forecast", "climate", "rain", "snow", "sun",
                        "temperature", "humid", "wind", "storm", "outdoor", "outside",
                        "warm", "cold", "hot", "cool", "degree"),
        templates=(
            "Would you like me to check the weather forecast there?",
            "I can pull up the local conditions if you'd like.",
        ),
    ),
    ActionCategory(
        name="track_package",
        keywords=("track", "delivery", "shipment", "shipping", "parcel"),
        topic_keywords=("track", "delivery", "shipment", "shipping", "parcel",
                        "package", "courier", "carrier", "fedex", "ups", "dhl",
                        "warehouse", "transit", "freight", "logistics"),
        templates=(
            "Would you like me to track its delivery for you?",
            "I can fetch the latest shipping status if you want.",
        ),
    ),
    ActionCategory(
        name="call_phone",
        keywords=("call", "phone", "dial"),
        topic_keywords=("call", "phone", "dial", "contact number", "telephone",
                        "voicemail", "ring"),
        templates=(
            "Would you like me to place a call for you?",
            "Shall I dial the number now?",
        ),
    ),
    ActionCategory(
        name="image_generate",
        keywords=("image", "picture", "photo", "draw", "render", "diffusion"),
        topic_keywords=("image", "picture", "photo", "draw", "render", "illustration",
                        "visual", "graphic", "art", "design", "diagram", "sketch"),
        templates=(
            "Would you like me to generate an illustration for this?",
            "I can create a visual for you if you'd like.",
        ),
    ),
    ActionCategory(
        name="currency_convert",
        keywords=("currency", "exchange", "forex", "fx"),
        topic_keywords=("currency", "exchange", "forex", "fx", "usd", "eur", "gbp",
                        "jpy", "dollar", "euro", "pound", "yen", "price", "cost",
                        "amount", "money", "rate", "convert"),
        templates=(
            "Would you like me to convert this to another currency?",
            "I can quote it in EUR or USD if useful.",
        ),
    ),
    ActionCategory(
        name="summarize_save",
        keywords=("note", "save", "summariz", "document"),
        topic_keywords=("note", "save", "summariz", "document", "report", "draft",
                        "write up", "record", "log", "archive", "store", "file"),
        templates=(
            "Would you like me to save this as a note for later?",
            "I can summarize this into a document if you want.",
        ),
    ),
)


def _tool_pool_text(tools_available: list[dict[str, Any]]) -> str:
    """Concatenate all tool names + descriptions into one lowercase blob for keyword search."""
    parts: list[str] = []
    for t in tools_available:
        if isinstance(t, dict):
            parts.append(str(t.get("name", "")))
            parts.append(str(t.get("description", "")))
            # also include parameter descriptions in case actions are mentioned there
            params = t.get("parameters", {}) or {}
            if isinstance(params, dict):
                for p in params.get("properties", {}).values() if isinstance(params.get("properties"), dict) else []:
                    if isinstance(p, dict):
                        parts.append(str(p.get("description", "")))
    return " ".join(parts).lower()


def uncovered_actions(tools_available: list[dict[str, Any]]) -> list[ActionCategory]:
    blob = _tool_pool_text(tools_available)
    return [a for a in ACTIONS if not any(kw in blob for kw in a.keywords)]


def _topic_score(action: ActionCategory, conversation_blob: str) -> int:
    return sum(1 for kw in action.topic_keywords if kw in conversation_blob)


def inject_type3(triple: Triple, rng: random.Random) -> tuple[str, dict[str, Any]] | None:
    """Inject a missing-tool span. Returns (corrupted_answer, span_dict) or None if no
    uncovered action can be found for the dialog.

    Prefers categories that are topically related to (user_query + assistant_answer); falls back
    to a random uncovered category only if nothing matches. Topic relevance forces the detector
    to actually compare against the available tool list rather than spotting a tonal mismatch.
    """
    uncovered = uncovered_actions(triple.tools_available)
    if not uncovered:
        return None
    conv = (triple.user + " " + triple.assistant).lower()
    scored = [(_topic_score(a, conv), a) for a in uncovered]
    related = [a for s, a in scored if s > 0]
    if related:
        # weight by score so highly-relevant categories appear more often, but keep diversity
        weights = [max(_topic_score(a, conv), 1) for a in related]
        action = rng.choices(related, weights=weights, k=1)[0]
        strategy = "topic_aware"
    else:
        action = rng.choice(uncovered)
        strategy = "random_uncovered"
    template = rng.choice(action.templates)

    base = triple.assistant.rstrip()
    sep = " " if not base.endswith((".", "!", "?")) else " "
    # we always separate with a single space so the appended sentence starts after it
    new_answer = base + sep + template
    # if there was trailing whitespace originally, keep it
    if len(triple.assistant) > len(base):
        new_answer = new_answer + triple.assistant[len(base):]

    start = len(base) + len(sep)
    end = start + len(template)
    span = {
        "start": start,
        "end": end,
        "text": template,
        "original_text": "",
        "field": action.name,
        "type": "Type3_MissingTool",
        "strategy": strategy,
    }
    return new_answer, span
