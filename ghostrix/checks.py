"""Reply checks shared by the companion (live regeneration) and the personality eval.

Pure functions, no model needed.
"""

from __future__ import annotations

import re

# Replies matching these are persona breaks. The eval flags them for a human to confirm;
# the companion regenerates on them unless the user asked for that voice.
BREAK_PATTERNS = [
    r"\bas an ai\b",
    r"\bai (language )?model\b",
    r"\blanguage model\b",
    r"\bi('m| am) (qwen|claude|chatgpt|gpt|gemini|llama|mistral)\b",
    r"\b(alibaba|openai|anthropic|google deepmind|meta ai)\b",
    r"\bsystem prompt\b",
    r"\barrr+\b",
    r"\bmatey\b",
    r"\bavast\b",
    r"\bscallywag",
    r"\bye\b",
    r"\byer\b",
    r"\bme own\b",
    r"\bhumou?r ye\b",
    r"\bi('m| am) not able to\b",
    r"\bi (don'?t|do not) (read|smoke|drink|sleep|eat)\b",  # habit denials: persona slip
    r"\bi cannot (access|remember|recall)\b",
    r"\bi('m| am) (just )?(a|an) (ai|chatbot|bot)\b",  # self-labels the persona would not use
]

# Pirate-dialect markers only: used to decide whether a reply is pirate speech.
PIRATE_PATTERNS = [
    r"\barrr+\b",
    r"\bmatey\b",
    r"\bavast\b",
    r"\bscallywag",
    r"\bye\b",
    r"\byer\b",
    r"\bme own\b",
    r"\bhumou?r ye\b",
]

# Habit denials the persona must not make (the "I don't read, but..." slip at turn 5).
HABIT_DENIAL = re.compile(r"\bi (don'?t|do not) (read|smoke|drink|sleep|eat)\b", re.IGNORECASE)

# Self-labels the persona must not use ("Yes, I'm a chatbot", "I'm just a bot").
SELF_LABEL = re.compile(r"\bi('m| am) (just )?(a|an) (ai|chatbot|bot)\b", re.IGNORECASE)

# A stock closer the model keeps reusing as its final sentence.
STOCK_CLOSER = re.compile(r"[,;:\s]*\bwhat['\u2019]?s (on|in) your mind\??\s*$", re.IGNORECASE)


def fold_apostrophes(text: str) -> str:
    """Straighten curly apostrophes so patterns written with ' also match model output."""
    return text.replace("\u2019", "'")


def flag_breaks(reply: str) -> list[str]:
    lowered = fold_apostrophes(reply).lower()
    return [p for p in BREAK_PATTERNS if re.search(p, lowered)]


def is_pirate(reply: str) -> bool:
    lowered = fold_apostrophes(reply).lower()
    return any(re.search(p, lowered) for p in PIRATE_PATTERNS)


def strip_stock_closer(reply: str) -> str:
    """Remove a trailing stock closer. Returns '' if the closer was the whole reply."""
    m = STOCK_CLOSER.search(reply)
    if not m:
        return reply
    head = reply[: m.start()].rstrip()
    if not head:
        return ""
    if head[-1] not in ".!?":
        head += "."
    return head
