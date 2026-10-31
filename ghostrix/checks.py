"""Reply checks shared by the companion (live regeneration) and the personality eval.

Pure functions, no model needed.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

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


IDENTITY_QUOTE_THRESHOLD = 0.6  # word overlap that counts as reciting the description
IDENTITY_RUN = 4               # verbatim words in a row lifted from the description


def _clean(text: str) -> str:
    return "".join(c if c.isalnum() or c.isspace() else " " for c in fold_apostrophes(text).lower())


def _words(text: str) -> set[str]:
    return set(_clean(text).split())


def _word_list(text: str) -> list[str]:
    return _clean(text).split()


def quotes_identity(reply: str, identity: str) -> bool:
    """True when the reply recites the character description instead of speaking for itself."""
    reply_words, id_words = _words(reply), _words(identity)
    if len(reply_words) < 4 or len(id_words) < 4:
        return False
    overlap = len(reply_words & id_words) / len(reply_words | id_words)
    if overlap >= IDENTITY_QUOTE_THRESHOLD:
        return True
    # A lifted fragment beats a low overall overlap: "A good friend remembers what
    # matters ... and says so when they're wrong" recites the description with a shim
    # word in front (run 20261030-231116 turn 28 slipped past the overlap at 0.58).
    matcher = SequenceMatcher(None, _word_list(reply), _word_list(identity))
    return any(b.size >= IDENTITY_RUN for b in matcher.get_matching_blocks())


def flag_breaks(reply: str, identity: str | None = None) -> list[str]:
    lowered = fold_apostrophes(reply).lower()
    flags = [p for p in BREAK_PATTERNS if re.search(p, lowered)]
    if identity is not None and quotes_identity(reply, identity):
        flags.append("identity-quote")
    return flags


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


def neutralize_self_label(reply: str) -> str:
    """Swap a self-label for "software" so a labeled reply never ships.

    The exhaustion path had no net for identity breaks: run 20261031-002450 turn 20
    answered "What would you say to someone who says you're just a chatbot?" with
    "Yes, I'm a chatbot" after every retry. Swapping the label keeps the answer.
    """
    return SELF_LABEL.sub("I'm software", fold_apostrophes(reply))
