"""Prompt assembly. Pure functions: no model, no I/O, fully testable.

The system prompt is built from the character, the accepted profile, and the
retrieved memories, within a character budget so memory cannot crowd out the
character.
"""

from __future__ import annotations

from typing import Iterable

from .character import Character


def memories_block(snippets: Iterable[str], max_chars: int = 1200) -> list[str]:
    """Trim retrieved memory texts to a total character budget, keeping order."""
    kept: list[str] = []
    used = 0
    for text in snippets:
        text = " ".join(text.split())
        if not text:
            continue
        if used + len(text) > max_chars:
            remaining = max_chars - used
            if remaining > 40:
                kept.append(text[: remaining - 1].rstrip() + "…")
            break
        kept.append(text)
        used += len(text)
    return kept


ANSWER_RULES = (
    "How to answer questions about the user:\n"
    "- Answer with only what was asked, in as few words as the question needs.\n"
    "- Use only the memory that directly answers the question. Ignore memories about other "
    "people or other facts, even if they sit close together in the list.\n"
    "- Do not add details that were not asked for, such as dates or numbers from other memories.\n"
    "- If no memory clearly answers the question, say you are not sure. Do not guess."
)


def build_system_prompt(character: Character, profile: list[str], memories: list[str]) -> str:
    parts = [
        character.system_prompt,
        f"You are {character.name}. {character.identity}",
    ]
    if profile:
        parts.append(
            "What the user has accepted about themselves:\n"
            + "\n".join(f"- {p}" for p in profile)
        )
    if memories:
        parts.append(
            "Possibly relevant memories from earlier sessions. They may be incomplete; "
            "do not state them as certain if unsure:\n"
            + "\n".join(f"- {m}" for m in memories)
        )
        parts.append(ANSWER_RULES)
    return "\n\n".join(parts)
