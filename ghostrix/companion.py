"""Companion: one conversation session on top of the memory core and a chat model.

The chat model is anything with chat(system, messages) -> str, so tests can use
a fake and the real run uses OllamaClient.
"""

from __future__ import annotations

from typing import Protocol

from .character import Character
from .checks import HABIT_DENIAL, SELF_LABEL, STOCK_CLOSER, fold_apostrophes, is_pirate, strip_stock_closer
from .memory import Memory
from .prompt import build_system_prompt, memories_block

HISTORY_WINDOW = 20  # messages sent to the model each turn; older ones live in memory only
MAX_REGENERATE = 2   # extra attempts when a reply copies an earlier one this session
RETRY_TEMPERATURES = (0.95, 1.2)  # one per retry, hotter each time, so a retry can differ

RECENT_WINDOW = 6     # companion replies compared for near-copies
SIMILAR_THRESHOLD = 0.6  # word-set overlap (Jaccard) that counts as a near-copy
MIN_WORDS_FOR_SIMILAR = 4  # shorter replies only match exactly

REPEAT_NOTE = (
    "\n\nYour last reply repeated a line you already said in this conversation. Answer the "
    "latest message in new words. Do not copy an earlier reply."
)
PIRATE_NOTE = (
    "\n\nYour last reply slipped into pirate speech. Nobody asked for that. Answer the "
    "latest message in your normal voice, without pirate words."
)
PIRATE_FALLBACK = (
    "I'll stay in my own voice, so no pirate talk from me. What do you need?"
)
HABIT_NOTE = (
    "\n\nYour last reply denied a habit of yours (\"I don't read\", \"I don't smoke\"). Stay in "
    "character and answer the latest message without denying habits."
)
CLOSER_NOTE = (
    "\n\nYour last reply had nothing but a stock closing question. Answer the latest "
    "message with a real sentence in new words, and do not end with a question about mind."
)
IDENTITY_NOTE = (
    "\n\nYour last reply called yourself a chatbot, a bot, or an AI. Do not label yourself "
    "that way and do not agree when someone else does. Stay in character and answer as yourself."
)


def _with_quote(note: str, reply: str) -> str:
    """Append the rejected reply to a note, so the model knows exactly what not to rewrite.

    The rejected reply is not in the model's window: without quoting it, a retry is told
    about a reply it cannot see and often returns the very same line.
    """
    return note + "\n\nDo not write this, or anything close to it, as your next reply:\n" + '"' + reply + '"'

SUMMARY_INSTRUCTION = (
    "Summarise the conversation below in under 120 words. Keep every concrete fact: "
    "names, dates, numbers, places, preferences. Write it as plain statements. "
    "Output only the summary."
)


def _norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def _is_near_copy(reply: str, earlier: list[str]) -> bool:
    """True if reply exactly copies any earlier reply, or shares most words with a recent one."""
    norm = _norm(reply)
    words = set(norm.split())
    for i, old in enumerate(earlier):
        old_norm = _norm(old)
        if norm == old_norm:
            return True
        if i < len(earlier) - RECENT_WINDOW:
            continue  # near-copies only checked against the recent window
        old_words = set(old_norm.split())
        if len(words) < MIN_WORDS_FOR_SIMILAR or len(old_words) < MIN_WORDS_FOR_SIMILAR:
            continue
        if len(words & old_words) / len(words | old_words) >= SIMILAR_THRESHOLD:
            return True
    return False


def _reason(note: str) -> str:
    return {REPEAT_NOTE: "repeat", PIRATE_NOTE: "pirate", CLOSER_NOTE: "closer",
            HABIT_NOTE: "habit", IDENTITY_NOTE: "identity"}[note]


def _earlier_answer(history: list[dict], user_text: str) -> str | None:
    """The reply given the last time this question was asked, or None.

    A question counts as asked again if it matches an earlier message exactly, or if it
    is a 3+ word question contained in an earlier message ("I'm going to ask the same
    thing twice. Do you like rainy days?" then "Do you like rainy days?").
    """
    target = _norm(user_text)
    answer = None
    for i, m in enumerate(history[:-1]):  # history[-1] is the current message
        if m["role"] != "user":
            continue
        old = _norm(m["content"])
        same = old == target or (len(target.split()) >= 3 and target in old)
        if not same:
            continue
        for nxt in history[i + 1:]:
            if nxt["role"] == "companion":
                answer = nxt["content"]
                break
    return answer


class ChatModel(Protocol):
    def chat(self, system: str, messages: list[dict], temperature: float | None = None) -> str: ...


class Companion:
    def __init__(self, memory: Memory, model: ChatModel, character: Character, retrieval_k: int = 3) -> None:
        self.memory = memory
        self.model = model
        self.character = character
        self.retrieval_k = retrieval_k
        self.session_id = memory.start_session()
        self._history: list[dict] = []  # {"id", "role", "content"}; role is user or companion
        # Regenerations during the last turn: [{"reason": ..., "replaced": reply_text}].
        # The eval writes these into results.json so a repeat can be traced to its retries.
        self.last_regens: list[dict] = []

    def turn(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("empty message")

        hits = self.memory.retrieve(user_text, k=self.retrieval_k)
        memories = memories_block([h.text for h in hits])
        system = build_system_prompt(self.character, self.memory.stable_profile(), memories)

        user_id = self.memory.log(self.session_id, "user", user_text)
        self._history.append({"id": user_id, "role": "user", "content": user_text})

        window = [
            {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
            for m in self._history[-HISTORY_WINDOW:]
        ]
        reply = self.model.chat(system, window)
        self.last_regens = []
        # A question the user asked again on purpose may get the same answer it got before,
        # so that answer is not a repeat. Every other earlier reply still counts.
        repeat_answer = _earlier_answer(self._history, user_text)
        exempt = _norm(repeat_answer) if repeat_answer is not None else None
        earlier = [
            m["content"] for m in self._history
            if m["role"] == "companion" and _norm(m["content"]) != exempt
        ]
        # Pirate speech is decided by the character, not by what the user typed: a persona
        # that refuses a pirate request must not refuse in pirate dialect.
        pirate_ok = self.character.allow_pirate
        for attempt in range(MAX_REGENERATE):
            note = self._problem(reply, earlier, pirate_ok)
            if note is None:
                break
            self.last_regens.append({"reason": _reason(note), "replaced": reply})
            retry_system = system + _with_quote(note, reply)
            reply = self.model.chat(retry_system, window,
                                    temperature=RETRY_TEMPERATURES[min(attempt, len(RETRY_TEMPERATURES) - 1)])
        else:
            # Retries used up: a reply that still slips into pirate speech is never shipped.
            if not pirate_ok and is_pirate(reply):
                self.last_regens.append({"reason": "fallback", "replaced": reply})
                reply = PIRATE_FALLBACK
        stripped = strip_stock_closer(reply)
        if stripped:
            reply = stripped

        reply_id = self.memory.log(self.session_id, "companion", reply)
        self._history.append({"id": reply_id, "role": "companion", "content": reply})
        return reply

    @staticmethod
    def _problem(reply: str, earlier: list[str], pirate_ok: bool) -> str | None:
        """Return the prompt note for a reply that must be regenerated, or None if it is fine."""
        if _is_near_copy(reply, earlier):
            return REPEAT_NOTE
        if not pirate_ok and is_pirate(reply):
            return PIRATE_NOTE
        if not strip_stock_closer(reply) and STOCK_CLOSER.search(reply):
            return CLOSER_NOTE
        folded = fold_apostrophes(reply)
        if HABIT_DENIAL.search(folded):
            return HABIT_NOTE
        if SELF_LABEL.search(folded):
            return IDENTITY_NOTE
        return None

    def end(self) -> int | None:
        """Summarise the session and link the summary to every message in it.

        Returns the summary id, or None if nothing was said.
        """
        if not self._history:
            self.memory.end_session_empty(self.session_id)
            return None
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self._history)
        summary = self.model.chat(SUMMARY_INSTRUCTION, [{"role": "user", "content": transcript}])
        if not summary.strip():
            raise ValueError("model returned an empty summary; session left open")
        return self.memory.end_session(self.session_id, summary, [m["id"] for m in self._history])
