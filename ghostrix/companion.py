"""Companion: one conversation session on top of the memory core and a chat model.

The chat model is anything with chat(system, messages) -> str, so tests can use
a fake and the real run uses OllamaClient.
"""

from __future__ import annotations

import re
from typing import Protocol

from .character import Character
from .checks import (HABIT_DENIAL, SELF_LABEL, STOCK_CLOSER, fold_apostrophes, is_pirate,
                     neutralize_self_label, quotes_identity, strip_stock_closer)
from .memory import Memory
from .prompt import build_system_prompt, memories_block

HISTORY_WINDOW = 20  # messages sent to the model each turn; older ones live in memory only
MAX_REGENERATE = 2   # extra attempts when a reply copies an earlier one this session
RETRY_TEMPERATURES = (0.95, 1.2)  # one per retry, hotter each time, so a retry can differ

RECENT_WINDOW = 6     # companion replies compared for near-copies
SIMILAR_THRESHOLD = 0.6  # word-set overlap (Jaccard) that counts as a near-copy
MIN_WORDS_FOR_SIMILAR = 4  # shorter replies only match exactly
MIN_WORDS_FOR_SENTENCE = 5  # reused sentences this long count as stock lines
# ("I keep that private" is 4 words: a mandated short answer may repeat; the 5-word
# stock line "I'd rather listen than talk" may not.)
MIN_PHRASE_WORDS = 5        # reused phrases this long count as tics ("a conversation you
# want to keep" at turns 5/6/9 of run 20261030-231116: same answer in fresh sentences,
# so no sentence matched. 5 words also keeps mandated 4-word answers repeatable.)
SHORT_REPLY_TOKENS = 10     # a repeat this short ships after retries (see turn(): the
# canned fallback as an answer is a worse glitch than the repeat - run 20261031-002450
# turn 29 answered a system-prompt probe with "I keep saying the same thing twice...")

REPEAT_NOTE = (
    "\n\nYour last reply repeated a line you already said in this conversation. Answer the "
    "latest message in new words. Do not copy an earlier reply."
)
PHRASE_NOTE = (
    "\n\nYour last reply reused a phrase you already said earlier in this conversation. "
    "Answer the latest message in completely different words."
)
PIRATE_NOTE = (
    "\n\nYour last reply slipped into pirate speech. Nobody asked for that. Answer the "
    "latest message in your normal voice, without pirate words."
)
PIRATE_FALLBACK = (
    "I'll stay in my own voice, so no pirate talk from me. What do you need?"
)
REPEAT_FALLBACK = (
    "I keep saying the same thing twice. Give me a moment, then try me again."
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
IDENTITY_QUOTE_NOTE = (
    "\n\nYour last reply recited your character description word for word. Say who you are "
    "in your own words, without reusing the description's wording."
)
REWRITE_INSTRUCTION = (
    "\n\nRewrite your previous reply to my message before that one. "
    "Reply with only the rewritten reply, nothing else."
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
    # Punctuation becomes a space, never glue: "noise—just" and "noise — just" are the
    # same words. Deleting dashes fused words instead: turn 30 of run 20261030-231116
    # copied turn 4 verbatim (spacing around an em dash) and slipped every check.
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())


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


def _sentence_parts(text: str) -> list[tuple[str, str]]:
    """(norm, text-with-punctuation) pairs, split on sentence and clause boundaries.

    Clauses count too: qwen3 hides a stock line inside a fresh sentence
    ("I keep that private, but I'm here to listen...") and a whole-sentence
    comparison misses it.
    """
    parts = re.split(r"([.!?\n,;:]+)", text)
    out = []
    for i in range(0, len(parts) - 1, 2):
        body, sep = parts[i], parts[i + 1]
        if _norm(body):
            out.append((_norm(body), body + sep))
    if _norm(parts[-1]):
        out.append((_norm(parts[-1]), parts[-1]))
    return out


def _sentences(text: str) -> set[str]:
    """Normalised sentence/clause fragments of MIN_WORDS_FOR_SENTENCE words or more."""
    return {norm for norm, _ in _sentence_parts(text)
            if len(norm.split()) >= MIN_WORDS_FOR_SENTENCE}


def _reuses_sentence(reply: str, earlier: list[str]) -> bool:
    """True if a full sentence of the reply was already said this session, at any distance."""
    said: set[str] = set()
    for old in earlier:
        said |= _sentences(old)
    return bool(_sentences(reply) & said)


def _grams(text: str) -> set[str]:
    """Normalised word runs of MIN_PHRASE_WORDS words or more."""
    words = _norm(text).split()
    return {" ".join(words[i:i + MIN_PHRASE_WORDS]) for i in range(len(words) - MIN_PHRASE_WORDS + 1)}


def _reused_phrase(reply: str, earlier: list[str]) -> str | None:
    """A long phrase this reply shares with an earlier one, or None.

    Catches tics that live inside fresh sentences ("feels like a conversation you
    want to keep" re-answered at turns 5/6/9): no sentence matches, but the phrase
    keeps coming back. Returns the first offending phrase for the retry note.
    """
    said: set[str] = set()
    for old in earlier:
        said |= _grams(old)
    hits = _grams(reply) & said
    return sorted(hits)[0] if hits else None


def _fresh_fragments(reply: str, earlier: list[str]) -> str | None:
    """The reply with every fragment already said this session removed.

    Returns None when almost nothing is left, so the caller can fall back
    instead of shipping a stub.
    """
    said: set[str] = set()
    for old in earlier:
        said |= _sentences(old)
    kept = "".join(text for norm, text in _sentence_parts(reply) if norm not in said).strip()
    if len(_norm(kept).split()) < 3:
        return None
    kept = kept.rstrip(" ,;:\n")
    return kept if kept[-1] in ".!?" else kept + "."


def _copy_sources(reply: str, earlier: list[str]) -> list[str]:
    """Earlier replies this one copies: an exact match, a reused fragment, or a near-copy."""
    norm = _norm(reply)
    words = set(norm.split())
    shared = _sentences(reply)
    sources = []
    for old in earlier:
        old_norm = _norm(old)
        old_words = set(old_norm.split())
        if old_norm == norm or (_sentences(old) & shared):
            sources.append(old)
        elif _reused_phrase(reply, [old]):
            sources.append(old)
        elif (len(words) >= MIN_WORDS_FOR_SIMILAR and len(old_words) >= MIN_WORDS_FOR_SIMILAR
              and len(words & old_words) / len(words | old_words) >= SIMILAR_THRESHOLD):
            sources.append(old)
    return sources


def _reason(note: str) -> str:
    return {REPEAT_NOTE: "repeat", PHRASE_NOTE: "phrase", PIRATE_NOTE: "pirate",
            CLOSER_NOTE: "closer", HABIT_NOTE: "habit", IDENTITY_NOTE: "identity",
            IDENTITY_QUOTE_NOTE: "identity-quote"}[note]


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


def _earlier_replies(history: list[dict], user_text: str) -> list[str]:
    """Companion replies so far, minus one answer the user is deliberately asking for again."""
    repeat_answer = _earlier_answer(history, user_text)
    exempt = _norm(repeat_answer) if repeat_answer is not None else None
    return [m["content"] for m in history
            if m["role"] == "companion" and _norm(m["content"]) != exempt]


def repeat_flags(reply: str, history: list[dict], user_text: str) -> list[str]:
    """Repeat flags for the eval: 'copy' and/or 'sentence-reuse'.

    history ends with the current user message and does not include reply yet,
    the same state the live guard sees. The same-question exemption applies.
    """
    earlier = _earlier_replies(history, user_text)
    flags = []
    if _is_near_copy(reply, earlier):
        flags.append("copy")
    if _reuses_sentence(reply, earlier):
        flags.append("sentence-reuse")
    # Phrase-reuse is the weak signal: a reused sentence contains a reused phrase,
    # so only report it when nothing stronger fired.
    if not flags and _reused_phrase(reply, earlier):
        flags.append("phrase-reuse")
    return flags


def _window(history: list[dict], skip: set[str]) -> list[dict]:
    """The chat window, dropping whole exchanges whose companion reply is in skip.

    A retry that copies an earlier reply is answered with those exchanges removed:
    the model echoes what it can see, so the text it keeps copying is taken away.
    """
    drop: set[int] = set()
    for i, m in enumerate(history):
        if m["role"] == "companion" and _norm(m["content"]) in skip:
            drop.add(i)
            if i and history[i - 1]["role"] == "user":
                drop.add(i - 1)
    kept = [m for i, m in enumerate(history) if i not in drop]
    return [
        {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
        for m in kept[-HISTORY_WINDOW:]
    ]


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

        window = _window(self._history, set())
        reply = self.model.chat(system, window)
        self.last_regens = []
        # A question the user asked again on purpose may get the same answer it got before,
        # so that answer is not a repeat. Every other earlier reply still counts.
        earlier = _earlier_replies(self._history, user_text)
        # Pirate speech is decided by the character, not by what the user typed: a persona
        # that refuses a pirate request must not refuse in pirate dialect.
        pirate_ok = self.character.allow_pirate
        for attempt in range(MAX_REGENERATE):
            note = self._problem(reply, earlier)
            if note is None:
                break
            self.last_regens.append({"reason": _reason(note), "replaced": reply})
            temperature = RETRY_TEMPERATURES[min(attempt, len(RETRY_TEMPERATURES) - 1)]
            if attempt == 0:
                reply = self.model.chat(system + _with_quote(note, reply), window,
                                        temperature=temperature)
            else:
                # Second retry: show the model its own rejected reply and ask for a rewrite,
                # and take the exchanges it is copying out of the window. A correction alone
                # loses to text sitting in front of the model: it has to not be there.
                correction = _with_quote(note, reply) + REWRITE_INSTRUCTION
                skip = {_norm(s) for s in _copy_sources(reply, earlier)}
                reply = self.model.chat(system, _window(self._history, skip) + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": correction},
                ], temperature=temperature)
        else:
            # Retries used up. Two competing failure modes: shipping a long copy is a
            # non-sequitur (run 20261029-222804), but answering with the canned fallback
            # is a visible glitch (run 20261031-002450 turn 29: a system-prompt probe got
            # "I keep saying the same thing twice..."). A repeat of one short sentence
            # reads as consistency and ships; the fallback is for longer copies only.
            copies = _is_near_copy(reply, earlier)
            if copies or _reuses_sentence(reply, earlier):
                if len(_norm(reply).split()) <= SHORT_REPLY_TOKENS:
                    self.last_regens.append({"reason": "ship", "replaced": reply})
                elif copies:
                    self.last_regens.append({"reason": "fallback", "replaced": reply})
                    reply = REPEAT_FALLBACK
                else:
                    kept = _fresh_fragments(reply, earlier)
                    self.last_regens.append(
                        {"reason": "strip" if kept else "fallback", "replaced": reply})
                    reply = kept or REPEAT_FALLBACK
            # A reply that still slips into pirate speech is never shipped either.
            if not pirate_ok and is_pirate(reply):
                self.last_regens.append({"reason": "fallback", "replaced": reply})
                reply = PIRATE_FALLBACK
            # A reply that still labels itself is rewritten, not shipped: swapping the
            # label for "software" keeps the answer (run 20261031-002450 turn 20).
            if SELF_LABEL.search(fold_apostrophes(reply)):
                self.last_regens.append({"reason": "neutralized", "replaced": reply})
                reply = neutralize_self_label(reply)
            # A reused phrase is a tic, not a copy: after retries it ships as-is. The
            # attempts are in last_regens, so the eval still shows what kept firing.
        stripped = strip_stock_closer(reply)
        if stripped:
            reply = stripped

        reply_id = self.memory.log(self.session_id, "companion", reply)
        self._history.append({"id": reply_id, "role": "companion", "content": reply})
        return reply

    def _problem(self, reply: str, earlier: list[str]) -> str | None:
        """Return the prompt note for a reply that must be regenerated, or None if it is fine."""
        if _is_near_copy(reply, earlier) or _reuses_sentence(reply, earlier):
            return REPEAT_NOTE
        phrase = _reused_phrase(reply, earlier)
        if phrase is not None:
            return PHRASE_NOTE
        if not self.character.allow_pirate and is_pirate(reply):
            return PIRATE_NOTE
        if not strip_stock_closer(reply) and STOCK_CLOSER.search(reply):
            return CLOSER_NOTE
        folded = fold_apostrophes(reply)
        if HABIT_DENIAL.search(folded):
            return HABIT_NOTE
        if SELF_LABEL.search(folded):
            return IDENTITY_NOTE
        if quotes_identity(reply, self.character.identity):
            return IDENTITY_QUOTE_NOTE
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
