"""Companion: one conversation session on top of the memory core and a chat model.

The chat model is anything with chat(system, messages) -> str, so tests can use
a fake and the real run uses OllamaClient.
"""

from __future__ import annotations

from typing import Protocol

from .character import Character
from .memory import Memory
from .prompt import build_system_prompt, memories_block

HISTORY_WINDOW = 20  # messages sent to the model each turn; older ones live in memory only

SUMMARY_INSTRUCTION = (
    "Summarise the conversation below in under 120 words. Keep every concrete fact: "
    "names, dates, numbers, places, preferences. Write it as plain statements. "
    "Output only the summary."
)


class ChatModel(Protocol):
    def chat(self, system: str, messages: list[dict]) -> str: ...


class Companion:
    def __init__(self, memory: Memory, model: ChatModel, character: Character, retrieval_k: int = 5) -> None:
        self.memory = memory
        self.model = model
        self.character = character
        self.retrieval_k = retrieval_k
        self.session_id = memory.start_session()
        self._history: list[dict] = []  # {"id", "role", "content"}; role is user or companion

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

        reply_id = self.memory.log(self.session_id, "companion", reply)
        self._history.append({"id": reply_id, "role": "companion", "content": reply})
        return reply

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
