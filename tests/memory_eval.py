"""Real-model memory test. Run it on the machine with Ollama and the chat model.

    python3 -m tests.memory_eval --character character.toml

Phase 1: teach the 20 facts over 5 sessions, through the real chat, with real
summaries written by the model at the end of each session.
Phase 2: close the memory file, reopen it (next-day analogue), and ask all 20
questions in a new session.

Pass threshold from the brief: at least 17 of 20 answered correctly, and zero
wrong or invented facts. The script checks the first part automatically. It
cannot judge invention, so it prints every answer for review. Read them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from ghostrix.character import Character
from ghostrix.companion import Companion
from ghostrix.config import load_config
from ghostrix.memory import Memory, _STOPWORDS
from ghostrix.ollama import OllamaClient
from tests.test_memory import FACTS  # the same 20 facts the stub test uses

THRESHOLD = 17

# Equivalent forms a correct answer may take.
ACCEPTED = {
    "four": ["four", "4"],
    "three": ["three", "3"],
    "twenty": ["twenty", "27", "twenty-seventh", "27th", "twenty seventh"],
}


def accepted_forms(answer: str) -> list[str]:
    return ACCEPTED.get(answer, [answer])


def contains_answer(text: str, answer: str) -> bool:
    lowered = text.lower()
    return any(form.lower() in lowered for form in accepted_forms(answer))


def _has_word(text: str, word: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(word.lower()) + r"(?![a-z0-9])", text.lower()) is not None


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def cross_mixed_with(index: int, reply: str) -> str | None:
    """Return what the reply carries in from another fact, or None.

    Two checks. First, another fact's expected answer appears in the reply ("twenty").
    Second, a content word from another fact's statement appears ("five" from
    "Phoebe turns five"). A word counts only if it is not in this fact's own statement
    or question, so "blue" in the spare-key reply is not flagged.
    Returns the offending word or answer, or None.
    """
    statement, question, _ = FACTS[index]
    own = _content_words(statement) | _content_words(question)
    for j, (other_statement, _q, other_answer) in enumerate(FACTS):
        if j == index:
            continue
        for form in accepted_forms(other_answer):
            if _has_word(reply, form) and not _has_word(statement, form) and not _has_word(question, form):
                return other_answer
        for word in sorted(_content_words(other_statement) - own):
            if _has_word(reply, word):
                return word
    return None


def run(character_path: str, config_path: str | None, out_root: str | None) -> int:
    cfg = load_config(config_path)
    character = Character.load(character_path)
    client = OllamaClient(cfg.ollama_host, cfg.chat_model, cfg.embedding_model)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path(os.path.expanduser(out_root or cfg.data_dir)) / "eval" / stamp
    base.mkdir(parents=True, exist_ok=True)
    db_path = str(base / "memory.db")
    print(f"store: {db_path}")

    # Phase 1: teach
    memory = Memory(db_path, embedder=client)
    for session_no, start in enumerate(range(0, 20, 4), start=1):
        companion = Companion(memory, client, character, retrieval_k=cfg.retrieval_k)
        for statement, _question, _answer in FACTS[start:start + 4]:
            companion.turn(statement)
        summary_id = companion.end()
        print(f"taught session {session_no}/5 (summary {summary_id})", flush=True)
    memory.close()

    # Phase 2: recall after reopening the store
    memory = Memory(db_path, embedder=client)
    companion = Companion(memory, client, character, retrieval_k=cfg.retrieval_k)
    rows = []
    for index, (_statement, question, answer) in enumerate(FACTS):
        reply = companion.turn(question)
        hits = memory.retrieve(question, k=cfg.retrieval_k)
        row = {
            "question": question,
            "expected": answer,
            "reply": reply,
            "answer_correct": contains_answer(reply, answer),
            "retrieved_in_top_k": any(contains_answer(h.text, answer) for h in hits),
            "cross_mixed_with": cross_mixed_with(index, reply),
        }
        rows.append(row)
        mark = "OK " if row["answer_correct"] else "MISS"
        if row["cross_mixed_with"]:
            mark += " MIXED"
        print(f"[{mark}] {question}\n       expected={answer!r}\n       reply={reply}", flush=True)
    companion.end()
    memory.close()

    correct = sum(r["answer_correct"] for r in rows)
    retrieved = sum(r["retrieved_in_top_k"] for r in rows)
    mixed = [r for r in rows if r["cross_mixed_with"]]
    verdict = "PASS" if correct >= THRESHOLD and not mixed else "FAIL"
    report = {
        "answers_correct": correct,
        "retrieved_in_top_k": retrieved,
        "cross_mixed_replies": len(mixed),
        "threshold": THRESHOLD,
        "verdict_before_manual_review": verdict,
        "rows": rows,
    }
    (base / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\nanswers correct: {correct}/20 (threshold {THRESHOLD})  "
        f"retrieval hits: {retrieved}/20  mixed-in replies: {len(mixed)}  -> {verdict} before review"
    )
    print("Now read every reply for wrong or invented facts. Results: " + str(base / "results.json"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory_eval")
    parser.add_argument("--character", default="character.toml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="output root (default: the data dir)")
    args = parser.parse_args(argv)
    return run(args.character, args.config, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
