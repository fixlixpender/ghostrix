"""Ghostrix command-line chat.

    python3 -m ghostrix --character character.toml [--config config.toml]

Type to talk. /quit (or Ctrl-D) ends the session: the companion writes a
summary linked to every message, and it is stored in the memory file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .character import Character
from .companion import Companion
from .config import load_config
from .memory import Memory
from .ollama import OllamaClient, OllamaError


def main(argv: list[str] | None = None, input_fn=input, output=print) -> int:
    parser = argparse.ArgumentParser(prog="ghostrix")
    parser.add_argument("--character", default="character.toml")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        character = Character.load(args.character)
    except (OSError, ValueError) as exc:
        output(f"setup error: {exc}")
        return 1

    data_dir = Path(os.path.expanduser(cfg.data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory(str(data_dir / "memory.db"), embedder=_embedder(cfg))
    model = OllamaClient(cfg.ollama_host, cfg.chat_model, cfg.embedding_model)
    companion = Companion(memory, model, character, retrieval_k=cfg.retrieval_k)

    output(f"{character.name} is here. /quit to end the session.")
    try:
        while True:
            try:
                line = input_fn("you> ")
            except EOFError:
                break
            if line.strip() == "/quit":
                break
            if not line.strip():
                continue
            try:
                reply = companion.turn(line)
            except OllamaError as exc:
                output(f"[model error] {exc}")
                continue
            output(f"{character.name}> {reply}")
    except KeyboardInterrupt:
        output("")

    try:
        summary_id = companion.end()
        output("session saved." if summary_id else "nothing said; no summary written.")
    except OllamaError as exc:
        output(f"[could not summarise: {exc}] session left open; messages are still stored.")
    finally:
        memory.close()
    return 0


def _embedder(cfg):
    """Embeddings come from Ollama. Built lazily so tests can run without a server."""
    return OllamaClient(cfg.ollama_host, cfg.chat_model, cfg.embedding_model)


if __name__ == "__main__":
    sys.exit(main())
