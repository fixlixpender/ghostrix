"""Config: defaults plus an optional TOML file. Unknown keys are errors.

Model names are defaults. Chat model: Qwen3 (fixed in the brief).
Embedding model: nomic-embed-text (decided by Fil, 2026-10-13).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    chat_model: str = "qwen3:latest"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://127.0.0.1:11434"
    data_dir: str = "~/.ghostrix"
    character_path: str = "character.toml"
    retrieval_k: int = 3

    def __post_init__(self) -> None:
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be at least 1")
        if not self.ollama_host.startswith(("http://", "https://")):
            raise ValueError("ollama_host must start with http:// or https://")
        if not self.chat_model.strip() or not self.embedding_model.strip():
            raise ValueError("model names cannot be empty")


def load_config(path: str | Path | None = None) -> Config:
    """Defaults if no file; otherwise the file overrides defaults."""
    if path is None or not Path(path).exists():
        return Config()
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    return Config(**data)
