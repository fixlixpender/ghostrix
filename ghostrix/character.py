"""Character: the user-defined identity loaded every session.

Stored as a TOML file with three required fields:

    name = "Ash"
    identity = "A dry, patient companion who remembers what matters to you."
    system_prompt = "Stay in character. Keep replies short unless asked."
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("name", "identity", "system_prompt")
OPTIONAL_FIELDS = ("allow_pirate",)  # bool; default False. Only a character that opts in may use pirate speech.


@dataclass(frozen=True)
class Character:
    name: str
    identity: str
    system_prompt: str
    allow_pirate: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Character":
        missing = [f for f in REQUIRED_FIELDS if not str(data.get(f, "")).strip()]
        if missing:
            raise ValueError(f"character is missing: {', '.join(missing)}")
        unknown = set(data) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
        if unknown:
            raise ValueError(f"unknown character fields: {', '.join(sorted(unknown))}")
        allow_pirate = data.get("allow_pirate", False)
        if not isinstance(allow_pirate, bool):
            raise ValueError("allow_pirate must be true or false")
        return cls(
            name=str(data["name"]).strip(),
            identity=str(data["identity"]).strip(),
            system_prompt=str(data["system_prompt"]).strip(),
            allow_pirate=allow_pirate,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Character":
        return cls.from_dict(tomllib.loads(Path(path).read_text(encoding="utf-8")))
