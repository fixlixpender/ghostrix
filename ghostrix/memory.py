"""Ghostrix memory core.

Raw log of every session, session summaries that link back to their source
messages, retrieval over both, and user-approved profile entries.

The embedder is pluggable. The default HashEmbedder is a deterministic
bag-of-words stub: it exercises the plumbing, it does not measure retrieval
quality. Real quality is measured with the local model on the user's machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Protocol

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    started_at  REAL NOT NULL,
    ended_at    REAL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'companion')),
    content     TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    created_at  REAL NOT NULL,
    UNIQUE (session_id, seq)
);

CREATE TABLE IF NOT EXISTS summaries (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL UNIQUE REFERENCES sessions(id),
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_sources (
    summary_id  INTEGER NOT NULL REFERENCES summaries(id),
    message_id  INTEGER NOT NULL REFERENCES messages(id),
    PRIMARY KEY (summary_id, message_id)
);

CREATE TABLE IF NOT EXISTS profile_entries (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('proposed', 'accepted', 'rejected')),
    created_at  REAL NOT NULL,
    decided_at  REAL
);
"""

_STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have he her his i if in into is it "
    "its me my no not of on or our she so that the their them then there they this to "
    "was we were what when where which who why will with would you your".split()
)


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic bag-of-words stub. Plumbing only, not a quality measure."""

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in re.findall(r"[a-z0-9']+", text.lower()):
            if tok in _STOPWORDS:
                continue
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class Hit:
    kind: str  # "summary" or "snippet"
    id: int
    text: str
    score: float
    source_message_ids: tuple[int, ...]


class Memory:
    def __init__(self, path: str, embedder: Embedder | None = None) -> None:
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.embedder: Embedder = embedder or HashEmbedder()

    def close(self) -> None:
        self.db.close()

    # Sessions and raw log -------------------------------------------------

    def start_session(self) -> int:
        with self.db:
            cur = self.db.execute("INSERT INTO sessions(started_at) VALUES (?)", (time.time(),))
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def _require_open(self, session_id: int) -> None:
        row = self.db.execute("SELECT ended_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown session {session_id}")
        if row["ended_at"] is not None:
            raise ValueError(f"session {session_id} has ended")

    def log(self, session_id: int, role: str, content: str) -> int:
        """Append one message to the raw log. Returns the message id."""
        self._require_open(session_id)
        if role not in ("user", "companion"):
            raise ValueError(f"bad role {role!r}")
        seq = self.db.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        emb = json.dumps(self.embedder.embed(content))
        with self.db:
            cur = self.db.execute(
                "INSERT INTO messages(session_id, seq, role, content, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, seq, role, content, emb, time.time()),
            )
        return cur.lastrowid

    def messages(self, session_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id, seq, role, content FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()

    # Summaries ------------------------------------------------------------

    def end_session(self, session_id: int, summary: str, source_message_ids: list[int]) -> int:
        """Close a session with a summary that links to its source messages.

        Every summary must cite at least one message from this same session.
        Returns the summary id.
        """
        self._require_open(session_id)
        if not summary.strip():
            raise ValueError("empty summary")
        sources = sorted(set(source_message_ids))
        if not sources:
            raise ValueError("summary must link to at least one source message")
        placeholders = ",".join("?" * len(sources))
        found = self.db.execute(
            f"SELECT id FROM messages WHERE session_id = ? AND id IN ({placeholders})",
            (session_id, *sources),
        ).fetchall()
        if len(found) != len(sources):
            raise ValueError("source messages must all belong to this session")
        with self.db:
            cur = self.db.execute(
                "INSERT INTO summaries(session_id, text, embedding, created_at) VALUES (?, ?, ?, ?)",
                (session_id, summary, json.dumps(self.embedder.embed(summary)), time.time()),
            )
            summary_id = cur.lastrowid
            self.db.executemany(
                "INSERT INTO summary_sources(summary_id, message_id) VALUES (?, ?)",
                [(summary_id, m) for m in sources],
            )
            self.db.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time(), session_id))
        return summary_id

    def end_session_empty(self, session_id: int) -> None:
        """Close a session that has no messages. No summary is created."""
        self._require_open(session_id)
        count = self.db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        if count:
            raise ValueError("session has messages; use end_session with a summary")
        with self.db:
            self.db.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time(), session_id))

    def sources_of(self, summary_id: int) -> tuple[int, ...]:
        rows = self.db.execute(
            "SELECT message_id FROM summary_sources WHERE summary_id = ? ORDER BY message_id",
            (summary_id,),
        ).fetchall()
        return tuple(r["message_id"] for r in rows)

    # Retrieval ------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        """Top-k matches across summaries and raw snippets, by cosine similarity."""
        q = self.embedder.embed(query)
        hits: list[Hit] = []
        for r in self.db.execute("SELECT id, text, embedding FROM summaries"):
            hits.append(
                Hit("summary", r["id"], r["text"], _dot(q, json.loads(r["embedding"])),
                    self.sources_of(r["id"]))
            )
        for r in self.db.execute("SELECT id, content, embedding FROM messages"):
            hits.append(
                Hit("snippet", r["id"], r["content"], _dot(q, json.loads(r["embedding"])), (r["id"],))
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # Profile --------------------------------------------------------------

    def propose_profile(self, text: str) -> int:
        """The companion proposes a profile change. Nothing changes until accepted."""
        if not text.strip():
            raise ValueError("empty profile entry")
        with self.db:
            cur = self.db.execute(
                "INSERT INTO profile_entries(text, status, created_at) VALUES (?, 'proposed', ?)",
                (text, time.time()),
            )
        return cur.lastrowid

    def decide_profile(self, entry_id: int, accept: bool) -> None:
        row = self.db.execute("SELECT status FROM profile_entries WHERE id = ?", (entry_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown profile entry {entry_id}")
        if row["status"] != "proposed":
            raise ValueError(f"entry {entry_id} is already {row['status']}")
        with self.db:
            self.db.execute(
                "UPDATE profile_entries SET status = ?, decided_at = ? WHERE id = ?",
                ("accepted" if accept else "rejected", time.time(), entry_id),
            )

    def stable_profile(self) -> list[str]:
        rows = self.db.execute(
            "SELECT text FROM profile_entries WHERE status = 'accepted' ORDER BY id"
        ).fetchall()
        return [r["text"] for r in rows]
