"""Ollama client. Stdlib only. Talks to a local Ollama server over HTTP.

Endpoints used:
  POST /api/chat   chat completion (non-streaming, thinking disabled)
  POST /api/embed  embeddings
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, chat_model: str, embed_model: str, timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Ollama answered, so the server is up. The body usually says why (e.g. model not found).
            detail = exc.read().decode("utf-8", "replace").strip()
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            raise OllamaError(
                f"Ollama at {self.host} answered HTTP {exc.code} for {path}: {detail[:300] or 'no detail'}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({exc.reason}). Is it running?"
            ) from exc
        except (ValueError, KeyError) as exc:
            raise OllamaError(f"unexpected response from Ollama: {exc}") from exc

    def chat(self, system: str, messages: list[dict]) -> str:
        payload = {
            "model": self.chat_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            # Qwen3 thinking output would leak into replies and memory summaries.
            "think": False,
        }
        return self._post("/api/chat", payload)["message"]["content"].strip()

    def embed(self, text: str) -> list[float]:
        data = self._post("/api/embed", {"model": self.embed_model, "input": text})
        return data["embeddings"][0]
