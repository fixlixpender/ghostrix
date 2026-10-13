"""Chat loop tests. A fake Ollama HTTP server stands in for the model.

These prove the wiring (HTTP shape, prompt, memory logging, summary linking).
They do not prove what Qwen3 says. That is tested on the user's machine.
"""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ghostrix.character import Character
from ghostrix.companion import Companion
from ghostrix.memory import Memory
from ghostrix.ollama import OllamaClient, OllamaError
from ghostrix.__main__ import main

CHARACTER = Character.from_dict(
    {"name": "Ash", "identity": "Patient.", "system_prompt": "Stay in character."}
)


class FakeOllama:
    """Records requests; replies with canned content."""

    def __init__(self):
        self.requests = []  # (path, payload)
        self.chat_reply = "Noted."
        self.summary_reply = "The user shared a fact."
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append((self.path, payload))
                if self.path == "/api/chat":
                    is_summary = payload["messages"][0]["content"].startswith("Summarise")
                    content = outer.summary_reply if is_summary else outer.chat_reply
                    body = {"message": {"role": "assistant", "content": f"  {content}  "}}
                elif self.path == "/api/embed":
                    body = {"embeddings": [[1.0, 0.0, 0.0]]}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class OllamaClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.client = OllamaClient(self.fake.url, "qwen3:latest", "nomic-embed-text")

    def tearDown(self):
        self.fake.close()

    def test_chat_sends_system_first_thinking_off_and_strips_reply(self):
        reply = self.client.chat("SYSTEM", [{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "Noted.")
        path, payload = self.fake.requests[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "SYSTEM"})
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "qwen3:latest")

    def test_embed_uses_embedding_model(self):
        vec = self.client.embed("some text")
        self.assertEqual(vec, [1.0, 0.0, 0.0])
        path, payload = self.fake.requests[-1]
        self.assertEqual(path, "/api/embed")
        self.assertEqual(payload["model"], "nomic-embed-text")

    def test_unreachable_server_gives_clear_error(self):
        dead = OllamaClient("http://127.0.0.1:1", "m", "e", timeout=2)
        with self.assertRaisesRegex(OllamaError, "cannot reach Ollama"):
            dead.chat("s", [])


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))
        self.fake = FakeOllama()
        self.client = OllamaClient(self.fake.url, "qwen3:latest", "nomic-embed-text")
        self.comp = Companion(self.mem, self.client, CHARACTER)

    def tearDown(self):
        self.fake.close()
        self.mem.close()
        self._tmp.cleanup()

    def test_turn_logs_both_sides(self):
        reply = self.comp.turn("My dog is called Biscuit.")
        self.assertEqual(reply, "Noted.")
        roles = [r["role"] for r in self.mem.messages(self.comp.session_id)]
        self.assertEqual(roles, ["user", "companion"])

    def test_accepted_profile_reaches_the_system_prompt(self):
        pid = self.mem.propose_profile("Drinks espresso.")
        self.mem.decide_profile(pid, accept=True)
        self.comp.turn("hello")
        system = self.fake.requests[-1][1]["messages"][0]["content"]
        self.assertIn("Drinks espresso.", system)

    def test_end_links_summary_to_every_message(self):
        self.comp.turn("first")
        self.comp.turn("second")
        summary_id = self.comp.end()
        assert summary_id is not None
        ids = sorted(r["id"] for r in self.mem.messages(self.comp.session_id))
        self.assertEqual(self.mem.sources_of(summary_id), tuple(ids))

    def test_empty_session_closes_without_summary(self):
        self.assertIsNone(self.comp.end())
        with self.assertRaises(ValueError):
            self.mem.log(self.comp.session_id, "user", "too late")

    def test_empty_summary_from_model_is_refused(self):
        self.fake.summary_reply = "   "
        self.comp.turn("something")
        with self.assertRaises(ValueError):
            self.comp.end()


class MainLoopTests(unittest.TestCase):
    """Runs the real CLI entry point against the fake server."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.fake = FakeOllama()
        self.char_path = os.path.join(self.dir, "character.toml")
        with open(self.char_path, "w", encoding="utf-8") as f:
            f.write('name = "Ash"\nidentity = "Patient."\nsystem_prompt = "Be brief."\n')
        self.cfg_path = os.path.join(self.dir, "config.toml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            f.write(f'ollama_host = "{self.fake.url}"\ndata_dir = "{self.dir}/data"\n')

    def tearDown(self):
        self.fake.close()
        self._tmp.cleanup()

    def run_cli(self, lines):
        feed = iter(lines)
        out = []

        def fake_input(_prompt):
            try:
                return next(feed)
            except StopIteration:
                raise EOFError

        code = main(
            ["--character", self.char_path, "--config", self.cfg_path],
            input_fn=fake_input,
            output=out.append,
        )
        return code, out

    def test_full_session_writes_memory_file(self):
        code, out = self.run_cli(["my dog is Biscuit", "/quit"])
        self.assertEqual(code, 0)
        self.assertTrue(any("Ash> Noted." in line for line in out))
        self.assertIn("session saved.", out)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "data", "memory.db")))

    def test_empty_session_says_so(self):
        code, out = self.run_cli(["/quit"])
        self.assertEqual(code, 0)
        self.assertIn("nothing said; no summary written.", out)

    def test_bad_character_file_is_a_setup_error(self):
        code = main(["--character", "/no/such/file.toml"], input_fn=lambda _p: "", output=lambda _m: None)
        self.assertEqual(code, 1)

    def test_unreachable_model_does_not_crash(self):
        self.fake.close()
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            f.write(f'ollama_host = "http://127.0.0.1:1"\ndata_dir = "{self.dir}/data"\n')
        self.fake = FakeOllama()  # so tearDown has something to close
        code, out = self.run_cli(["hello", "/quit"])
        self.assertEqual(code, 0)
        self.assertTrue(any("[model error]" in line for line in out))


if __name__ == "__main__":
    unittest.main()
