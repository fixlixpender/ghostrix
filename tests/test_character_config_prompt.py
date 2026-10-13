"""Tests for character, config, and prompt assembly. Stdlib only."""

import os
import tempfile
import unittest

from ghostrix.character import Character
from ghostrix.config import Config, load_config
from ghostrix.prompt import build_system_prompt, memories_block

GOOD = {
    "name": "Ash",
    "identity": "A dry, patient companion.",
    "system_prompt": "Stay in character.",
}


class CharacterTests(unittest.TestCase):
    def test_loads_valid_character(self):
        c = Character.from_dict(GOOD)
        self.assertEqual(c.name, "Ash")

    def test_missing_field_rejected(self):
        bad = dict(GOOD)
        del bad["identity"]
        with self.assertRaisesRegex(ValueError, "identity"):
            Character.from_dict(bad)

    def test_blank_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "name"):
            Character.from_dict({**GOOD, "name": "   "})

    def test_unknown_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            Character.from_dict({**GOOD, "voice": "gravel"})

    def test_loads_from_toml_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "character.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write('name = "Ash"\nidentity = "Patient."\nsystem_prompt = "Be brief."\n')
            c = Character.load(path)
            self.assertEqual(c.system_prompt, "Be brief.")


class ConfigTests(unittest.TestCase):
    def test_defaults_when_no_file(self):
        self.assertEqual(load_config(None), Config())
        self.assertEqual(load_config("/nonexistent/path.toml"), Config())

    def test_file_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write('chat_model = "qwen3:4b"\nretrieval_k = 3\n')
            cfg = load_config(path)
            self.assertEqual(cfg.chat_model, "qwen3:4b")
            self.assertEqual(cfg.retrieval_k, 3)
            self.assertEqual(cfg.embedding_model, Config().embedding_model)

    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write('cloud_fallback = "openai"\n')
            with self.assertRaisesRegex(ValueError, "cloud_fallback"):
                load_config(path)

    def test_bad_values_rejected(self):
        with self.assertRaises(ValueError):
            Config(retrieval_k=0)
        with self.assertRaises(ValueError):
            Config(ollama_host="localhost:11434")


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.char = Character.from_dict(GOOD)

    def test_character_always_present(self):
        p = build_system_prompt(self.char, [], [])
        self.assertIn("Stay in character.", p)
        self.assertIn("You are Ash.", p)

    def test_profile_and_memories_included(self):
        p = build_system_prompt(self.char, ["Likes espresso."], ["Dog is called Biscuit."])
        self.assertIn("- Likes espresso.", p)
        self.assertIn("- Dog is called Biscuit.", p)
        self.assertIn("may be incomplete", p)

    def test_memory_budget_respected_and_ordered(self):
        snippets = ["a" * 500, "b" * 500, "c" * 500]
        kept = memories_block(snippets, max_chars=1000)
        self.assertEqual(kept[0], "a" * 500)
        self.assertLessEqual(sum(len(k) for k in kept), 1000)
        self.assertFalse(any(k.startswith("c") and len(k) == 500 for k in kept))

    def test_memory_whitespace_collapsed_and_empty_dropped(self):
        self.assertEqual(memories_block(["  hello \n world  ", "   "]), ["hello world"])


if __name__ == "__main__":
    unittest.main()
