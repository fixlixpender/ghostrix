"""Repeat guard: a reply that copies one of the last few companion replies is regenerated."""

import os
import tempfile
import unittest

from ghostrix.character import Character
from ghostrix.companion import Companion
from ghostrix.memory import Memory

CHARACTER = Character.from_dict(
    {"name": "Ash", "identity": "Patient.", "system_prompt": "Stay in character."}
)


class ScriptedModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, messages):
        self.calls.append(system)
        return self.replies.pop(0)


class RepeatGuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_repeated_reply_is_regenerated(self):
        model = ScriptedModel(["Lisbon.", "Lisbon.", "Still Lisbon, new words."])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("capital of Portugal?"), "Lisbon.")
        self.assertEqual(comp.turn("capital again?"), "Still Lisbon, new words.")
        self.assertEqual(len(model.calls), 3)
        self.assertIn("repeated a line", model.calls[2])

    def test_repeat_match_ignores_case_and_punctuation(self):
        model = ScriptedModel(["Yes, I like rain.", "yes i like rain", "Different now."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("do you like rain?")
        self.assertEqual(comp.turn("do you like rain?"), "Different now.")

    def test_new_reply_is_not_regenerated(self):
        model = ScriptedModel(["First.", "Second."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("one")
        self.assertEqual(comp.turn("two"), "Second.")
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
