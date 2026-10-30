"""Repeat guard: a reply that copies one of the last few companion replies is regenerated."""

import os
import tempfile
import unittest

from ghostrix.character import Character
from ghostrix.companion import REPEAT_FALLBACK, Companion
from ghostrix.memory import Memory

CHARACTER = Character.from_dict(
    {"name": "Ash", "identity": "Patient.", "system_prompt": "Stay in character."}
)


class ScriptedModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.msgs = []

    def chat(self, system, messages, temperature=None):
        self.calls.append((system, temperature))
        self.msgs.append(list(messages))
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
        self.assertIn("repeated a line", model.calls[2][0])
        self.assertIsNone(model.calls[1][1])
        self.assertIsNotNone(model.calls[2][1])

    def test_repeat_match_ignores_case_and_punctuation(self):
        model = ScriptedModel(["Yes, I like rain.", "yes i like rain", "Different now."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("do you like rain?")
        # A different question is not an exempt repeat, so the copy is still regenerated.
        self.assertEqual(comp.turn("Rain, though: do you like it?"), "Different now.")

    def test_repeat_from_early_in_session_is_caught(self):
        replies = ["The quiet before the night."] + [f"Other line {i}." for i in range(6)]
        replies.append("The quiet before the night.")
        replies.append("Something new.")
        model = ScriptedModel(replies)
        comp = Companion(self.mem, model, CHARACTER)
        for i in range(7):
            comp.turn(f"msg {i}")
        self.assertEqual(comp.turn("late question"), "Something new.")

    def test_new_reply_is_not_regenerated(self):
        model = ScriptedModel(["First.", "Second."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("one")
        self.assertEqual(comp.turn("two"), "Second.")
        self.assertEqual(len(model.calls), 2)

    def test_copy_surviving_retries_falls_back(self):
        # Run 20261029-222804 turns 12/13, 22/23, 21/29: the model returned the copy
        # every time, and the copy shipped because nothing guarded the exhaustion path.
        copy = "The harbour is quiet at this hour and the boats are still."
        model = ScriptedModel([copy, copy, copy, copy])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("tell me about the harbour")
        self.assertEqual(comp.turn("what about the trains?"), REPEAT_FALLBACK)
        self.assertEqual([g["reason"] for g in comp.last_regens],
                         ["repeat", "repeat", "fallback"])

    def test_reused_sentence_is_stripped_after_retries(self):
        said = "The rain should ease off by morning and the wind will drop."
        fresh = "Bring a coat anyway, just in case it turns."
        model = ScriptedModel([said, said + " " + fresh, said + " " + fresh,
                               said + " " + fresh])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("weather?")
        self.assertEqual(comp.turn("and the weekend?"), fresh)
        self.assertEqual([g["reason"] for g in comp.last_regens],
                         ["repeat", "repeat", "strip"])

    def test_reply_of_only_reused_sentences_falls_back(self):
        a = "The rain should ease off by morning and the wind will drop."
        b = "Evenings have a quiet magic and the day's noise fades away."
        model = ScriptedModel([a, b, a + " " + b, a + " " + b, a + " " + b])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("weather?")
        comp.turn("evenings?")
        self.assertEqual(comp.turn("anything else?"), REPEAT_FALLBACK)
        self.assertEqual([g["reason"] for g in comp.last_regens],
                         ["repeat", "repeat", "fallback"])

    def test_second_retry_drops_the_copied_exchange(self):
        copy = "The harbour is quiet at this hour and the boats are still."
        model = ScriptedModel([copy, copy, copy, "Fresh answer in new words here."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("tell me about the harbour")
        self.assertEqual(comp.turn("what about the trains?"), "Fresh answer in new words here.")
        # Retry 2's window drops the copied exchange: the first question is gone, and
        # the only copy left is the rejected reply shown to the model for the rewrite.
        retry2 = [m["content"] for m in model.msgs[3]]
        self.assertNotIn("tell me about the harbour", retry2)
        self.assertEqual(retry2.count(copy), 1)
        self.assertIn("tell me about the harbour", [m["content"] for m in model.msgs[1]])


if __name__ == "__main__":
    unittest.main()
