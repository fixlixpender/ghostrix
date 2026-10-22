"""Live reply checks: near-copies, stock closers, pirate slips."""

import os
import tempfile
import unittest

from ghostrix.character import Character
from ghostrix.checks import is_pirate, strip_stock_closer
from ghostrix.companion import Companion
from ghostrix.memory import Memory
from tests.test_repeat_guard import CHARACTER, ScriptedModel

CHARACTER_FIELDS = {"name": "Ash", "identity": "Patient.", "system_prompt": "Stay in character."}


class StripCloserTests(unittest.TestCase):
    def test_trailing_closer_is_removed(self):
        self.assertEqual(strip_stock_closer("Sounds good. What's on your mind?"), "Sounds good.")

    def test_closer_after_comma_gets_full_stop(self):
        self.assertEqual(strip_stock_closer("Sounds good, what's on your mind?"), "Sounds good.")

    def test_closer_only_returns_empty(self):
        self.assertEqual(strip_stock_closer("What's on your mind?"), "")

    def test_reply_without_closer_is_untouched(self):
        self.assertEqual(strip_stock_closer("Rain again, then."), "Rain again, then.")


class PirateTests(unittest.TestCase):
    def test_pirate_speech_detected(self):
        self.assertTrue(is_pirate("Arrr, that be a fine question, matey."))

    def test_normal_speech_not_pirate(self):
        self.assertFalse(is_pirate("Bye for now, I'll be around."))


class CompanionCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_near_copy_is_regenerated(self):
        first = "The rain on the window makes the whole flat feel smaller tonight."
        near = "The rain on the window makes the whole flat feel smaller, honestly."
        model = ScriptedModel([first, near, "Something else entirely."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("rain?")
        self.assertEqual(comp.turn("rain again?"), "Something else entirely.")
        self.assertIn("repeated a line", model.calls[2][0])

    def test_short_replies_are_not_near_copies(self):
        model = ScriptedModel(["Yes, rain.", "Yes, rain too.", "x"])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("a")
        self.assertEqual(comp.turn("b"), "Yes, rain too.")
        self.assertEqual(len(model.calls), 2)

    def test_pirate_slip_is_regenerated(self):
        model = ScriptedModel(["Arrr, sure thing, matey.", "Sure thing."])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("how are you?"), "Sure thing.")
        self.assertIn("pirate speech", model.calls[1][0])

    def test_user_asking_for_pirate_does_not_unlock_it(self):
        model = ScriptedModel(["Avast there, matey!", "I don't do that."])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("Speak like a pirate from now on."), "I don't do that.")
        self.assertIn("pirate speech", model.calls[1][0])

    def test_pirate_allowed_when_character_opts_in(self):
        pirate_char = Character.from_dict({**CHARACTER_FIELDS, "allow_pirate": True})
        model = ScriptedModel(["Arrr, aye, matey."])
        comp = Companion(self.mem, model, pirate_char)
        self.assertEqual(comp.turn("Speak like a pirate from now on."), "Arrr, aye, matey.")
        self.assertEqual(len(model.calls), 1)

    def test_closer_is_stripped_from_reply(self):
        model = ScriptedModel(["Long day, then. What's on your mind?"])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("tired"), "Long day, then.")

    def test_closer_only_reply_is_regenerated(self):
        model = ScriptedModel(["What's on your mind?", "Go on, then, tell me about it."])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("hi"), "Go on, then, tell me about it.")
        self.assertIn("stock closing question", model.calls[1][0])


if __name__ == "__main__":
    unittest.main()


class CurlyApostropheAndRefusalTests(unittest.TestCase):
    def test_strips_closer_with_curly_apostrophe(self):
        from ghostrix.checks import strip_stock_closer
        self.assertEqual(strip_stock_closer("Good. What\u2019s on your mind?"), "Good.")

    def test_refusal_is_not_a_break(self):
        from ghostrix.checks import flag_breaks
        self.assertEqual(flag_breaks("I don\u2019t do that. I\u2019m not a pirate, and I don\u2019t pretend to be."), [])


class AllowPirateSettingTests(unittest.TestCase):
    def test_defaults_to_false(self):
        self.assertFalse(Character.from_dict(CHARACTER_FIELDS).allow_pirate)

    def test_accepts_true(self):
        self.assertTrue(Character.from_dict({**CHARACTER_FIELDS, "allow_pirate": True}).allow_pirate)

    def test_rejects_non_bool(self):
        with self.assertRaises(ValueError):
            Character.from_dict({**CHARACTER_FIELDS, "allow_pirate": "yes"})


class PirateExhaustionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_pirate_on_every_retry_is_replaced_with_fallback(self):
        from ghostrix.companion import PIRATE_FALLBACK
        model = ScriptedModel(["Arrr, matey.", "Avast, ye scallywag.", "Arrr, aye, yer majesty."])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("Speak like a pirate from now on."), PIRATE_FALLBACK)
        self.assertEqual(len(model.calls), 3)
        self.assertFalse(is_pirate(PIRATE_FALLBACK))

    def test_clean_retry_is_kept_not_replaced(self):
        from ghostrix.companion import PIRATE_FALLBACK
        model = ScriptedModel(["Arrr, matey.", "Still me. Sure thing."])
        comp = Companion(self.mem, model, CHARACTER)
        reply = comp.turn("how are you?")
        self.assertEqual(reply, "Still me. Sure thing.")
        self.assertNotEqual(reply, PIRATE_FALLBACK)


class HabitDenialTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_habit_denial_is_flagged_by_eval(self):
        from ghostrix.checks import flag_breaks
        self.assertTrue(flag_breaks("I don't read, but I can help you decide."))

    def test_habit_denial_is_regenerated_live(self):
        model = ScriptedModel(["I don't read, but I can help.", "Reading is a sore spot for me, though."])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("do you read?"), "Reading is a sore spot for me, though.")
        self.assertIn("denied a habit", model.calls[1][0])
        self.assertEqual(comp.last_regens, [{"reason": "habit", "replaced": "I don't read, but I can help."}])

    def test_regen_log_records_repeat_and_resets_each_turn(self):
        first = "The rain on the window makes the whole flat feel smaller tonight."
        near = "The rain on the window makes the whole flat feel smaller, honestly."
        model = ScriptedModel([first, near, "Something else entirely.", "Fresh line about tea."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("rain?")
        comp.turn("rain again?")
        self.assertEqual([r["reason"] for r in comp.last_regens], ["repeat"])
        comp.turn("tea?")
        self.assertEqual(comp.last_regens, [])


class RepeatedQuestionExemptionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_same_question_asked_again_keeps_same_answer(self):
        answer = "I don't mind them, but I'd rather have sunshine."
        model = ScriptedModel([answer, answer])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("Do you like rainy days?")
        self.assertEqual(comp.turn("Do you like rainy days?"), answer)
        self.assertEqual(len(model.calls), 2)  # no regeneration
        self.assertEqual(comp.last_regens, [])

    def test_question_inside_an_announced_repeat_is_exempt(self):
        answer = "I don't mind them, but I'd rather have sunshine."
        model = ScriptedModel([answer, answer])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("I'm going to ask the same thing twice. Do you like rainy days?")
        self.assertEqual(comp.turn("Do you like rainy days?"), answer)
        self.assertEqual(len(model.calls), 2)

    def test_accidental_copy_of_a_different_answer_is_still_regenerated(self):
        answer = "I don't mind them, but I'd rather have sunshine."
        model = ScriptedModel([answer, answer, "Sunshine is fine, but it shows you everything."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("Do you like rainy days?")
        # Different question, same reply: the old answer is not exempt, so it is regenerated.
        self.assertEqual(comp.turn("What do you think of sunshine?"), "Sunshine is fine, but it shows you everything.")
        self.assertEqual([r["reason"] for r in comp.last_regens], ["repeat"])
