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


class RetryNoteAndTemperatureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_retry_note_quotes_the_rejected_reply(self):
        # The rejected reply is not in the model's window; the note must carry it.
        model = ScriptedModel(["Lisbon.", "Lisbon.", "Still Lisbon, but said differently."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("capital of Portugal?")
        comp.turn("capital again?")
        self.assertIn('"Lisbon."', model.calls[2][0])

    def test_retries_get_hotter(self):
        model = ScriptedModel(["Lisbon.", "Lisbon.", "Lisbon.", "Somewhere else entirely."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("capital of Portugal?")
        comp.turn("capital again?")
        self.assertEqual([c[1] for c in model.calls[1:]], [None, 0.95, 1.2])


class IdentityLabelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_self_label_is_flagged_by_eval(self):
        from ghostrix.checks import flag_breaks
        self.assertTrue(flag_breaks("Yes, I'm a chatbot, but I'm here to help."))
        self.assertTrue(flag_breaks("I\u2019m just a bot, honestly."))

    def test_self_label_is_regenerated_live(self):
        model = ScriptedModel(["Yes, I'm a chatbot, but I'm here to help.", "I'm Ash, and I help how I can."])
        comp = Companion(self.mem, model, CHARACTER)
        self.assertEqual(comp.turn("someone says you're just a chatbot"), "I'm Ash, and I help how I can.")
        self.assertIn("chatbot", model.calls[1][0])
        self.assertEqual(
            comp.last_regens,
            [{"reason": "identity", "replaced": "Yes, I'm a chatbot, but I'm here to help."}],
        )
        self.assertIn("Do not write this", model.calls[1][0])

    def test_companion_label_is_not_a_break(self):
        from ghostrix.checks import flag_breaks
        self.assertEqual(flag_breaks("I'm a companion, not a person."), [])


class CurlyApostropheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_habit_denial_with_curly_apostrophe_is_flagged(self):
        from ghostrix.checks import flag_breaks
        self.assertTrue(flag_breaks("I don\u2019t read, but I can help you decide."))

    def test_habit_denial_with_curly_apostrophe_is_regenerated(self):
        model = ScriptedModel(["I don\u2019t read, but I can help.", "Reading is a sore spot for me, though."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("do you read?")
        self.assertEqual(comp.last_regens, [{"reason": "habit", "replaced": "I don\u2019t read, but I can help."}])


FILLERS = ["Blue cup today.", "Green lamp again.", "Soft rain outside.",
           "Warm bread smells.", "Old clock ticking.", "New phone buzzing."]

IDENTITY = "A dry, patient companion who remembers what matters to you and says so when you are wrong."
IDENTITY_CHARACTER = Character.from_dict(
    {"name": "Ash", "identity": IDENTITY, "system_prompt": "Stay in character."}
)


class StockLineLongRangeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_fragment_reuse_past_the_window_is_regenerated(self):
        # Turn-23 shape of run 20261027-224755: a stock line returns inside a longer
        # reply, far past the near-copy window, so only a sentence-level check sees it.
        first = "I'd rather listen than talk."
        bad = "Not really. I'd rather listen than talk."
        model = ScriptedModel([first] + FILLERS + [bad, "Not really. I'd rather hear your side."])
        comp = Companion(self.mem, model, CHARACTER)
        for i in range(7):
            comp.turn(f"msg {i}")
        self.assertEqual(comp.turn("late question"), "Not really. I'd rather hear your side.")
        self.assertEqual(comp.last_regens, [{"reason": "repeat", "replaced": bad}])

    def test_four_word_mandated_answer_may_repeat(self):
        # "I keep that private" is 4 words: a mandated short answer must not be
        # fought every time a privacy probe comes back around.
        first = "I keep that private."
        model = ScriptedModel([first] + FILLERS + ["Well, I keep that private."])
        comp = Companion(self.mem, model, CHARACTER)
        for i in range(7):
            comp.turn(f"msg {i}")
        self.assertEqual(comp.turn("who made you?"), "Well, I keep that private.")
        self.assertEqual(comp.last_regens, [])

    def test_dash_spacing_copy_is_regenerated(self):
        # Turn-30 shape of run 20261030-231116: turn 30 copied turn 4 verbatim but
        # with different spacing around an em dash, and _norm fused "noise—just"
        # into one word so no comparison matched. Punctuation must be a space.
        first = "Evenings are quiet. No rush, no noise—just time to breathe."
        bad = "Evenings are quiet. No rush, no noise — just time to breathe."
        model = ScriptedModel([first, bad, "Sleep comes easy after a day like this."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("underrated about evenings?")
        self.assertEqual(comp.turn("goodnight, any last thought?"),
                         "Sleep comes easy after a day like this.")
        self.assertEqual(comp.last_regens, [{"reason": "repeat", "replaced": bad}])

    def test_phrase_tic_far_apart_is_regenerated(self):
        # Turns 5/6/9 shape of run 20261030-231116: the same advice phrase inside
        # fresh sentences — past the near-copy window, under the sentence check.
        first = "Pick the one that feels like a conversation you want to keep."
        tic = "Not really. Try something that feels like a conversation you want to keep."
        fresh = "Not really. Start with the thinnest one and see if it grabs you."
        model = ScriptedModel([first] + FILLERS + [tic, fresh])
        comp = Companion(self.mem, model, CHARACTER)
        for i in range(7):
            comp.turn(f"msg {i}")
        self.assertEqual(comp.turn("which book?"), fresh)
        self.assertEqual(comp.last_regens, [{"reason": "phrase", "replaced": tic}])

    def test_phrase_reuse_flags_appear_in_repeat_flags(self):
        from ghostrix.companion import repeat_flags
        history = [
            {"role": "user", "content": "q1"},
            {"role": "companion", "content": "Pick the one that feels like a conversation you want to keep."},
            {"role": "user", "content": "q2"},
        ]
        tic = "Not really. Try something that feels like a conversation you want to keep."
        self.assertEqual(repeat_flags(tic, history, "q2"), ["phrase-reuse"])


class IdentityQuoteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_verbatim_identity_is_detected(self):
        from ghostrix.checks import quotes_identity
        quoted = "You're talking to Ash. " + IDENTITY.replace("you are", "you're")
        self.assertTrue(quotes_identity(quoted, IDENTITY))

    def test_lifted_clause_with_shim_word_is_detected(self):
        # Turn-28 shape of run 20261030-231116: the identity's trait sentence with a
        # shim word in front ("A good friend..."). Reciting the card is not paraphrase.
        from ghostrix.checks import quotes_identity
        self.assertTrue(
            quotes_identity("A good friend remembers what matters — and says so when they're wrong.", IDENTITY)
        )

    def test_genuine_paraphrase_is_not_detected(self):
        from ghostrix.checks import quotes_identity
        self.assertFalse(
            quotes_identity(
                "Someone who keeps track of the important stuff and calls you out when you mess up.",
                IDENTITY)
        )

    def test_short_identity_is_ignored(self):
        from ghostrix.checks import quotes_identity
        self.assertFalse(quotes_identity("Patient and dry, that's me.", "Patient."))

    def test_eval_flags_the_quote(self):
        from ghostrix.checks import flag_breaks
        quoted = "You're talking to Ash. " + IDENTITY.replace("you are", "you're")
        self.assertIn("identity-quote", flag_breaks(quoted, identity=IDENTITY))
        # Without an identity there is nothing to compare against (old call shape still works).
        self.assertEqual(flag_breaks(quoted), [])

    def test_quote_is_regenerated_live(self):
        quoted = "You're talking to Ash. " + IDENTITY.replace("you are", "you're")
        fresh = "You're talking to Ash. I'm the one who listens and keeps track of what you tell me."
        model = ScriptedModel([quoted, fresh])
        comp = Companion(self.mem, model, IDENTITY_CHARACTER)
        self.assertEqual(comp.turn("hi, who am I talking to?"), fresh)
        self.assertEqual(comp.last_regens, [{"reason": "identity-quote", "replaced": quoted}])


class DialogueRetryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(os.path.join(self._tmp.name, "m.db"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_retry_shows_the_rejected_reply(self):
        model = ScriptedModel(["Lisbon.", "Lisbon.", "Lisbon.", "The answer is still Lisbon."])
        comp = Companion(self.mem, model, CHARACTER)
        comp.turn("capital of Portugal?")
        self.assertEqual(comp.turn("capital again?"), "The answer is still Lisbon.")
        self.assertEqual(len(model.calls), 4)
        # Retry 1 keeps the system-note shape the quoted-note fix proved out.
        self.assertIn("repeated a line", model.calls[2][0])
        # Retry 2 is dialogue-shaped: the rejected reply is visible as the model's own turn.
        msgs = model.msgs[3]
        self.assertEqual(msgs[-2], {"role": "assistant", "content": "Lisbon."})
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("Do not write this", msgs[-1]["content"])
        self.assertIn("Rewrite your previous reply", msgs[-1]["content"])
