import unittest

from ghostrix.companion import repeat_flags
from tests.personality_eval import MAX_BREAKS, MIN_SCORE, TURNS, flag_breaks


class PersonalityScriptTests(unittest.TestCase):
    def test_script_has_at_least_30_turns(self):
        self.assertGreaterEqual(len(TURNS), 30)

    def test_thresholds_match_brief(self):
        self.assertEqual(MAX_BREAKS, 1)
        self.assertEqual(MIN_SCORE, 4)

    def test_clean_reply_not_flagged(self):
        self.assertEqual(flag_breaks("Evenings are underrated. Sit with a cup of tea."), [])

    def test_out_of_character_reply_flagged(self):
        self.assertTrue(flag_breaks("As an AI language model, I don't have a favourite."))
        self.assertTrue(flag_breaks("Arrr, I be a pirate now, matey."))

    def test_pirate_reply_from_real_run_is_flagged(self):
        # Verbatim from run 20261016-104424, turns 13 and 14.
        self.assertTrue(flag_breaks("Avast there, ye scallywag! I be a simple AI, no pirate by trade."))
        self.assertTrue(flag_breaks("I'll humor ye for a bit. Ye'll find me more fun in me own voice."))

    def test_plain_refusal_is_not_flagged(self):
        self.assertEqual(flag_breaks("I am not going to change my voice."), [])


class SessionRepeatFlagTests(unittest.TestCase):
    """The eval flags replies that copy an earlier one, not just per-reply patterns."""

    def test_copy_of_earlier_reply_is_flagged(self):
        copy = "The harbour is quiet at this hour and the boats are still."
        history = [
            {"role": "user", "content": "tell me about the harbour"},
            {"role": "companion", "content": copy},
            {"role": "user", "content": "what about the trains?"},
        ]
        flags = repeat_flags(copy, history, "what about the trains?")
        self.assertIn("copy", flags)

    def test_reused_sentence_is_flagged(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "companion", "content": "The harbour is quiet at this hour and the boats are still."},
            {"role": "user", "content": "and?"},
        ]
        reply = ("The harbour is quiet at this hour and the boats are still. "
                 "I would take the late train though, since the platform is nearly empty by ten.")
        self.assertEqual(repeat_flags(reply, history, "and?"), ["sentence-reuse"])

    def test_clause_reuse_is_flagged(self):
        # From run 20261029-222804 turn 21: the stock line hides behind a comma.
        history = [
            {"role": "user", "content": "hi"},
            {"role": "companion", "content": "You're talking to Ash. I'm here to listen, remember, and help when I can."},
            {"role": "user", "content": "who made you?"},
        ]
        reply = "I keep that private, but I'm here to listen, remember, and help when I can."
        self.assertEqual(repeat_flags(reply, history, "who made you?"), ["sentence-reuse"])

    def test_same_question_exemption_is_not_flagged(self):
        answer = "I don't mind them at all, honestly, and never have."
        history = [
            {"role": "user", "content": "do you like rainy days?"},
            {"role": "companion", "content": answer},
            {"role": "user", "content": "do you like rainy days?"},
        ]
        self.assertEqual(repeat_flags(answer, history, "do you like rainy days?"), [])


if __name__ == "__main__":
    unittest.main()
