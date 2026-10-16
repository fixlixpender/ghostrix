import unittest

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


if __name__ == "__main__":
    unittest.main()
