"""Checks for the real-model evaluation's cross-mixing detector. No model needed.

Cases come from the first real run (20261013-100747). The Nancy reply is the
known bad one: it carried the birthday date into an answer about school.
"""

import unittest

from tests.memory_eval import FACTS, cross_mixed_with, contains_answer


def index_of(question):
    return next(i for i, (_s, q, _a) in enumerate(FACTS) if q == question)


class CrossMixingTests(unittest.TestCase):
    def test_nancy_reply_with_birthday_date_is_flagged(self):
        i = index_of("When does Nancy start school?")
        self.assertEqual(cross_mixed_with(i, "September the twenty-seventh."), "twenty")

    def test_clean_replies_are_not_flagged(self):
        clean = {
            "What is my dog called?": "Biscuit.",
            "When does Phoebe turn five?": "March.",
            "What colour is the Volvo car?": "Green.",
            "What is my favourite coffee?": "A double espresso.",
            "How old is Nancy?": "Three.",
            "How many WordPress shops do I manage?": "Four.",
        }
        for question, reply in clean.items():
            with self.subTest(question=question):
                self.assertIsNone(cross_mixed_with(index_of(question), reply))

    def test_word_already_in_own_statement_is_not_flagged(self):
        # "The spare key is under the blue pot." - "blue" is part of its own fact.
        i = index_of("Where is the spare key hidden?")
        self.assertIsNone(cross_mixed_with(i, "Under the blue pot."))

    def test_birthday_reply_mentioning_september_is_not_flagged(self):
        # "September" is Nancy's answer, but it is in the birthday fact's own statement.
        i = index_of("When is my birthday?")
        self.assertIsNone(cross_mixed_with(i, "September the twenty-seventh."))

    def test_answer_still_detected_for_known_forms(self):
        self.assertTrue(contains_answer("Four.", "four"))
        self.assertTrue(contains_answer("About 27 September.", "twenty"))


if __name__ == "__main__":
    unittest.main()
