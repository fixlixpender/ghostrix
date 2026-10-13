"""Memory core tests. Stdlib only: run with `python3 -m unittest discover -s tests -v`.

Note on the 20-fact recall test: it runs against the HashEmbedder stub. It
verifies the plumbing (facts are stored, summaries carry them, retrieval finds
them). It is NOT the real memory test from the brief, which must run with the
local model on the user's machine.
"""

import os
import tempfile
import unittest

from ghostrix.memory import Memory

# 20 facts, grouped into 5 sessions of 4. Each: (statement, question, answer).
FACTS = [
    ("My dog is called Biscuit.", "What is my dog called?", "Biscuit"),
    ("My daughter Phoebe turns five in March.", "When does Phoebe turn five?", "March"),
    ("I work in marketing at Rockwell.", "Where do I work in marketing?", "Rockwell"),
    ("The car is a green Volvo estate.", "What colour is the Volvo car?", "green"),
    ("My favourite coffee is a double espresso.", "What is my favourite coffee?", "espresso"),
    ("The band rehearses on Thursday nights.", "On which night does the band rehearse?", "Thursday"),
    ("Nancy starts school in September.", "When does Nancy start school?", "September"),
    ("I manage four WordPress shops.", "How many WordPress shops do I manage?", "four"),
    ("My guitar is a Gibson Les Paul.", "Which guitar do I play?", "Gibson"),
    ("Rebecca teaches English at a secondary school.", "What does Rebecca teach?", "English"),
    ("The flat has a blue front door.", "What colour is the front door?", "blue"),
    ("Madalena lives with her mum in Porto.", "Where does Madalena live?", "Porto"),
    ("The project codename is Lighthouse.", "What is the project codename?", "Lighthouse"),
    ("My birthday is on the twenty seventh of September.", "When is my birthday?", "twenty"),
    ("The office kettle broke in June.", "When did the office kettle break?", "June"),
    ("Our router is a Netgear model.", "Which brand is the router?", "Netgear"),
    ("I quit smoking in April.", "When did I quit smoking?", "April"),
    ("The cover band plays Nirvana songs.", "Which band songs does the cover band play?", "Nirvana"),
    ("The spare key is under the blue pot.", "Where is the spare key hidden?", "pot"),
    ("Phoebe is four and Nancy is three.", "How old is Nancy?", "three"),
]


class _Fixture(unittest.TestCase):
    """Shared setup. Holds no tests, so subclasses don't inherit duplicates."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "memory.db")
        self.mem = Memory(self.path)

    def tearDown(self):
        self.mem.close()
        self._tmp.cleanup()

    def _session(self, statements):
        sid = self.mem.start_session()
        ids = [self.mem.log(sid, "user", s) for s in statements]
        return sid, ids

    def _store_batches(self):
        for start in range(0, 20, 4):
            batch = FACTS[start:start + 4]
            sid, ids = self._session([statement for statement, _, _ in batch])
            # Stand-in for the model's summary: the summary carries the facts.
            self.mem.end_session(sid, " ".join(s for s, _, _ in batch), ids)


class StoreTests(_Fixture):
    def test_log_keeps_order_and_roles(self):
        sid, _ = self._session(["hello", "hi there"])
        self.mem.log(sid, "companion", "what's up")
        rows = self.mem.messages(sid)
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3])
        self.assertEqual([r["role"] for r in rows], ["user", "user", "companion"])

    def test_bad_role_rejected(self):
        sid = self.mem.start_session()
        with self.assertRaises(ValueError):
            self.mem.log(sid, "system", "nope")

    def test_summary_requires_source_links(self):
        sid, _ = self._session(["a fact"])
        with self.assertRaises(ValueError):
            self.mem.end_session(sid, "summary", [])

    def test_summary_sources_must_belong_to_session(self):
        _, ids1 = self._session(["fact in session one"])
        sid2 = self.mem.start_session()
        with self.assertRaises(ValueError):
            self.mem.end_session(sid2, "summary", ids1)

    def test_summary_links_back_to_sources(self):
        sid, ids = self._session(["first fact", "second fact"])
        summary_id = self.mem.end_session(sid, "Two facts were shared.", ids)
        self.assertEqual(self.mem.sources_of(summary_id), tuple(sorted(ids)))

    def test_ended_session_rejects_new_messages(self):
        sid, ids = self._session(["a fact"])
        self.mem.end_session(sid, "a summary", ids)
        with self.assertRaises(ValueError):
            self.mem.log(sid, "user", "too late")

    def test_profile_is_proposed_then_decided(self):
        pid = self.mem.propose_profile("Likes espresso.")
        self.assertEqual(self.mem.stable_profile(), [])
        self.mem.decide_profile(pid, accept=True)
        self.assertEqual(self.mem.stable_profile(), ["Likes espresso."])

    def test_rejected_profile_entry_never_appears(self):
        pid = self.mem.propose_profile("Is a secret agent.")
        self.mem.decide_profile(pid, accept=False)
        self.assertEqual(self.mem.stable_profile(), [])
        with self.assertRaises(ValueError):
            self.mem.decide_profile(pid, accept=True)

    def test_retrieval_returns_summary_with_sources(self):
        sid, ids = self._session(["My dog is called Biscuit."])
        summary_id = self.mem.end_session(sid, "The user has a dog called Biscuit.", ids)
        hits = self.mem.retrieve("What is my dog called?", k=5)
        summaries = [h for h in hits if h.kind == "summary"]
        self.assertTrue(summaries, "summary missing from top-5")
        self.assertEqual(summaries[0].id, summary_id)
        self.assertEqual(summaries[0].source_message_ids, tuple(ids))


class TwentyFactRecallTest(_Fixture):
    """Stub-level check: 20 facts over 5 sessions, then questions afterwards.

    Pass threshold from the brief: at least 17 of 20 recalled, zero wrong.
    Here "recalled" means a top-5 hit contains the answer word.
    """

    def test_twenty_facts_across_five_sessions(self):
        self._store_batches()
        recalled = 0
        for _statement, question, answer in FACTS:
            hits = self.mem.retrieve(question, k=5)
            if any(answer.lower() in h.text.lower() for h in hits):
                recalled += 1
        self.assertGreaterEqual(recalled, 17, f"recalled {recalled}/20 with stub embedder")

    def test_memory_survives_reopen(self):
        """Next-day analogue: close the store, reopen the file, recall still works."""
        self._store_batches()
        self.mem.close()
        self.mem = Memory(self.path)
        top = self.mem.retrieve("When does Phoebe turn five?", k=5)
        self.assertTrue(any("March" in h.text for h in top))


if __name__ == "__main__":
    unittest.main()
