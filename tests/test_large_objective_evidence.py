import tempfile
import unittest
from pathlib import Path

from tamfis_code import evidence
from tamfis_code.runner_local import (
    MAX_DIRECT_OBJECTIVE_CHARS,
    MAX_LOCAL_OBJECTIVE_CHARS,
    archive_oversized_objective,
)


class LargeObjectiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_evidence_dir = evidence.EVIDENCE_DIR
        evidence.EVIDENCE_DIR = Path(self.temp_dir.name) / "evidence"

    def tearDown(self):
        evidence.EVIDENCE_DIR = self.original_evidence_dir
        self.temp_dir.cleanup()

    def test_direct_size_objective_is_unchanged(self):
        messages = [{"role": "user", "content": "x" * MAX_DIRECT_OBJECTIVE_CHARS}]
        prepared, evidence_id = archive_oversized_objective(messages, session_id=41)

        self.assertIs(prepared, messages)
        self.assertIsNone(evidence_id)

    def test_oversized_objective_is_archived_and_prompt_is_bounded(self):
        middle_marker = "MIDDLE-MARKER-EXACT"
        objective = (
            "start instructions\n"
            + "a" * 1_100_000
            + middle_marker
            + "b" * 1_100_000
            + "\nend instructions"
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": objective},
        ]

        prepared, evidence_id = archive_oversized_objective(messages, session_id=42)

        self.assertIsNotNone(evidence_id)
        self.assertLess(len(prepared[-1]["content"]), 10_000)
        self.assertNotIn(middle_marker, prepared[-1]["content"])
        self.assertEqual(messages[-1]["content"], objective)
        segment = evidence.load_segment(42, evidence_id)
        self.assertEqual(segment["objective"], objective)

        found = evidence.objective_chunk(segment, query=middle_marker, max_chars=2_000)
        self.assertIn(middle_marker, found["objective"])
        self.assertEqual(found["objective_total_chars"], len(objective))
        self.assertGreater(found["objective_offset"], 0)

    def test_objective_retrieval_is_paged_and_clamped(self):
        objective = "0123456789" * 10_000
        evidence_id = evidence.store_segment(
            43, objective=objective, messages=[], summary="large objective",
        )
        segment = evidence.load_segment(43, evidence_id)

        first = evidence.objective_chunk(segment, offset=10, max_chars=25)
        self.assertEqual(first["objective"], objective[10:35])
        self.assertEqual(first["next_offset"], 35)
        self.assertTrue(first["has_more"])

        clamped = evidence.objective_chunk(segment, max_chars=10**9)
        self.assertEqual(len(clamped["objective"]), evidence.MAX_OBJECTIVE_CHUNK_CHARS)

    def test_more_than_five_million_characters_is_rejected(self):
        messages = [{"role": "user", "content": "x" * (MAX_LOCAL_OBJECTIVE_CHARS + 1)}]
        with self.assertRaisesRegex(ValueError, "5,000,000"):
            archive_oversized_objective(messages, session_id=44)


if __name__ == "__main__":
    unittest.main()
