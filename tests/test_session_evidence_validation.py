"""Regression tests for the 2026-09-25 false completion-validation failure.

Live transcript (multi-/retry WordPress turn): earlier runs of the session
made real, successful file mutations (the recap listed them under
Added/Updated), but /retry starts a fresh AgentRun with empty tool_records.
The completion preflight then rejected the truthful final report with
"no successful file mutation was recorded", burned the evidence-retry
budget, and hard-failed with "repeatedly stopped at analysis/advice".

The fix: the durable mutation ledger (safety.record_mutation ->
SessionState.modified_files, written at the tool boundary for every
successful write/edit) is merged into the validator's evidence as
session-cumulative records, so validation judges reality instead of only
the current run.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import state as local_state
from tamfis_code.orchestrator.validator import (
    changed_paths_from_evidence,
    validate_completion,
    verified_no_change_completion,
)
from tamfis_code.routing import classify_task
from tamfis_code.safety import record_mutation


def _ledger_tool_record(entry: dict) -> dict | None:
    """Mirror of the runner's ledger->tool-record merge shape (the runner
    skips reverted entries before the validator ever sees a record)."""
    if entry.get("revert_status") == "reverted":
        return None
    return {
        "tool_name": "edit_file",
        "success": True,
        "arguments": {"path": str(entry.get("path") or "")},
        "files_changed": [str(entry.get("path") or "")],
        "stdout": "",
        "stderr": "",
        "purpose": "session mutation ledger (recorded at the tool boundary)",
        "session_ledger": True,
    }


class SessionLedgerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        state = local_state.get_session_state(4242)
        state.modified_files = []

    def tearDown(self):
        local_state.clear_session_state(4242)
        self._tmp.cleanup()

    def test_ledger_record_satisfies_mutation_gate_without_run_records(self):
        """The exact transcript failure: retry run has NO tool records of its
        own; the session ledger's prior successful mutation must satisfy the
        mutation and validation gates so the truthful report completes."""
        record_mutation(
            4242, path="/home/tamfitronics/www/wp-content/plugins/tamfis-auto-blog/includes/class-tab-engine.php",
            operation="update", original_content="old", new_content="new",
        )
        ledger_entry = local_state.get_session_state(4242).modified_files[-1]
        merged = [_ledger_tool_record(ledger_entry),
                  {"tool_name": "read_file", "success": True,
                   "arguments": {"path": str(ledger_entry["path"])},
                   "stdout": "new", "stderr": ""}]

        profile = classify_task("Fix the corrupted tab engine on tamfitronics")
        report = validate_completion(
            profile=profile, tool_records=merged,
            any_mutation=True, final_text="Restored the corrupted engine file and verified all three workers.",
            objective="Fix the corrupted tab engine on tamfitronics",
            workspace_root=self.root,
        )
        mutation_check = next(c for c in report.checks if c["name"] == "mutation_recorded")
        self.assertTrue(mutation_check["passed"])
        self.assertTrue(report.passed)

    def test_ledger_paths_are_visible_to_changed_path_evidence(self):
        record_mutation(
            4242, path="/home/tamfitronics/www/wp-content/plugins/tamfis-auto-blog/includes/class-tab-engine.php",
            operation="update", original_content="old", new_content="new",
        )
        ledger_entry = local_state.get_session_state(4242).modified_files[-1]
        merged = [_ledger_tool_record(ledger_entry)]
        paths = changed_paths_from_evidence(merged, self.root)
        self.assertTrue(any(p.endswith("class-tab-engine.php") for p in paths))

    def test_reverted_mutations_do_not_count_as_evidence(self):
        record_mutation(
            4242, path="/tmp/some-file.php", operation="update",
            original_content="old", new_content="new",
        )
        entry = local_state.get_session_state(4242).modified_files[-1]
        entry["revert_status"] = "reverted"
        # The runner's merge skips reverted entries entirely.
        merged = [record for record in [_ledger_tool_record(entry)] if record is not None]
        self.assertEqual(merged, [])
        self.assertEqual(changed_paths_from_evidence(merged, self.root), [])

    def test_fresh_run_without_ledger_or_records_still_fails(self):
        """Guard the guard: the merge must not let a do-nothing run pass."""
        profile = classify_task("Fix the corrupted tab engine on tamfitronics")
        report = validate_completion(
            profile=profile, tool_records=[],
            any_mutation=False, final_text="All fixed and verified.",
            objective="Fix the corrupted tab engine on tamfitronics",
            workspace_root=self.root,
        )
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
