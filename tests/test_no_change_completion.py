"""A validation pass that finds nothing to fix is a COMPLETED task, not a failed one.

Owner report 2026-09-21: `bash -n` over every script exited 0 and the report said "Changes: None required ...
Remaining issues: None", yet the run was failed with "The request required a code change, but no successful file
mutation was recorded" -- and the next-message box then pre-filled that verdict as a task to "resolve".
"""
import unittest

from tamfis_code.orchestrator.validator import validate_completion, verified_no_change_completion
from tamfis_code.routing import classify_task

REPORT = """Summary

 • Validation command `find . -name '*.sh' | xargs -n1 bash -n` now returns exit code 0 with no output.
 • 20 shell scripts were found under ./scripts/ and all passed bash -n syntax check.

Changes

 • None required; the shell scripts are syntactically valid.

Verification

 • bash -n syntax check: ✅ (exit 0)

Remaining issues

 • None."""

GREEN = {"tool_name": "execute_command", "success": True, "exit_code": 0, "arguments": {"command": "bash -n a.sh"}}


class TerseNoChangeReportTests(unittest.TestCase):
    def test_read_only_recovery_wrapper_does_not_require_a_mutation(self):
        # The UI's machine-generated "Repair the failed plan step" wrapper
        # contains mutation language, while the durable step can still be a
        # pure inspection. Effective runtime mode must win over wrapper text.
        report = validate_completion(
            profile=classify_task("Repair the failed plan step, then revalidate it: Read pyproject.toml"),
            tool_records=[{"tool_name": "read_file", "success": True, "arguments": {"path": "pyproject.toml"}}],
            any_mutation=False,
            final_text="Summary\n- Read pyproject.toml.\n\nChanges\n- None; this was inspection only.",
            read_only=True,
        )
        self.assertTrue(report.passed, report.unresolved)

    def test_a_green_check_plus_a_none_required_report_is_a_verified_no_op(self):
        self.assertTrue(verified_no_change_completion(tool_records=[GREEN], final_text=REPORT))

    def test_the_whole_gate_now_passes_for_a_fix_request_with_nothing_to_fix(self):
        report = validate_completion(
            profile=classify_task("fix the syntax errors in the shell scripts"),
            tool_records=[GREEN], any_mutation=False, final_text=REPORT,
        )
        self.assertTrue(report.passed, report.unresolved)

    def test_inline_and_terse_phrasings(self):
        for text in ("Changes: none.", "Nothing to fix here; all scripts pass.", "No edits were needed.",
                     "None needed -- the config is already valid."):
            self.assertTrue(verified_no_change_completion(tool_records=[GREEN], final_text=text), text)

    def test_without_a_command_that_really_ran_green_it_is_not_accepted(self):
        for records in ([], [{"tool_name": "read_file", "success": True}], [{**GREEN, "success": False}],
                        [{**GREEN, "exit_code": 2}]):
            self.assertFalse(verified_no_change_completion(tool_records=records, final_text=REPORT), records)

    def test_a_report_that_also_claims_an_edit_is_contradictory_and_rejected(self):
        self.assertFalse(verified_no_change_completion(
            tool_records=[GREEN], final_text=REPORT + "\nI updated the file scripts/deploy.sh to fix it."))

    def test_an_attempted_edit_disqualifies_it_even_if_it_failed(self):
        self.assertFalse(verified_no_change_completion(
            tool_records=[{"tool_name": "edit_file", "success": False}, GREEN], final_text=REPORT))

    def test_a_change_request_that_did_nothing_and_says_nothing_useful_still_fails(self):
        report = validate_completion(
            profile=classify_task("add a login page to the app"),
            tool_records=[{"tool_name": "read_file", "success": True}],
            any_mutation=False, final_text="I looked at the project structure.",
        )
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
