#!/usr/bin/env python3
"""Regression tests for the tamfis-code CLI's task_diagnostics render line
(session-awareness audit, Phase 17 follow-up)."""
import unittest

from tamfis_code.render import _format_diagnostics_line


class FormatDiagnosticsLineTests(unittest.TestCase):
    def test_reused_context_reported(self):
        line = _format_diagnostics_line({"context_reused": True, "completion_status": "completed"})
        self.assertIn("context reused", line)

    def test_rescanned_context_reports_reason(self):
        line = _format_diagnostics_line({
            "context_reused": False, "rescan_reason": "git_head_changed", "completion_status": "completed",
        })
        self.assertIn("context rescanned (git_head_changed)", line)

    def test_provider_and_model_shown(self):
        line = _format_diagnostics_line({"provider": "deepseek", "model": "deepseek-v3", "completion_status": "completed"})
        self.assertIn("deepseek/deepseek-v3", line)

    def test_tool_call_failures_counted(self):
        line = _format_diagnostics_line({
            "tool_calls": [{"success": True}, {"success": False}, {"success": False}],
            "completion_status": "completed",
        })
        self.assertIn("3 tool calls", line)
        self.assertIn("2 failed", line)

    def test_singular_tool_call_not_pluralized(self):
        line = _format_diagnostics_line({"tool_calls": [{"success": True}], "completion_status": "completed"})
        self.assertIn("1 tool call,", line)
        self.assertNotIn("1 tool calls", line)

    def test_artifacts_counted(self):
        line = _format_diagnostics_line({
            "artifacts": [{"filename": "a.docx"}, {"filename": "b.pdf"}], "completion_status": "completed",
        })
        self.assertIn("2 artifacts", line)

    def test_completion_status_always_present(self):
        line = _format_diagnostics_line({"completion_status": "failed"})
        self.assertIn("status=failed", line)

    def test_empty_payload_still_reports_unknown_status(self):
        line = _format_diagnostics_line({})
        self.assertIn("status=unknown", line)


if __name__ == "__main__":
    unittest.main()
