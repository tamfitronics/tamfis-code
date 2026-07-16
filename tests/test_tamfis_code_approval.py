import unittest
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from tamfis_code.runner import resolve_approval_decision


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200)


class ResolveApprovalDecisionTests(unittest.TestCase):
    def test_full_auto_approves_regardless_of_risk(self):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "rm -rf build", "dangerous", "full-auto", interactive=False),
            "approve_once",
        )

    def test_workspace_approves_non_dangerous(self):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "workspace", interactive=False),
            "approve_once",
        )

    def test_workspace_denies_dangerous_when_non_interactive(self):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "rm -rf /", "dangerous", "workspace", interactive=False),
            "deny",
        )

    def test_suggest_always_denies(self):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "safe", "suggest", interactive=True),
            "deny",
        )

    def test_never_always_denies(self):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "never", interactive=True),
            "deny",
        )

    def test_ask_denies_when_non_interactive(self):
        # Regression guard: the CLI's one-shot commands (ask/audit/plan/
        # exec/run/retry) must never fall through to a blocking
        # console.input() prompt when there is no human able to answer it
        # -- this is exactly what caused a real ~10 minute hang before it
        # was fixed.
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "ask", interactive=False),
            "deny",
        )

    @patch.object(Console, "input", return_value="y")
    def test_ask_prompts_when_interactive_and_yes_approves(self, mock_input):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "ask", interactive=True),
            "approve_once",
        )
        mock_input.assert_called_once()

    @patch.object(Console, "input", return_value="a")
    def test_ask_prompt_always_returns_approve_session(self, _mock_input):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "ask", interactive=True),
            "approve_session",
        )

    @patch.object(Console, "input", return_value="n")
    def test_ask_prompt_no_denies(self, _mock_input):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "ask", interactive=True),
            "deny",
        )

    @patch.object(Console, "input", return_value="")
    def test_ask_prompt_empty_answer_denies(self, _mock_input):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "ask", interactive=True),
            "deny",
        )

    @patch.object(Console, "input", return_value="y")
    def test_workspace_dangerous_interactive_prompts_instead_of_auto_denying(self, mock_input):
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "rm -rf /", "dangerous", "workspace", interactive=True),
            "approve_once",
        )
        mock_input.assert_called_once()


if __name__ == "__main__":
    unittest.main()
