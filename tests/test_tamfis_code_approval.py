import unittest
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from tamfis_code.config import Config
from tamfis_code.runner import _MODE_SWITCH_SENTINEL, _prompt, resolve_approval_decision


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


class LiveConfigApprovalDecisionTests(unittest.TestCase):
    """Coverage for the interactive-REPL, Shift+Tab-at-the-approval-gate
    mode switch: a live Config object, when passed, wins over the
    (possibly stale) approval_policy snapshot -- since a mode switch made
    at the prompt below needs to affect the CURRENT decision, not just
    future ones. Not a substitute for live keystroke verification (no pty
    here), but guards the actual decision logic against regressing."""

    def test_live_config_policy_overrides_stale_snapshot_string(self):
        console = _console()
        config = Config(approval_policy="auto")
        self.assertEqual(
            resolve_approval_decision(
                console, "npm install", "medium",
                # Stale snapshot says "ask" -- would normally block on
                # console.input() and hang with no client attached. The
                # live config's "auto" must win instead.
                "ask", interactive=True, config=config,
            ),
            "approve_once",
        )

    def test_no_config_falls_back_to_snapshot_unchanged(self):
        # config=None (the default for every non-interactive/one-shot
        # caller) must behave exactly as before this feature existed.
        console = _console()
        self.assertEqual(
            resolve_approval_decision(console, "npm install", "medium", "full-auto", interactive=False, config=None),
            "approve_once",
        )

    @patch("prompt_toolkit.PromptSession.prompt")
    def test_shift_tab_sentinel_switches_mode_and_resolves_without_a_second_answer(self, mock_prompt):
        # Simulates what the s-tab key binding itself does (mutate
        # config.approval_policy, then hand back the sentinel) without
        # needing a real pty/keystroke -- the part under test is _prompt's
        # handling of that sentinel, not prompt_toolkit's own key dispatch.
        console = _console()
        config = Config(approval_policy="ask")

        def _simulate_shift_tab(*_args, **_kwargs):
            config.approval_policy = "auto"
            return _MODE_SWITCH_SENTINEL

        mock_prompt.side_effect = _simulate_shift_tab
        decision = _prompt(console, "npm install", "medium", display_preview=False, config=config)
        self.assertEqual(decision, "approve_once")
        mock_prompt.assert_called_once()

    @patch("prompt_toolkit.PromptSession.prompt")
    def test_shift_tab_into_another_interactive_mode_keeps_prompting(self, mock_prompt):
        # Cycling into "plan-only" (config.py's raw value for the "plan"
        # label, still interactive-adjacent but read-only) must not
        # fabricate a decision by itself here -- plan-only actually
        # auto-denies (see _decision_for_policy), so use a policy that
        # stays genuinely undecided: "ask" itself, reached by cycling all
        # the way around MODE_CYCLE.
        console = _console()
        config = Config(approval_policy="accept-edits")
        calls = {"n": 0}

        def _side_effect(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                config.approval_policy = "ask"
                return _MODE_SWITCH_SENTINEL
            return "y"

        mock_prompt.side_effect = _side_effect
        decision = _prompt(console, "npm install", "medium", display_preview=False, config=config)
        self.assertEqual(decision, "approve_once")
        self.assertEqual(mock_prompt.call_count, 2)


if __name__ == "__main__":
    unittest.main()


class AsyncApprovalDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_config_uses_prompt_async_inside_running_loop(self):
        console = _console()
        config = Config(approval_policy="ask")
        with patch("prompt_toolkit.PromptSession.prompt_async", return_value="y") as prompt_async:
            from tamfis_code.runner import resolve_approval_decision_async

            decision = await resolve_approval_decision_async(
                console,
                "npm install",
                "medium",
                "ask",
                interactive=True,
                config=config,
            )

        self.assertEqual(decision, "approve_once")
        prompt_async.assert_awaited_once()

    async def test_no_config_console_input_runs_without_nested_event_loop(self):
        console = _console()
        with patch.object(Console, "input", return_value="n") as mock_input:
            from tamfis_code.runner import resolve_approval_decision_async

            decision = await resolve_approval_decision_async(
                console,
                "npm install",
                "medium",
                "ask",
                interactive=True,
            )

        self.assertEqual(decision, "deny")
        mock_input.assert_called_once()

    async def test_auto_policy_returns_without_prompting(self):
        console = _console()
        with patch("prompt_toolkit.PromptSession.prompt_async") as prompt_async:
            from tamfis_code.runner import resolve_approval_decision_async

            decision = await resolve_approval_decision_async(
                console,
                "npm install",
                "medium",
                "auto",
                interactive=True,
                config=Config(approval_policy="auto"),
            )

        self.assertEqual(decision, "approve_once")
        prompt_async.assert_not_called()
