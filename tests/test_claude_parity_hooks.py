"""Integration tests for the two Claude-Code-parity hook events added
alongside the Codex `interrupt_hooks.rs`/`session_interrupted` work:
`user_prompt_submit` (fires before the objective is classified/sent to a
provider; can block the whole turn or add context) and `session_completed`
(an observe-only notification fired on a successful completion, symmetric
to `session_interrupted` for the failure case). Confirmed live via
hooks.py's own unit tests first (test_hooks.py); these prove the actual
runner_local.py wiring, not just that the hooks.py functions work in
isolation -- the same pattern test_round_budget_extension.py already
established for session_interrupted.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import run_local_agent_turn

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _RecordingRenderer,
    _StatePatchMixin,
    _chunk,
    _delta,
)


class UserPromptSubmitAndSessionCompletedHookTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def test_blocking_hook_stops_the_turn_before_any_provider_call(self):
        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[user_prompt_submit]]\ncommand = "echo \\"not allowed\\" 1>&2; exit 2"\n'
            )
            client = _FakeClient([[_chunk(_delta(content="This must never be reached."))]])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "do something risky"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("not allowed", outcome.error or "")
            self.assertEqual(client.calls, [], "provider must never be called once the hook blocked the turn")

    def test_context_adding_hook_reaches_the_provider_and_completion_fires_session_completed(self):
        with tempfile.TemporaryDirectory() as ws:
            marker = Path(ws) / "completed_marker.txt"
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[user_prompt_submit]]\n'
                'command = "echo \\"reminder: mention RFC 7519\\" 1>&2"\n'
                '\n'
                '[[session_completed]]\n'
                f'command = "cat > {marker}"\n'
            )
            # A question-style objective classifies as chat/inspect, not a
            # coding task -- deliberately avoiding a mutating-task objective
            # (e.g. "add authentication"), which would fail unrelated
            # tool-evidence/mutation-verification gates before ever reaching
            # the completion path this test is actually checking.
            client = _FakeClient([[_chunk(_delta(content="JWT stands for JSON Web Token, mentioning RFC 7519 as reminded."))]])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what is JWT authentication"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            # The injected context must have actually reached the provider
            # call, not just been computed and discarded.
            sent_messages = client.calls[0]["messages"]
            sent_text = json.dumps(sent_messages)
            self.assertIn("Additional context from hook: reminder: mention RFC 7519", sent_text)

            self.assertTrue(marker.is_file(), "session_completed hook never ran")
            payload = json.loads(marker.read_text())
            self.assertEqual(payload["event"], "session_completed")
            self.assertEqual(payload["session_id"], 1)
            self.assertIn("JWT stands for JSON Web Token", payload["summary"])


if __name__ == "__main__":
    unittest.main()
