import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.runner import (
    PROVIDER_NAME_MAP,
    approval_command_preview,
    normalize_provider,
    submit_ai_task_background,
    _print_command_budget_if_notable,
)


class _StatePatchMixin:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


class NormalizeProviderTests(unittest.TestCase):
    def test_none_normalizes_to_auto(self):
        self.assertEqual(normalize_provider(None), "auto")

    def test_aliases_are_expanded(self):
        self.assertEqual(normalize_provider("hf"), "huggingface")
        self.assertEqual(normalize_provider("nvidia"), "nvidia_nim")
        self.assertEqual(normalize_provider("or"), "openrouter")

    def test_canonical_names_pass_through_unchanged(self):
        for name in ("huggingface", "openrouter", "ollama", "nvidia_nim", "gemini", "apiframe", "auto"):
            self.assertEqual(normalize_provider(name), name)

    def test_unsupported_provider_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalize_provider("not-a-real-provider")


class ProviderNameMapTests(unittest.TestCase):
    def test_every_allowed_alias_target_has_a_display_name(self):
        for canonical in ("huggingface", "openrouter", "ollama", "nvidia_nim", "gemini", "apiframe", "auto"):
            self.assertIn(canonical, PROVIDER_NAME_MAP)


class ApprovalCommandPreviewTests(unittest.TestCase):
    def test_short_command_is_returned_unmodified(self):
        self.assertEqual(approval_command_preview("echo hi"), "echo hi")

    def test_long_command_is_truncated_with_omission_marker(self):
        long_command = "x" * 10_000
        preview = approval_command_preview(long_command, limit=100)
        self.assertLess(len(preview), len(long_command))
        self.assertIn("characters omitted", preview)
        self.assertTrue(preview.startswith("x" * 10))
        self.assertTrue(preview.rstrip().endswith("x"))

    def test_preview_keeps_head_and_tail_of_command(self):
        command = "HEAD" + ("x" * 10_000) + "TAIL"
        preview = approval_command_preview(command, limit=200)
        self.assertTrue(preview.startswith("HEAD"))
        self.assertTrue(preview.endswith("TAIL"))


class SubmitAiTaskBackgroundTests(_StatePatchMixin, unittest.TestCase):
    def test_submits_task_and_persists_backgrounded_state(self):
        client = AsyncMock()
        client.run_ai_task = AsyncMock(return_value={"task_id": "t-123"})
        result = asyncio.run(submit_ai_task_background(
            client, session_id=1, objective="fix the bug", mode="coding",
        ))
        self.assertEqual(result["task_id"], "t-123")
        client.run_ai_task.assert_awaited_once()
        state = state_module.get_session_state(1)
        self.assertEqual(state.last_task_id, "t-123")
        self.assertEqual(state.current_phase, "queued")
        # Regression guard: finish_action() (closing out the "submit the
        # task" bookkeeping action) unconditionally sets execution_status to
        # "idle"/"running" as a side effect. submit_ai_task_background must
        # call finish_action() BEFORE writing execution_status="backgrounded"
        # via save_session_state, not after, or the real background-task
        # status gets clobbered back to "idle" immediately even though a
        # real task is running server-side.
        self.assertEqual(state.execution_status, "backgrounded")

    def test_normalizes_provider_alias_before_submitting(self):
        client = AsyncMock()
        client.run_ai_task = AsyncMock(return_value={"task_id": "t-456"})
        asyncio.run(submit_ai_task_background(
            client, session_id=2, objective="x", mode="chat", provider="hf",
        ))
        _, kwargs = client.run_ai_task.call_args
        self.assertEqual(kwargs["provider"], "huggingface")

    def test_chat_mode_is_recorded_as_read_only_risk(self):
        client = AsyncMock()
        client.run_ai_task = AsyncMock(return_value={"task_id": "t-789"})
        asyncio.run(submit_ai_task_background(
            client, session_id=3, objective="explain this", mode="chat",
        ))
        # No direct getter for the risk recorded on the action, but this at
        # least proves the call completes and state is saved without error
        # for a read-only mode.
        state = state_module.get_session_state(3)
        self.assertEqual(state.active_task["mode"], "chat")


class PrintCommandBudgetIfNotableTests(unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200), None

    def test_says_nothing_when_budget_fields_are_absent(self):
        console, _ = self._console()
        client = AsyncMock()
        client.get_task = AsyncMock(return_value={})
        asyncio.run(_print_command_budget_if_notable(client, console, "t1"))
        self.assertEqual(console.file.getvalue(), "")

    def test_says_nothing_when_usage_is_low(self):
        console, _ = self._console()
        client = AsyncMock()
        client.get_task = AsyncMock(return_value={"command_budget": 100, "command_count": 5})
        asyncio.run(_print_command_budget_if_notable(client, console, "t1"))
        self.assertEqual(console.file.getvalue(), "")

    def test_warns_when_usage_crosses_80_percent(self):
        console, _ = self._console()
        client = AsyncMock()
        client.get_task = AsyncMock(return_value={"command_budget": 100, "command_count": 85})
        asyncio.run(_print_command_budget_if_notable(client, console, "t1"))
        self.assertIn("Commands used: 85/100", console.file.getvalue())

    def test_silently_ignores_client_errors(self):
        from tamfis_code.api_client import RemoteAPIError

        console, _ = self._console()
        client = AsyncMock()
        client.get_task = AsyncMock(side_effect=RemoteAPIError(500, "boom"))
        asyncio.run(_print_command_budget_if_notable(client, console, "t1"))  # must not raise
        self.assertEqual(console.file.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
