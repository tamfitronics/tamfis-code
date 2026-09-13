"""Tests for the user-configurable pre/post-tool-use hook mechanism
(hooks.py) -- a real Claude-Code-style PreToolUse/PostToolUse parity gap
that had no equivalent in this codebase at all before this module."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tamfis_code import hooks as hooks_module
from tamfis_code.hooks import (
    HookDefinition, load_hooks, run_session_completed_hooks, run_session_hooks,
    run_tool_hooks, run_user_prompt_submit_hooks,
)


class TestLoadHooks:
    def setup_method(self):
        self._original_hooks_path = hooks_module.HOOKS_PATH
        self.tmp = tempfile.TemporaryDirectory()
        hooks_module.HOOKS_PATH = Path(self.tmp.name) / "user" / "hooks.toml"

    def teardown_method(self):
        hooks_module.HOOKS_PATH = self._original_hooks_path
        self.tmp.cleanup()

    def test_missing_files_return_no_hooks(self):
        assert load_hooks(str(Path(self.tmp.name) / "project")) == []

    def test_loads_user_hooks(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text(
            '[[pre_tool_use]]\nmatcher = "write_file"\ncommand = "echo hi"\n'
        )
        loaded = load_hooks()
        assert len(loaded) == 1
        assert loaded[0] == HookDefinition(
            event="pre_tool_use", matcher="write_file", command="echo hi", source="user config",
        )

    def test_loads_project_hooks_after_user_hooks(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[pre_tool_use]]\ncommand = "echo user"\n')
        project_root = Path(self.tmp.name) / "project"
        (project_root / ".tamfis").mkdir(parents=True)
        (project_root / ".tamfis" / "hooks.toml").write_text(
            '[[post_tool_use]]\ncommand = "echo project"\n'
        )
        loaded = load_hooks(str(project_root))
        assert [h.source for h in loaded] == ["user config", "project config"]

    def test_entries_without_a_command_are_skipped(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[pre_tool_use]]\nmatcher = "x"\n')
        assert load_hooks() == []

    def test_malformed_toml_returns_no_hooks_instead_of_raising(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text("this is not valid toml [[[")
        assert load_hooks() == []

    def test_loads_session_interrupted_hooks(self):
        # Codex-parity addition (interrupt_hooks.rs): a third hook-event
        # category, alongside pre/post_tool_use, for a checkpointed
        # interruption rather than a specific tool call.
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text(
            '[[session_interrupted]]\ncommand = "notify-send interrupted"\n'
        )
        loaded = load_hooks()
        assert len(loaded) == 1
        assert loaded[0] == HookDefinition(
            event="session_interrupted", matcher="", command="notify-send interrupted", source="user config",
        )


class TestRunToolHooks:
    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_tool_hooks([], "pre_tool_use", tool_name="write_file", tool_input={}, session_id=1, workspace_root=".") == []

    @pytest.mark.asyncio
    async def test_matcher_filters_by_tool_name(self):
        hooks = [HookDefinition(event="pre_tool_use", matcher="write_file", command="exit 0", source="user config")]
        results = await run_tool_hooks(hooks, "pre_tool_use", tool_name="read_file", tool_input={}, session_id=1, workspace_root=".")
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_matcher_matches_every_tool(self):
        hooks = [HookDefinition(event="pre_tool_use", matcher="", command="echo matched 1>&2", source="user config")]
        results = await run_tool_hooks(hooks, "pre_tool_use", tool_name="anything_at_all", tool_input={}, session_id=1, workspace_root=".")
        assert len(results) == 1
        assert results[0].message == "matched"
        assert results[0].blocked is False

    @pytest.mark.asyncio
    async def test_exit_code_2_blocks_pre_tool_use(self):
        hooks = [HookDefinition(event="pre_tool_use", matcher="", command='echo "no, not that" 1>&2; exit 2', source="user config")]
        results = await run_tool_hooks(hooks, "pre_tool_use", tool_name="execute_command", tool_input={}, session_id=1, workspace_root=".")
        assert len(results) == 1
        assert results[0].blocked is True
        assert results[0].message == "no, not that"

    @pytest.mark.asyncio
    async def test_exit_code_2_has_no_special_meaning_for_post_tool_use(self):
        # PostToolUse can never veto -- the tool already ran.
        hooks = [HookDefinition(event="post_tool_use", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_tool_hooks(
            hooks, "post_tool_use", tool_name="write_file", tool_input={}, tool_output={"success": True},
            session_id=1, workspace_root=".",
        )
        assert len(results) == 1
        assert results[0].blocked is False
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_non_zero_non_two_exit_does_not_block(self):
        hooks = [HookDefinition(event="pre_tool_use", matcher="", command='echo "just a warning" 1>&2; exit 1', source="user config")]
        results = await run_tool_hooks(hooks, "pre_tool_use", tool_name="execute_command", tool_input={}, session_id=1, workspace_root=".")
        assert results[0].blocked is False
        assert results[0].message == "just a warning"

    @pytest.mark.asyncio
    async def test_first_blocking_hook_stops_evaluation_of_later_pre_hooks(self):
        hooks = [
            HookDefinition(event="pre_tool_use", matcher="", command='echo "blocked" 1>&2; exit 2', source="user config"),
            HookDefinition(event="pre_tool_use", matcher="", command='echo "should never run" 1>&2', source="project config"),
        ]
        results = await run_tool_hooks(hooks, "pre_tool_use", tool_name="execute_command", tool_input={}, session_id=1, workspace_root=".")
        assert len(results) == 1
        assert results[0].blocked is True

    @pytest.mark.asyncio
    async def test_a_command_that_cannot_start_reports_a_diagnostic_not_a_crash(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="", command="/definitely/not/a/real/executable --flag",
            source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="execute_command", tool_input={}, session_id=1, workspace_root=".",
        )
        assert len(results) == 1
        assert results[0].blocked is False
        assert results[0].message

    @pytest.mark.asyncio
    async def test_receives_the_event_payload_on_stdin(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="",
            command="python3 -c \"import sys, json; d = json.load(sys.stdin); print(d['tool_name'] + ':' + d['tool_input']['path'], file=sys.stderr)\"",
            source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="write_file", tool_input={"path": "app.py"},
            session_id=42, workspace_root=".",
        )
        assert results[0].message == "write_file:app.py"

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        # HOOK_TIMEOUT_SECONDS defaults to 30s, which would make this test
        # itself hang for 30s on every run -- patch it down so the timeout
        # path is actually exercised instead of the hook simply finishing
        # first. Mirrors Codex's own hooks_executor test intent: a hook that
        # never returns must never be able to stall the whole turn.
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(
                event="pre_tool_use", matcher="", command="sleep 5", source="user config",
            )]
            results = await run_tool_hooks(
                hooks, "pre_tool_use", tool_name="execute_command", tool_input={},
                session_id=1, workspace_root=".",
            )
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert len(results) == 1
        assert results[0].blocked is False
        assert "timed out" in results[0].message
        assert "killed" in results[0].message


class TestRunSessionHooks:
    """session_interrupted hooks (Codex-parity: interrupt_hooks.rs) have no
    tool_name/matcher to filter on -- every configured hook for the event
    runs unconditionally, and, like PostToolUse, is always observe-only
    since the interruption already happened by the time it fires."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_session_hooks([], "session_interrupted", session_id=1, workspace_root=".") == []

    @pytest.mark.asyncio
    async def test_every_configured_hook_runs_unconditionally(self):
        hooks = [
            HookDefinition(event="session_interrupted", matcher="", command='echo "one" 1>&2', source="user config"),
            HookDefinition(event="session_interrupted", matcher="", command='echo "two" 1>&2', source="project config"),
        ]
        results = await run_session_hooks(hooks, "session_interrupted", session_id=1, workspace_root=".")
        assert [r.message for r in results] == ["one", "two"]
        assert all(r.blocked is False for r in results)

    @pytest.mark.asyncio
    async def test_exit_code_2_has_no_special_meaning(self):
        # Unlike pre_tool_use, there is nothing left to block -- the turn
        # was already checkpointed as interrupted before this fires.
        hooks = [HookDefinition(event="session_interrupted", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_session_hooks(hooks, "session_interrupted", session_id=1, workspace_root=".")
        assert results[0].blocked is False
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_a_command_that_cannot_start_reports_a_diagnostic_not_a_crash(self):
        hooks = [HookDefinition(
            event="session_interrupted", matcher="", command="/definitely/not/a/real/executable",
            source="user config",
        )]
        results = await run_session_hooks(hooks, "session_interrupted", session_id=1, workspace_root=".")
        assert len(results) == 1
        assert results[0].blocked is False
        assert results[0].message

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="session_interrupted", matcher="", command="sleep 5", source="user config")]
            results = await run_session_hooks(hooks, "session_interrupted", session_id=1, workspace_root=".")
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert len(results) == 1
        assert "timed out" in results[0].message
        assert "killed" in results[0].message

    @pytest.mark.asyncio
    async def test_receives_the_session_and_reason_on_stdin(self):
        hooks = [HookDefinition(
            event="session_interrupted", matcher="",
            command="python3 -c \"import sys, json; d = json.load(sys.stdin); "
                    "print(str(d['session_id']) + ':' + d['reason'], file=sys.stderr)\"",
            source="user config",
        )]
        results = await run_session_hooks(
            hooks, "session_interrupted", session_id=42, workspace_root=".", reason="provider streaming failed",
        )
        assert results[0].message == "42:provider streaming failed"


class TestRunUserPromptSubmitHooks:
    """Claude-Code-parity addition: fires once per turn before the objective
    is classified/sent to a provider. Unlike pre_tool_use, there is no
    tool_name/matcher -- every configured hook runs. Exit code 2 blocks the
    whole turn (mirrors pre_tool_use's own block contract); any other
    output is folded in by the caller as additional context, Claude Code's
    "add context" capability for this event."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_user_prompt_submit_hooks([], session_id=1, workspace_root=".", objective="x") == []

    @pytest.mark.asyncio
    async def test_exit_code_2_blocks_the_turn(self):
        hooks = [HookDefinition(event="user_prompt_submit", matcher="", command='echo "not allowed" 1>&2; exit 2', source="user config")]
        results = await run_user_prompt_submit_hooks(hooks, session_id=1, workspace_root=".", objective="do something risky")
        assert results[0].blocked is True
        assert results[0].message == "not allowed"

    @pytest.mark.asyncio
    async def test_non_blocking_output_is_returned_as_added_context(self):
        hooks = [HookDefinition(event="user_prompt_submit", matcher="", command='echo "reminder: use JWT" 1>&2', source="user config")]
        results = await run_user_prompt_submit_hooks(hooks, session_id=1, workspace_root=".", objective="add auth")
        assert results[0].blocked is False
        assert results[0].message == "reminder: use JWT"

    @pytest.mark.asyncio
    async def test_first_blocking_hook_stops_evaluation_of_later_hooks(self):
        hooks = [
            HookDefinition(event="user_prompt_submit", matcher="", command='echo "blocked" 1>&2; exit 2', source="user config"),
            HookDefinition(event="user_prompt_submit", matcher="", command='echo "should never run" 1>&2', source="project config"),
        ]
        results = await run_user_prompt_submit_hooks(hooks, session_id=1, workspace_root=".", objective="x")
        assert len(results) == 1
        assert results[0].blocked is True

    @pytest.mark.asyncio
    async def test_receives_the_objective_on_stdin(self):
        hooks = [HookDefinition(
            event="user_prompt_submit", matcher="",
            command="python3 -c \"import sys, json; d = json.load(sys.stdin); print(d['objective'], file=sys.stderr)\"",
            source="user config",
        )]
        results = await run_user_prompt_submit_hooks(hooks, session_id=1, workspace_root=".", objective="fix the flaky test")
        assert results[0].message == "fix the flaky test"

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="user_prompt_submit", matcher="", command="sleep 5", source="user config")]
            results = await run_user_prompt_submit_hooks(hooks, session_id=1, workspace_root=".", objective="x")
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert results[0].blocked is False
        assert "timed out" in results[0].message


class TestRunSessionCompletedHooks:
    """Observe-only completion notification, symmetric to
    TestRunSessionHooks (session_interrupted) for the success case. Never
    blocks -- there is nothing left to block by the time a turn has
    already completed successfully."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_session_completed_hooks([], session_id=1, workspace_root=".") == []

    @pytest.mark.asyncio
    async def test_exit_code_2_has_no_special_meaning(self):
        hooks = [HookDefinition(event="session_completed", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_session_completed_hooks(hooks, session_id=1, workspace_root=".")
        assert results[0].blocked is False
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_receives_the_summary_on_stdin(self):
        hooks = [HookDefinition(
            event="session_completed", matcher="",
            command="python3 -c \"import sys, json; d = json.load(sys.stdin); print(d['summary'], file=sys.stderr)\"",
            source="user config",
        )]
        results = await run_session_completed_hooks(hooks, session_id=1, workspace_root=".", summary="fixed the bug")
        assert results[0].message == "fixed the bug"

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="session_completed", matcher="", command="sleep 5", source="user config")]
            results = await run_session_completed_hooks(hooks, session_id=1, workspace_root=".")
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert "timed out" in results[0].message
