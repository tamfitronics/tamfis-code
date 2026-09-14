"""Tests for the user-configurable pre/post-tool-use hook mechanism
(hooks.py) -- a real Claude-Code-style PreToolUse/PostToolUse parity gap
that had no equivalent in this codebase at all before this module."""

import asyncio
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tamfis_code import hooks as hooks_module
from tamfis_code import state as state_module
from tamfis_code.hooks import (
    HookDefinition, load_hooks, run_notification_hooks, run_pre_compact_hooks,
    run_session_completed_hooks, run_session_end_hooks, run_session_hooks,
    run_session_start_hooks, run_subagent_stop_hooks, run_tool_hooks,
    run_user_prompt_submit_hooks,
)


def _fake_async_client(post_side_effect):
    """Mirrors test_mcp.py's/test_session_title.py's own helper of the
    same name -- stands in for ``async with httpx.AsyncClient(...) as
    client: await client.post(...)``."""
    client = MagicMock()
    client.post = AsyncMock(side_effect=post_side_effect)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


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

    def test_loads_session_start_and_session_end_hooks(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text(
            '[[session_start]]\ncommand = "echo start"\n\n'
            '[[session_end]]\ncommand = "echo end"\n'
        )
        loaded = load_hooks()
        assert [h.event for h in loaded] == ["session_start", "session_end"]

    def test_loads_subagent_stop_hooks(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[subagent_stop]]\ncommand = "echo done"\n')
        loaded = load_hooks()
        assert [h.event for h in loaded] == ["subagent_stop"]

    def test_loads_pre_compact_hooks(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[pre_compact]]\ncommand = "echo keep this"\n')
        loaded = load_hooks()
        assert [h.event for h in loaded] == ["pre_compact"]

    def test_loads_notification_hooks(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[notification]]\ncommand = "notify-send done"\n')
        loaded = load_hooks()
        assert [h.event for h in loaded] == ["notification"]

    def test_loads_the_optional_if_command_field(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text(
            '[[pre_tool_use]]\nmatcher = "execute_command"\nif_command = "git commit*"\ncommand = "echo hi"\n'
        )
        loaded = load_hooks()
        assert loaded[0].if_command == "git commit*"

    def test_if_command_defaults_to_empty_when_absent(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[pre_tool_use]]\ncommand = "echo hi"\n')
        loaded = load_hooks()
        assert loaded[0].if_command == ""


class TestIfCommandMatching:
    """Claude-Code-parity addition: a simplified, fnmatch-glob equivalent
    of Claude Code's real `"if": "Bash(git commit:*)"` syntax -- gates a
    pre_tool_use/post_tool_use hook on the actual command content, not
    just the tool name."""

    @pytest.mark.asyncio
    async def test_matching_command_fires_the_hook(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="execute_command", if_command="git commit*",
            command='echo "blocked commit" 1>&2; exit 2', source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="execute_command",
            tool_input={"command": "git commit -m fix"}, session_id=1, workspace_root=".",
        )
        assert results[0].blocked is True
        assert results[0].message == "blocked commit"

    @pytest.mark.asyncio
    async def test_non_matching_command_does_not_fire_the_hook(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="execute_command", if_command="git commit*",
            command='echo "should never run" 1>&2; exit 2', source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="execute_command",
            tool_input={"command": "ls -la"}, session_id=1, workspace_root=".",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_a_tool_with_no_command_argument_never_matches_if_command(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="write_file", if_command="git commit*",
            command='echo "should never run" 1>&2; exit 2', source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="write_file",
            tool_input={"path": "app.py"}, session_id=1, workspace_root=".",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_matcher_still_applies_alongside_if_command(self):
        # A hook can require both the tool name AND the command content to
        # match -- if_command alone is not a substitute for matcher.
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="write_file", if_command="git commit*",
            command='echo "should never run" 1>&2; exit 2', source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="execute_command",
            tool_input={"command": "git commit -m fix"}, session_id=1, workspace_root=".",
        )
        assert results == []


class TestUpdatedInputMutation:
    """Claude-Code-parity addition: a non-blocked pre_tool_use hook can
    rewrite the pending call's arguments by printing
    {"updated_input": {...}} as its whole stdout -- a flatter, simpler
    shape than Claude Code's real nested hookSpecificOutput.updatedInput
    (tamfis-code has no systemMessage/permissionDecision concept to nest
    this alongside)."""

    @pytest.mark.asyncio
    async def test_updated_input_is_parsed_from_stdout(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="write_file",
            command='echo \'{"updated_input": {"content": "sanitized"}}\'',
            source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="write_file",
            tool_input={"path": "x.py", "content": "raw"}, session_id=1, workspace_root=".",
        )
        assert results[0].blocked is False
        assert results[0].updated_input == {"content": "sanitized"}
        # The raw JSON blob is not also dumped as a diagnostic message.
        assert results[0].message == ""

    @pytest.mark.asyncio
    async def test_plain_non_json_stdout_is_not_mistaken_for_updated_input(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="write_file", command='echo "just some text"', source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="write_file", tool_input={"path": "x.py"},
            session_id=1, workspace_root=".",
        )
        assert results[0].updated_input is None
        assert results[0].message == "just some text"

    @pytest.mark.asyncio
    async def test_json_without_an_updated_input_key_is_not_mistaken_for_a_mutation(self):
        hooks = [HookDefinition(
            event="pre_tool_use", matcher="write_file", command='echo \'{"other": "field"}\'', source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "pre_tool_use", tool_name="write_file", tool_input={"path": "x.py"},
            session_id=1, workspace_root=".",
        )
        assert results[0].updated_input is None

    @pytest.mark.asyncio
    async def test_updated_input_is_never_parsed_for_post_tool_use(self):
        # Mutating a call after it already ran makes no sense -- post_tool_use
        # never attempts updated_input parsing at all.
        hooks = [HookDefinition(
            event="post_tool_use", matcher="write_file",
            command='echo \'{"updated_input": {"content": "sanitized"}}\'',
            source="user config",
        )]
        results = await run_tool_hooks(
            hooks, "post_tool_use", tool_name="write_file", tool_input={"path": "x.py"},
            tool_output={"success": True}, session_id=1, workspace_root=".",
        )
        assert results[0].updated_input is None
        assert "updated_input" in results[0].message  # surfaced as plain text instead


class TestParallelHookExecution:
    """Claude-Code-parity addition: every matching hook for one event runs
    concurrently (asyncio.gather), matching Claude Code's own documented
    "all matching hooks run in parallel" contract, rather than one at a
    time."""

    @pytest.mark.asyncio
    async def test_two_slow_hooks_run_concurrently_not_sequentially(self):
        import time
        hooks = [
            HookDefinition(event="post_tool_use", matcher="", command="sleep 0.5", source="user config"),
            HookDefinition(event="post_tool_use", matcher="", command="sleep 0.5", source="project config"),
        ]
        start = time.monotonic()
        await run_tool_hooks(
            hooks, "post_tool_use", tool_name="write_file", tool_input={}, session_id=1, workspace_root=".",
        )
        elapsed = time.monotonic() - start
        # Sequential execution would take ~1.0s; concurrent takes ~0.5s.
        # A generous ceiling avoids flaking under real subprocess/CI jitter.
        assert elapsed < 0.9, f"hooks did not run concurrently (took {elapsed:.2f}s)"

    @pytest.mark.asyncio
    async def test_a_hook_after_a_blocking_one_still_actually_runs_but_its_result_is_dropped(self):
        # Documents the real trade-off: under true parallel execution, a
        # later hook's subprocess still runs to completion even though an
        # earlier hook already blocked the call -- there is no way to
        # cancel an already-launched hook without breaking Claude Code's
        # own "hooks don't see each other's output" independence contract.
        # Only its reported RESULT is dropped from the returned list.
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "ran.txt"
            hooks = [
                HookDefinition(event="pre_tool_use", matcher="", command='echo "blocked" 1>&2; exit 2', source="user config"),
                HookDefinition(event="pre_tool_use", matcher="", command=f"touch {marker}", source="project config"),
            ]
            results = await run_tool_hooks(
                hooks, "pre_tool_use", tool_name="execute_command", tool_input={}, session_id=1, workspace_root=".",
            )
            assert len(results) == 1
            assert results[0].blocked is True
            assert marker.is_file(), "the later hook's subprocess should still have run"


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
    """Completion notification with Claude-Code Stop-hook semantics."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_session_completed_hooks([], session_id=1, workspace_root=".") == []

    @pytest.mark.asyncio
    async def test_exit_code_2_blocks_for_stop_hooks(self):
        hooks = [HookDefinition(event="session_completed", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_session_completed_hooks(hooks, session_id=1, workspace_root=".")
        assert results[0].blocked is True
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_structured_block_decision_is_returned(self):
        hooks = [HookDefinition(
            event="session_completed", matcher="",
            command="python3 -c \"import json; print(json.dumps({'decision': 'block', 'reason': 'run tests'}))\"",
            source="user config",
        )]
        results = await run_session_completed_hooks(hooks, session_id=1, workspace_root=".")
        assert results[0].blocked is True
        assert results[0].message == "run tests"

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


class TestRunSessionStartHooks:
    """Claude-Code-parity addition: fires once when an interactive REPL
    session begins. Observe-only -- there is no tool_name/matcher, and no
    way to block a session from starting."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_session_start_hooks([], session_id=1, workspace_root=".") == []

    @pytest.mark.asyncio
    async def test_every_configured_hook_runs_and_output_is_returned(self):
        hooks = [
            HookDefinition(event="session_start", matcher="", command='echo "one" 1>&2', source="user config"),
            HookDefinition(event="session_start", matcher="", command='echo "two" 1>&2', source="project config"),
        ]
        results = await run_session_start_hooks(hooks, session_id=1, workspace_root=".")
        assert [r.message for r in results] == ["one", "two"]
        assert all(r.blocked is False for r in results)

    @pytest.mark.asyncio
    async def test_exit_code_2_has_no_special_meaning(self):
        hooks = [HookDefinition(event="session_start", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_session_start_hooks(hooks, session_id=1, workspace_root=".")
        assert results[0].blocked is False
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_receives_the_session_id_on_stdin(self):
        hooks = [HookDefinition(
            event="session_start", matcher="",
            command="python3 -c \"import sys, json; d = json.load(sys.stdin); print(d['session_id'], file=sys.stderr)\"",
            source="user config",
        )]
        results = await run_session_start_hooks(hooks, session_id=99, workspace_root=".")
        assert results[0].message == "99"

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="session_start", matcher="", command="sleep 5", source="user config")]
            results = await run_session_start_hooks(hooks, session_id=1, workspace_root=".")
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert "timed out" in results[0].message


class TestRunSessionEndHooks:
    """Claude-Code-parity addition: fires once when an interactive REPL
    session ends, regardless of which exit path was taken (see
    test_tamfis_code_repl_exit.py's SessionStartEndHookTests for the real
    end-to-end proof across two different exit paths)."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_session_end_hooks([], session_id=1, workspace_root=".") == []

    @pytest.mark.asyncio
    async def test_exit_code_2_has_no_special_meaning(self):
        hooks = [HookDefinition(event="session_end", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_session_end_hooks(hooks, session_id=1, workspace_root=".")
        assert results[0].blocked is False
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_receives_the_workspace_root_on_stdin(self):
        with tempfile.TemporaryDirectory() as ws:
            hooks = [HookDefinition(
                event="session_end", matcher="",
                command="python3 -c \"import sys, json; d = json.load(sys.stdin); print(d['workspace_root'], file=sys.stderr)\"",
                source="user config",
            )]
            results = await run_session_end_hooks(hooks, session_id=1, workspace_root=ws)
            assert results[0].message == ws

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="session_end", matcher="", command="sleep 5", source="user config")]
            results = await run_session_end_hooks(hooks, session_id=1, workspace_root=".")
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert "timed out" in results[0].message


class TestRunSubagentStopHooks:
    """Claude-Code-parity addition: fires once per delegated swarm sub-task
    when it finishes (see test_swarm.py's ExecuteTasksSubagentStopHookTests
    for the real end-to-end proof against agents.py's execute_tasks).
    Observe-only, like session_completed -- no way for a hook to force a
    finished sub-task to keep working."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        result = await run_subagent_stop_hooks(
            [], session_id=1, workspace_root=".", task_id="t1", description="x", status="completed",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_exit_code_2_has_no_special_meaning(self):
        hooks = [HookDefinition(event="subagent_stop", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_subagent_stop_hooks(
            hooks, session_id=1, workspace_root=".", task_id="t1", description="x", status="completed",
        )
        assert results[0].blocked is False
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_receives_the_task_id_description_status_and_error_on_stdin(self):
        hooks = [HookDefinition(
            event="subagent_stop", matcher="",
            command=(
                "python3 -c \"import sys, json; d = json.load(sys.stdin); "
                "print(d['task_id'] + ':' + d['description'] + ':' + d['status'] + ':' + d['error'], "
                "file=sys.stderr)\""
            ),
            source="user config",
        )]
        results = await run_subagent_stop_hooks(
            hooks, session_id=1, workspace_root=".", task_id="delegated_abc",
            description="fix the bug", status="failed", error="boom",
        )
        assert results[0].message == "delegated_abc:fix the bug:failed:boom"

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="subagent_stop", matcher="", command="sleep 5", source="user config")]
            results = await run_subagent_stop_hooks(
                hooks, session_id=1, workspace_root=".", task_id="t1", description="x", status="completed",
            )
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert "timed out" in results[0].message


class TestRunPreCompactHooks:
    """Claude-Code-parity addition: fires immediately before /compact folds
    older turns into the conversation summary -- tamfis-code's only
    compaction trigger. A hook's output is folded into the preserved
    summary by the caller (see test_thread_compression.py's
    test_preserve_note_from_a_pre_compact_hook_survives_the_fold), not
    just logged."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_pre_compact_hooks([], session_id=1, workspace_root=".") == []

    @pytest.mark.asyncio
    async def test_every_configured_hook_runs_and_output_is_returned(self):
        hooks = [
            HookDefinition(event="pre_compact", matcher="", command='echo "keep this" 1>&2', source="user config"),
        ]
        results = await run_pre_compact_hooks(hooks, session_id=1, workspace_root=".")
        assert results[0].message == "keep this"
        assert results[0].blocked is False

    @pytest.mark.asyncio
    async def test_receives_the_session_id_on_stdin(self):
        hooks = [HookDefinition(
            event="pre_compact", matcher="",
            command="python3 -c \"import sys, json; d = json.load(sys.stdin); print(d['session_id'], file=sys.stderr)\"",
            source="user config",
        )]
        results = await run_pre_compact_hooks(hooks, session_id=42, workspace_root=".")
        assert results[0].message == "42"

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="pre_compact", matcher="", command="sleep 5", source="user config")]
            results = await run_pre_compact_hooks(hooks, session_id=1, workspace_root=".")
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert "timed out" in results[0].message


class TestRunNotificationHooks:
    """Claude-Code-parity addition: fires whenever tamfis-code delivers a
    notification-style message into a session (see
    test_background_lifecycle.py's real end-to-end proof against
    background.py's update_job_status). Observe-only."""

    @pytest.mark.asyncio
    async def test_no_hooks_configured_is_a_cheap_noop(self):
        assert await run_notification_hooks([], session_id=1, workspace_root=".", message="x") == []

    @pytest.mark.asyncio
    async def test_exit_code_2_has_no_special_meaning(self):
        hooks = [HookDefinition(event="notification", matcher="", command='echo "fyi" 1>&2; exit 2', source="user config")]
        results = await run_notification_hooks(hooks, session_id=1, workspace_root=".", message="job done")
        assert results[0].blocked is False
        assert results[0].message == "fyi"

    @pytest.mark.asyncio
    async def test_receives_the_message_on_stdin(self):
        hooks = [HookDefinition(
            event="notification", matcher="",
            command="python3 -c \"import sys, json; d = json.load(sys.stdin); print(d['message'], file=sys.stderr)\"",
            source="user config",
        )]
        results = await run_notification_hooks(hooks, session_id=1, workspace_root=".", message="background job finished")
        assert results[0].message == "background job finished"

    @pytest.mark.asyncio
    async def test_a_hanging_hook_is_killed_and_reported_as_non_blocking(self):
        original = hooks_module.HOOK_TIMEOUT_SECONDS
        hooks_module.HOOK_TIMEOUT_SECONDS = 0.2
        try:
            hooks = [HookDefinition(event="notification", matcher="", command="sleep 5", source="user config")]
            results = await run_notification_hooks(hooks, session_id=1, workspace_root=".", message="x")
        finally:
            hooks_module.HOOK_TIMEOUT_SECONDS = original
        assert "timed out" in results[0].message


def _prompt_hook(**overrides):
    fields = dict(
        event="pre_tool_use", matcher="execute_command", command="", source="user config",
        hook_type="prompt", prompt="Evaluate: $TOOL_INPUT",
    )
    fields.update(overrides)
    return HookDefinition(**fields)


class TestLoadHooksPromptAndRewakeFields:
    def test_loads_a_prompt_type_hook(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text(
            '[[pre_tool_use]]\ntype = "prompt"\nprompt = "Evaluate: $TOOL_INPUT"\n'
        )
        loaded = load_hooks()
        assert len(loaded) == 1
        assert loaded[0].hook_type == "prompt"
        assert loaded[0].prompt == "Evaluate: $TOOL_INPUT"

    def test_a_prompt_type_hook_with_no_prompt_text_is_skipped(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[pre_tool_use]]\ntype = "prompt"\n')
        assert load_hooks() == []

    def test_command_type_is_the_default_and_unaffected(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[pre_tool_use]]\ncommand = "echo hi"\n')
        loaded = load_hooks()
        assert loaded[0].hook_type == "command"
        assert loaded[0].prompt == ""

    def test_loads_async_rewake_and_rewake_message(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text(
            '[[post_tool_use]]\ncommand = "echo hi"\nasync_rewake = true\n'
            'rewake_message = "Found: $FINDINGS"\n'
        )
        loaded = load_hooks()
        assert loaded[0].async_rewake is True
        assert loaded[0].rewake_message == "Found: $FINDINGS"

    def test_async_rewake_defaults_to_false(self):
        hooks_module.HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hooks_module.HOOKS_PATH.write_text('[[post_tool_use]]\ncommand = "echo hi"\n')
        loaded = load_hooks()
        assert loaded[0].async_rewake is False
        assert loaded[0].rewake_message == ""


class TestSubstituteTemplate:
    def test_substitutes_available_payload_keys(self):
        text = hooks_module._substitute_template(
            "Tool input was $TOOL_INPUT, reason was $REASON",
            {"tool_input": {"path": "x.py"}, "reason": "boom"},
        )
        assert '"path": "x.py"' in text
        assert "boom" in text

    def test_missing_keys_are_left_as_literal_text(self):
        text = hooks_module._substitute_template("Summary: $SUMMARY", {})
        assert text == "Summary: $SUMMARY"

    def test_findings_placeholder_uses_the_findings_kwarg(self):
        text = hooks_module._substitute_template("Found: $FINDINGS", {}, findings="a bug")
        assert text == "Found: a bug"


class TestPromptBasedHooks:
    """Claude-Code-parity addition: a `type = "prompt"` hook asks an LLM
    to decide the outcome instead of running a shell command, via the
    same internal Tier IV endpoint state.py's upgrade_session_title_with_ai
    already uses."""

    @pytest.mark.asyncio
    async def test_approve_decision_does_not_block(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": '{"decision": "approve", "reason": "looks safe"}'}}],
        }

        async def fake_post(*args, **kwargs):
            return response

        hook = _prompt_hook()
        with patch("httpx.AsyncClient", _fake_async_client(fake_post)):
            results = await run_tool_hooks(
                [hook], "pre_tool_use", tool_name="execute_command",
                tool_input={"command": "ls"}, session_id=1, workspace_root=".",
            )
        assert results[0].blocked is False
        assert results[0].message == "looks safe"

    @pytest.mark.asyncio
    async def test_deny_decision_blocks_a_blocking_capable_event(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": '{"decision": "deny", "reason": "too risky"}'}}],
        }

        async def fake_post(*args, **kwargs):
            return response

        hook = _prompt_hook()
        with patch("httpx.AsyncClient", _fake_async_client(fake_post)):
            results = await run_tool_hooks(
                [hook], "pre_tool_use", tool_name="execute_command",
                tool_input={"command": "rm -rf /"}, session_id=1, workspace_root=".",
            )
        assert results[0].blocked is True
        assert results[0].message == "too risky"

    @pytest.mark.asyncio
    async def test_deny_decision_never_blocks_a_non_blocking_capable_event(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": '{"decision": "deny", "reason": "flagged"}'}}],
        }

        async def fake_post(*args, **kwargs):
            return response

        hook = _prompt_hook(event="post_tool_use")
        with patch("httpx.AsyncClient", _fake_async_client(fake_post)):
            results = await run_tool_hooks(
                [hook], "post_tool_use", tool_name="execute_command",
                tool_input={"command": "ls"}, tool_output={"success": True}, session_id=1, workspace_root=".",
            )
        assert results[0].blocked is False
        assert results[0].message == "flagged"

    @pytest.mark.asyncio
    async def test_unreachable_endpoint_fails_open_with_a_diagnostic(self):
        import httpx as httpx_module

        async def fake_post(*args, **kwargs):
            raise httpx_module.ConnectError("boom")

        hook = _prompt_hook()
        with patch("httpx.AsyncClient", _fake_async_client(fake_post)):
            results = await run_tool_hooks(
                [hook], "pre_tool_use", tool_name="execute_command",
                tool_input={"command": "ls"}, session_id=1, workspace_root=".",
            )
        assert results[0].blocked is False
        assert "could not be evaluated" in results[0].message
        assert "fail open" in results[0].message

    @pytest.mark.asyncio
    async def test_malformed_json_response_fails_open_with_a_diagnostic(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"choices": [{"message": {"content": "not json at all"}}]}

        async def fake_post(*args, **kwargs):
            return response

        hook = _prompt_hook()
        with patch("httpx.AsyncClient", _fake_async_client(fake_post)):
            results = await run_tool_hooks(
                [hook], "pre_tool_use", tool_name="execute_command",
                tool_input={"command": "ls"}, session_id=1, workspace_root=".",
            )
        assert results[0].blocked is False
        assert "fail open" in results[0].message

    @pytest.mark.asyncio
    async def test_non_200_response_fails_open_with_a_diagnostic(self):
        response = MagicMock()
        response.status_code = 500

        async def fake_post(*args, **kwargs):
            return response

        hook = _prompt_hook()
        with patch("httpx.AsyncClient", _fake_async_client(fake_post)):
            results = await run_tool_hooks(
                [hook], "pre_tool_use", tool_name="execute_command",
                tool_input={"command": "ls"}, session_id=1, workspace_root=".",
            )
        assert results[0].blocked is False
        assert "HTTP 500" in results[0].message
        assert "fail open" in results[0].message


class TestAsyncRewake:
    """Claude-Code-parity addition: a hook with async_rewake=true runs
    completely detached from the triggering call, and its findings are
    queued as a classification="follow_up" instruction once it finishes."""

    def setup_method(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None

    def teardown_method(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self._tmp.cleanup()

    async def _drain(self):
        # Every test in this class must let its rewake task(s) finish (or
        # explicitly drain them) before returning -- pytest-asyncio tears
        # down its event loop per test the same way asyncio.run() does,
        # and a still-pending subprocess-backed task left dangling at that
        # point reproduces the exact hang drain_pending_rewake_tasks
        # exists to prevent (confirmed live against a bare minimal repro
        # during development of this feature).
        await hooks_module.drain_pending_rewake_tasks()

    @pytest.mark.asyncio
    async def test_returns_immediately_without_waiting_for_the_rewake_hook(self):
        import time
        state_module.save_session_state(1, workspace_root=".")
        hook = HookDefinition(
            event="post_tool_use", matcher="", command="sleep 0.5; echo done",
            source="user config", async_rewake=True,
        )
        start = time.monotonic()
        results = await run_tool_hooks(
            [hook], "post_tool_use", tool_name="write_file", tool_input={},
            tool_output={"success": True}, session_id=1, workspace_root=".",
        )
        elapsed = time.monotonic() - start
        try:
            assert results == []
            assert elapsed < 0.3, f"should not have waited for the backgrounded hook (took {elapsed:.2f}s)"
        finally:
            await self._drain()

    @pytest.mark.asyncio
    async def test_findings_are_queued_as_a_follow_up_once_the_hook_finishes(self):
        state_module.save_session_state(1, workspace_root=".")
        hook = HookDefinition(
            event="post_tool_use", matcher="", command="sleep 0.3; echo 'found a bug'",
            source="user config", async_rewake=True, rewake_message="Background review: $FINDINGS",
        )
        await run_tool_hooks(
            [hook], "post_tool_use", tool_name="write_file", tool_input={},
            tool_output={"success": True}, session_id=1, workspace_root=".",
        )
        assert state_module.get_session_state(1).queued_user_instructions == []
        await asyncio.sleep(0.6)
        queued = state_module.get_session_state(1).queued_user_instructions
        assert len(queued) == 1
        assert queued[0]["classification"] == "follow_up"
        assert queued[0]["text"] == "Background review: found a bug"

    @pytest.mark.asyncio
    async def test_a_hook_with_no_output_and_no_rewake_message_queues_nothing(self):
        state_module.save_session_state(1, workspace_root=".")
        hook = HookDefinition(
            event="post_tool_use", matcher="", command="true",
            source="user config", async_rewake=True,
        )
        await run_tool_hooks(
            [hook], "post_tool_use", tool_name="write_file", tool_input={},
            tool_output={"success": True}, session_id=1, workspace_root=".",
        )
        await asyncio.sleep(0.3)
        assert state_module.get_session_state(1).queued_user_instructions == []

    @pytest.mark.asyncio
    async def test_a_synchronous_hook_alongside_a_rewake_hook_still_returns_normally(self):
        state_module.save_session_state(1, workspace_root=".")
        hooks = [
            HookDefinition(event="post_tool_use", matcher="", command='echo "sync message" 1>&2', source="user config"),
            HookDefinition(event="post_tool_use", matcher="", command="sleep 0.3; echo done", source="project config", async_rewake=True),
        ]
        try:
            results = await run_tool_hooks(
                hooks, "post_tool_use", tool_name="write_file", tool_input={},
                tool_output={"success": True}, session_id=1, workspace_root=".",
            )
            assert len(results) == 1
            assert results[0].message == "sync message"
        finally:
            await self._drain()
