"""Coverage for interactive.py's standalone mode (client=None) -- the
interactive REPL calling a provider directly instead of the TamfisGPT
Remote Workspace backend. Uses the same run_interactive test harness as
test_tamfis_code_repl_exit.py (mock PromptSession, feed scripted inputs,
capture everything printed), with real local-state isolation added since
these tests exercise state-backed slash commands (unlike the two exit-path
tests in that file, which never touch state.json).
"""
import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from prompt_toolkit.document import Document

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.interactive import PASTE_COLLAPSE_LINE_THRESHOLD, SLASH_COMMANDS, _SlashCommandCompleter, paste_placeholder
from tamfis_code.runner import TaskOutcome
from tamfis_code.workspace import WorkspaceContext


class PastePlaceholderTests(unittest.TestCase):
    """Live-reported: pasting a long clipboard block into the interactive
    prompt inserted the whole raw text into the input line instead of
    collapsing to a placeholder the way Claude Code/Codex do -- there was
    no such collapsing logic anywhere in this codebase at all before this."""

    def test_short_paste_is_not_collapsed(self):
        text = "\n".join(f"line {i}" for i in range(PASTE_COLLAPSE_LINE_THRESHOLD))
        self.assertIsNone(paste_placeholder(text, 1))

    def test_paste_over_the_threshold_collapses_with_a_line_count(self):
        text = "\n".join(f"line {i}" for i in range(86)) + "\n"
        result = paste_placeholder(text, 1)
        self.assertIsNotNone(result)
        placeholder, normalized = result
        self.assertEqual(placeholder, "[Pasted text #1 +86 lines]")
        self.assertEqual(normalized, text)

    def test_placeholder_count_is_per_call_not_hardcoded(self):
        text = "\n".join(f"line {i}" for i in range(10))
        placeholder, _ = paste_placeholder(text, 3)
        self.assertTrue(placeholder.startswith("[Pasted text #3"))

    def test_crlf_and_cr_line_endings_are_normalized_like_the_default_binding(self):
        # Matches prompt_toolkit's own default Keys.BracketedPaste handler
        # (some terminals, e.g. iTerm2, paste \r\n line endings).
        text = "a\r\nb\r\nc\r\nd\r\n"
        placeholder, normalized = paste_placeholder(text, 1)
        self.assertNotIn("\r", normalized)
        self.assertEqual(normalized, "a\nb\nc\nd\n")

    def test_single_line_paste_with_no_trailing_newline_is_not_collapsed(self):
        self.assertIsNone(paste_placeholder("just one long line, no newline at all", 1))

    def test_empty_paste_returns_none(self):
        self.assertIsNone(paste_placeholder("", 1))

    def test_a_pasted_line_count_exactly_at_the_threshold_is_not_collapsed(self):
        text = "".join(f"line {i}\n" for i in range(PASTE_COLLAPSE_LINE_THRESHOLD))
        self.assertEqual(text.count("\n"), PASTE_COLLAPSE_LINE_THRESHOLD)
        self.assertIsNone(paste_placeholder(text, 1))

    def test_one_more_line_than_the_threshold_does_collapse(self):
        text = "\n".join(f"line {i}" for i in range(PASTE_COLLAPSE_LINE_THRESHOLD + 2))
        self.assertIsNotNone(paste_placeholder(text, 1))


class SlashCommandCompleterTests(unittest.TestCase):
    """Before this, PromptSession had no completer= at all -- Tab did
    nothing while typing a slash-command or an ordinary objective."""

    def setUp(self):
        self.completer = _SlashCommandCompleter()

    def _complete(self, text: str) -> list[str]:
        doc = Document(text=text, cursor_position=len(text))
        return [c.text for c in self.completer.get_completions(doc, None)]

    def test_partial_slash_command_completes_to_matching_names(self):
        results = self._complete("/mo")
        self.assertIn("/mode", results)
        self.assertIn("/model", results)
        self.assertNotIn("/status", results)

    def test_bare_slash_offers_every_command(self):
        results = self._complete("/")
        self.assertEqual(set(results), {name for name, _ in SLASH_COMMANDS})


class SlashCommandCompleterCustomCommandsTests(unittest.TestCase):
    """The completer's custom_commands dict is mutated in place by the REPL
    loop every turn (not reconstructed) -- these lock in that live-update
    contract and that custom commands never shadow a built-in name."""

    def setUp(self):
        from tamfis_code.custom_commands import CustomCommand
        self.commands: dict = {}
        self.completer = _SlashCommandCompleter(self.commands)
        self.CustomCommand = CustomCommand

    def _complete(self, text: str) -> list[str]:
        doc = Document(text=text, cursor_position=len(text))
        return [c.text for c in self.completer.get_completions(doc, None)]

    def test_custom_command_appears_in_completions(self):
        self.commands["review"] = self.CustomCommand(name="review", description="review a diff", template="x", source="user config")
        results = self._complete("/rev")
        self.assertIn("/review", results)

    def test_mutating_the_dict_in_place_updates_completions_without_rebuilding(self):
        self.assertEqual(self._complete("/xyz"), [])
        self.commands["xyzcmd"] = self.CustomCommand(name="xyzcmd", description="", template="x", source="user config")
        self.assertIn("/xyzcmd", self._complete("/xyz"))

    def test_custom_command_never_shadows_a_built_in_name(self):
        self.commands["plan"] = self.CustomCommand(name="plan", description="custom plan", template="x", source="user config")
        results = self._complete("/plan")
        self.assertEqual(results.count("/plan"), 1)

    def test_no_completions_for_natural_language_text(self):
        self.assertEqual(self._complete("fix the bug in calc.py"), [])

    def test_no_completions_once_a_space_follows_the_command(self):
        # A command with an argument already being typed (e.g. "/model ")
        # must not keep suggesting command names.
        self.assertEqual(self._complete("/model auto"), [])

    def test_completion_replaces_the_whole_partial_token(self):
        doc = Document(text="/mo", cursor_position=3)
        completions = list(self.completer.get_completions(doc, None))
        mode_completion = next(c for c in completions if c.text == "/mode")
        self.assertEqual(mode_completion.start_position, -3)


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


def _run(scripted_inputs, *, session_id=1, workspace_root="/tmp/fake-workspace", config=None):
    buf = io.StringIO()
    fake_console = Console(file=buf, no_color=True, width=200)
    workspace = WorkspaceContext(session_id=session_id, workspace_root=workspace_root)
    config = config or Config()
    prompt_mock = AsyncMock(side_effect=scripted_inputs)

    with patch("tamfis_code.interactive.Console", return_value=fake_console), \
            patch("tamfis_code.interactive.PromptSession") as session_cls, \
            patch("tamfis_code.interactive.print_banner"):
        session_cls.return_value.prompt_async = prompt_mock
        asyncio.run(run_interactive_import(client=None, config=config, workspace=workspace))

    return buf.getvalue()


def run_interactive_import(**kwargs):
    from tamfis_code.interactive import run_interactive
    return run_interactive(**kwargs)


class StandaloneStatusAndToolsTests(_StatePatchMixin, unittest.TestCase):
    def test_status_shows_standalone_not_server_id(self):
        output = _run(["/status", EOFError()])
        self.assertIn("standalone, local session", output)
        self.assertNotIn("server_id=", output)

    def test_tools_shows_real_local_tool_names(self):
        output = _run(["/tools", EOFError()])
        self.assertIn("edit_file", output)
        self.assertIn("execute_command", output)
        self.assertNotIn("glob_files", output)  # old remote-only tool name

    def test_help_mentions_standalone_limitations(self):
        output = _run(["/help", EOFError()])
        self.assertIn("Standalone mode", output)
        self.assertIn("/pty", output)


class StandaloneDiffsAndRevertTests(_StatePatchMixin, unittest.TestCase):
    def test_diffs_empty_session_prints_dim_message(self):
        output = _run(["/diffs", EOFError()])
        self.assertIn("No file mutations recorded", output)

    def test_diffs_and_diff_and_revert_use_local_ledger(self):
        with tempfile.TemporaryDirectory() as ws:
            target = Path(ws) / "app.py"
            target.write_text("new\n")
            from tamfis_code.safety import record_mutation
            entry = record_mutation(1, path=str(target), operation="update", original_content="old\n", new_content="new\n")

            output = _run(["/diffs", f"/diff {entry['mutation_id']}", f"/revert {entry['mutation_id']}", EOFError()], workspace_root=ws)
            self.assertIn(entry["mutation_id"], output)
            self.assertIn("Reverted", output)
            self.assertEqual(target.read_text(), "old\n")

    def test_revert_unknown_mutation_id_shows_error_not_crash(self):
        output = _run(["/revert mut_doesnotexist", EOFError()])
        self.assertIn("No recorded mutation", output)

    def test_revert_turn_id_reverts_every_mutation_in_that_turn(self):
        with tempfile.TemporaryDirectory() as ws:
            a, b = Path(ws) / "a.py", Path(ws) / "b.py"
            a.write_text("new a\n")
            b.write_text("new b\n")
            from tamfis_code.safety import record_mutation
            record_mutation(1, path=str(a), operation="update", original_content="old a\n", new_content="new a\n", transaction_id="turn_repl1")
            record_mutation(1, path=str(b), operation="update", original_content="old b\n", new_content="new b\n", transaction_id="turn_repl1")

            output = _run(["/revert turn_repl1", EOFError()], workspace_root=ws)
            self.assertIn("Reverted 2 mutation(s)", output)
            self.assertEqual(a.read_text(), "old a\n")
            self.assertEqual(b.read_text(), "old b\n")

    def test_revert_unknown_turn_id_shows_error_not_crash(self):
        output = _run(["/revert turn_doesnotexist", EOFError()])
        self.assertIn("No recorded mutations", output)


class StandaloneAgentsAndResumeTests(_StatePatchMixin, unittest.TestCase):
    def test_agents_lists_local_sessions_not_remote_call(self):
        state_module.save_session_state(1, workspace_root="/tmp/fake-workspace")
        state_module.save_session_state(2, workspace_root="/tmp/other-project")
        output = _run(["/agents", EOFError()])
        self.assertIn("/tmp/fake-workspace", output)
        self.assertIn("/tmp/other-project", output)

    def test_agents_hides_swarm_child_sessions_by_default(self):
        state_module.save_session_state(1, workspace_root="/tmp/fake-workspace")
        state_module.save_session_state(2, workspace_root="/tmp/fake-workspace", is_swarm_child=True, parent_session_id=1, swarm_label="fix bug a")
        output = _run(["/agents", EOFError()])
        self.assertNotIn("fix bug a", output)
        output_all = _run(["/agents --all", EOFError()])
        self.assertIn("fix bug a", output_all)
        self.assertIn("2  /tmp/fake-workspace", output_all)

    def test_agents_hides_swarm_child_with_no_known_parent(self):
        # Live-caught regression: a swarm child minted with no pre-existing
        # parent session (e.g. via agent-cmd delegate, a one-shot CLI
        # invocation) has parent_session_id=None -- indistinguishable from
        # an ordinary session if that were the hide/show filter. is_swarm_child
        # is the actual marker, set unconditionally regardless of whether a
        # real parent was known.
        state_module.save_session_state(3, workspace_root="/tmp/fake-workspace", is_swarm_child=True, parent_session_id=None, swarm_label="no known parent")
        output = _run(["/agents", EOFError()])
        self.assertNotIn("no known parent", output)
        output_all = _run(["/agents --all", EOFError()])
        self.assertIn("no known parent", output_all)

    def test_resume_with_no_other_sessions(self):
        output = _run(["/resume", EOFError()])
        self.assertIn("No other sessions to resume", output)

    def test_resume_unknown_session_id_shows_error(self):
        output = _run(["/resume 999", EOFError()])
        self.assertIn("No known local session 999", output)

    def test_resume_shows_incomplete_plan_step_progress(self):
        # Before this, /resume showed only a conversation summary -- a plan
        # left mid-execution (some steps done, one in_progress, one still
        # pending) was completely invisible once resumed.
        state_module.save_session_state(
            2, workspace_root="/tmp/other-project",
            saved_plans=[{
                "id": "plan_resume1", "objective": "fix the login bug",
                "steps": [
                    {"index": 0, "step": "Read auth.py", "status": "completed"},
                    {"index": 1, "step": "Fix the off-by-one bug", "status": "in_progress"},
                    {"index": 2, "step": "Run the test suite", "status": "pending"},
                ],
            }],
            active_plan_id="plan_resume1",
        )
        output = _run(["/resume 2", EOFError()])
        self.assertIn("Plan in progress", output)
        self.assertIn("Read auth.py", output)
        self.assertIn("Fix the off-by-one bug", output)
        self.assertIn("Run the test suite", output)

    def test_resume_with_fully_completed_plan_shows_no_plan_status(self):
        state_module.save_session_state(
            2, workspace_root="/tmp/other-project",
            saved_plans=[{
                "id": "plan_done1", "objective": "fix the login bug",
                "steps": [{"index": 0, "step": "Fix it", "status": "completed"}],
            }],
            active_plan_id="plan_done1",
        )
        output = _run(["/resume 2", EOFError()])
        self.assertNotIn("Plan in progress", output)


class StandaloneDelegateTests(_StatePatchMixin, unittest.TestCase):
    """/delegate had zero dedicated test coverage before this -- only
    AgentManager.execute_tasks itself (tests/test_agents_delegation.py) was
    tested, not the REPL command that dispatches to it."""

    def test_delegate_disabled_by_default_shows_error(self):
        output = _run(["/delegate fix the bug", EOFError()])
        self.assertIn("Subagent delegation is disabled", output)
        self.assertIn("enable_subagent_delegation", output)

    def test_delegate_with_no_objective_shows_usage(self):
        cfg = Config()
        cfg.enable_subagent_delegation = True
        output = _run(["/delegate", EOFError()], config=cfg)
        self.assertIn("Usage: /delegate", output)

    def test_delegate_splits_pipe_separated_objectives_and_reports_results(self):
        cfg = Config()
        cfg.enable_subagent_delegation = True

        captured = {}

        async def fake_execute_tasks(self, descriptions, **kwargs):
            captured["descriptions"] = descriptions
            captured["kwargs"] = kwargs
            return [
                {"task_id": "t1", "description": descriptions[0], "status": "completed", "result": {"summary": "did thing one"}},
                {"task_id": "t2", "description": descriptions[1], "status": "failed", "result": {"error": "boom"}},
            ]

        with patch("tamfis_code.agents.AgentManager.execute_tasks", new=fake_execute_tasks):
            output = _run(["/delegate fix bug a | write tests for b", EOFError()], config=cfg)

        self.assertEqual(captured["descriptions"], ["fix bug a", "write tests for b"])
        self.assertEqual(captured["kwargs"]["workspace_root"], "/tmp/fake-workspace")
        self.assertEqual(captured["kwargs"]["approval_policy"], cfg.approval_policy)
        self.assertIn("✅ fix bug a", output)
        self.assertIn("did thing one", output)
        self.assertIn("❌ write tests for b", output)
        self.assertIn("boom", output)


class StandaloneSwarmTests(_StatePatchMixin, unittest.TestCase):
    """/swarm mirrors /delegate's dispatch shape but goes through
    swarm.run_swarm (aggregate status display + mutation gate) instead of
    calling AgentManager.execute_tasks directly."""

    def test_swarm_disabled_by_default_shows_error(self):
        output = _run(["/swarm look into this", EOFError()])
        self.assertIn("Subagent delegation is disabled", output)

    def test_swarm_with_no_objective_shows_usage(self):
        cfg = Config()
        cfg.enable_subagent_delegation = True
        output = _run(["/swarm", EOFError()], config=cfg)
        self.assertIn("Usage: /swarm", output)

    def test_swarm_splits_pipe_separated_objectives_and_reports_results(self):
        cfg = Config()
        cfg.enable_subagent_delegation = True

        captured = {}

        async def fake_run_swarm(descriptions, **kwargs):
            captured["descriptions"] = descriptions
            captured["kwargs"] = kwargs
            return [
                {"task_id": "t1", "description": descriptions[0], "status": "completed", "result": {"summary": "did thing one"}},
                {"task_id": "t2", "description": descriptions[1], "status": "failed", "result": {"error": "boom"}},
            ]

        with patch("tamfis_code.swarm.run_swarm", new=fake_run_swarm):
            output = _run(["/swarm look at a | audit b", EOFError()], config=cfg)

        self.assertEqual(captured["descriptions"], ["look at a", "audit b"])
        self.assertEqual(captured["kwargs"]["workspace_root"], "/tmp/fake-workspace")
        self.assertEqual(captured["kwargs"]["session_id"], 1)
        self.assertFalse(captured["kwargs"]["mutate"])
        self.assertIn("✅ look at a", output)
        self.assertIn("did thing one", output)
        self.assertIn("❌ audit b", output)
        self.assertIn("boom", output)

    def test_swarm_mutate_flag_is_parsed_and_stripped_from_objectives(self):
        cfg = Config()
        cfg.enable_subagent_delegation = True
        captured = {}

        async def fake_run_swarm(descriptions, **kwargs):
            captured["descriptions"] = descriptions
            captured["mutate"] = kwargs["mutate"]
            return []

        with patch("tamfis_code.swarm.run_swarm", new=fake_run_swarm):
            _run(["/swarm fix the bug --mutate", EOFError()], config=cfg)

        self.assertEqual(captured["descriptions"], ["fix the bug"])
        self.assertTrue(captured["mutate"])

    def test_swarm_surfaces_mutation_gate_refusal_as_an_error_not_a_crash(self):
        cfg = Config()
        cfg.enable_subagent_delegation = True

        async def fake_run_swarm(descriptions, **kwargs):
            raise ValueError("Swarm sub-tasks run non-interactively and cannot prompt for approval")

        with patch("tamfis_code.swarm.run_swarm", new=fake_run_swarm):
            output = _run(["/swarm fix the bug --mutate", EOFError()], config=cfg)

        self.assertIn("cannot prompt for approval", output)


class ModeCommandNeverPolicyDisambiguationTests(_StatePatchMixin, unittest.TestCase):
    """/mode's help text used to describe "auto" as "never prompt" right
    next to a completely unrelated policy literally named "never" (which
    denies everything) -- a real naming footgun, now disambiguated."""

    def test_bare_mode_help_disambiguates_never_from_auto(self):
        output = _run(["/mode", EOFError()])
        self.assertIn("never prompts", output.replace("\n", " "))
        self.assertIn("DENY everything", output)
        self.assertIn("not a synonym for 'auto'", output)

    def test_unknown_mode_error_mentions_never_is_a_real_raw_value(self):
        output = _run(["/mode not-a-real-mode", EOFError()])
        self.assertIn("never", output)

    def test_mode_never_is_still_reachable_directly_and_denies(self):
        output = _run(["/mode never", "/mode", EOFError()])
        self.assertIn("Mode set to", output)
        self.assertIn("Current mode:", output)
        self.assertIn("(never)", output)


class StandalonePtyAndDoctorTests(_StatePatchMixin, unittest.TestCase):
    def test_pty_is_available_in_standalone_mode(self):
        with tempfile.TemporaryDirectory() as workspace_root:
            output = _run(["/pty start", EOFError()], workspace_root=workspace_root)
        self.assertIn("Started local background terminal", output)

    def test_doctor_shows_provider_status_table(self):
        output = _run(["/doctor", EOFError()])
        self.assertIn("PROVIDER", output)
        self.assertIn("CONFIGURED", output)


class StandaloneAiDispatchTests(_StatePatchMixin, unittest.TestCase):
    def test_natural_language_objective_calls_local_agent_loop(self):
        fake_outcome = TaskOutcome(status="completed", summary="Done.")
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(return_value=fake_outcome)):
            output = _run(["fix the bug", EOFError()])
        self.assertIn("Done.", output)

    def test_retry_with_no_previous_turn(self):
        output = _run(["/retry", EOFError()])
        self.assertIn("No previous turn", output)

    def test_retry_resends_last_objective(self):
        fake_outcome = TaskOutcome(status="completed", summary="Done again.")
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(return_value=fake_outcome)) as mock_turn:
            output = _run(["fix the bug", "/retry", EOFError()])
        self.assertEqual(mock_turn.call_count, 2)
        self.assertIn("Done again.", output)

    def test_plan_mode_saves_plan_on_completion(self):
        fake_outcome = TaskOutcome(status="completed", summary="# Plan\n1. Do X")
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(return_value=fake_outcome)):
            output = _run(["/plan add a feature", EOFError()])
        self.assertIn("Plan saved", output)
        state = state_module.get_session_state(1)
        self.assertEqual(len(state.saved_plans), 1)

    def test_second_turn_includes_the_first_turns_conversation_in_messages(self):
        """Confirmed live: every standalone turn used to be built as a
        fresh single-element `[{"role": "user", "content": objective}]`,
        with no prior turn ever attached -- the model had no memory of
        anything said earlier in the same interactive session. A follow-up
        like "yes" (expanded by contextualize_short_reply into "Yes.
        Proceed with the action or next step you just proposed.") was sent
        as the ENTIRE conversation, referring to a proposal the model had
        never actually seen."""
        fake_outcome = TaskOutcome(status="completed", summary="I found the bug in calc.py.")
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(return_value=fake_outcome)) as mock_turn:
            _run(["find the bug", "now fix it", EOFError()])

        self.assertEqual(mock_turn.call_count, 2)
        first_messages = mock_turn.call_args_list[0].args[3]
        second_messages = mock_turn.call_args_list[1].args[3]
        self.assertEqual(first_messages, [{"role": "user", "content": "find the bug"}])
        self.assertEqual(second_messages, [
            {"role": "user", "content": "find the bug"},
            {"role": "assistant", "content": "I found the bug in calc.py."},
            {"role": "user", "content": "now fix it"},
        ])

    def test_failed_turn_still_records_its_objective_without_a_missing_answer(self):
        with patch(
            "tamfis_code.interactive.run_local_agent_turn",
            new=AsyncMock(side_effect=[
                TaskOutcome(status="failed", error="boom"),
                TaskOutcome(status="completed", summary="Done now."),
            ]),
        ) as mock_turn:
            _run(["do the risky thing", "try again", EOFError()])

        second_messages = mock_turn.call_args_list[1].args[3]
        self.assertEqual(second_messages, [
            {"role": "user", "content": "do the risky thing"},
            {"role": "user", "content": "try again"},
        ])

    def test_provider_error_does_not_crash_the_repl(self):
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(side_effect=RuntimeError("boom"))):
            output = _run(["fix the bug", "/status", EOFError()])
        self.assertIn("boom", output)
        self.assertIn("standalone, local session", output)  # REPL kept running after the error


class UnsupportedProviderTests(_StatePatchMixin, unittest.TestCase):
    def test_invalid_provider_reported_cleanly_not_a_traceback(self):
        buf = io.StringIO()
        fake_console = Console(file=buf, no_color=True, width=200)
        workspace = WorkspaceContext(session_id=1, workspace_root="/tmp/fake-workspace")
        config = Config()
        with patch("tamfis_code.interactive.Console", return_value=fake_console), \
                patch("tamfis_code.interactive.print_banner"):
            from tamfis_code.interactive import run_interactive
            asyncio.run(run_interactive(client=None, config=config, workspace=workspace, provider="not-a-real-provider"))
        self.assertIn("Unknown local provider", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
