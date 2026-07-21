import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from tamfis_code import config as config_module
from tamfis_code import state as state_module
from tamfis_code.cli import (
    _explicit_absolute_paths,
    _print_bg_hint,
    _project_root_for_target,
    _session_for_primary,
    _use_remote,
    cli,
)
from tamfis_code.config import Config


class ExplicitAbsolutePathsTests(unittest.TestCase):
    def test_extracts_a_single_absolute_path(self):
        self.assertEqual(
            _explicit_absolute_paths("please fix /home/user/project/app.py now"),
            [Path("/home/user/project/app.py")],
        )

    def test_extracts_multiple_paths_and_strips_trailing_punctuation(self):
        result = _explicit_absolute_paths("see /a/b.py, and also /c/d.py.")
        self.assertEqual(result, [Path("/a/b.py"), Path("/c/d.py")])

    def test_no_absolute_paths_returns_empty_list(self):
        self.assertEqual(_explicit_absolute_paths("just fix the bug please"), [])

    def test_relative_looking_path_is_not_matched(self):
        self.assertEqual(_explicit_absolute_paths("edit src/app.py"), [])


class ProjectRootForTargetTests(unittest.TestCase):
    def test_finds_git_root_from_nested_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            target = nested / "file.py"
            target.write_text("x")
            self.assertEqual(_project_root_for_target(target), root.resolve())

    def test_no_git_root_falls_back_to_containing_directory(self):
        # Patched rather than relying on the real filesystem: some sandboxes
        # (this one included) have a stray .git under /tmp itself, which
        # would otherwise make the walk-up-to-find-.git logic under test
        # find a false positive several directories above the temp root.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "file.py"
            target.write_text("x")
            with patch.object(Path, "exists", return_value=False):
                self.assertEqual(_project_root_for_target(target), root.resolve())

    def test_directory_target_uses_itself_as_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            self.assertEqual(_project_root_for_target(root), root.resolve())


class SessionForPrimaryTests(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()

    def test_returns_none_when_no_session_known_for_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_session_for_primary(Path(tmp)))

    def test_finds_session_by_primary_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_module.save_session_state(42, workspace_root=str(root.resolve()))
            state_module.save_session_state(42, primary_workspace=str(root.resolve()))
            self.assertEqual(_session_for_primary(root), 42)

    def test_most_recently_known_session_wins_when_multiple_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_module.save_session_state(1, workspace_root=str(root.resolve()))
            state_module.save_session_state(2, workspace_root=str(root.resolve()))
            self.assertEqual(_session_for_primary(root), 2)


class PrintBgHintTests(unittest.TestCase):
    def test_prints_session_and_task_hints(self):
        from io import StringIO
        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        _print_bg_hint(console, 7, "task-abc")
        output = buf.getvalue()
        self.assertIn("session 7", output)
        self.assertIn("task-abc", output)
        self.assertIn("tamfis-code attach 7", output)


class _CliConfigIsolationMixin:
    """Redirects config/credentials/state to a temp dir so CliRunner
    invocations never touch the real ~/.config/tamfis-code/."""

    def setUp(self):
        self._config_originals = (
            config_module.CONFIG_DIR, config_module.CREDENTIALS_PATH, config_module.USER_CONFIG_PATH,
        )
        self._state_originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)
        config_module.CONFIG_DIR = tmp_path
        config_module.CREDENTIALS_PATH = tmp_path / "credentials.json"
        config_module.USER_CONFIG_PATH = tmp_path / "config.toml"
        state_module.CONFIG_DIR = tmp_path / "state"
        state_module.STATE_PATH = tmp_path / "state" / "state.json"
        self._env_token = os.environ.pop("TAMFIS_CODE_TOKEN", None)
        self.runner = CliRunner()

    def tearDown(self):
        (config_module.CONFIG_DIR, config_module.CREDENTIALS_PATH,
         config_module.USER_CONFIG_PATH) = self._config_originals
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._state_originals
        self.tmp.cleanup()
        if self._env_token is not None:
            os.environ["TAMFIS_CODE_TOKEN"] = self._env_token


class LoginCommandTests(_CliConfigIsolationMixin, unittest.TestCase):
    def test_login_with_existing_token_succeeds(self):
        fake_client = AsyncMock()
        fake_client.me = AsyncMock(return_value={
            "authenticated": True,
            "user": {"id": "u1", "email": "dev@example.com"},
        })
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=False)
        with patch("tamfis_code.cli.RemoteAPIClient", return_value=fake_client):
            result = self.runner.invoke(cli, ["login", "--token", "sometoken123"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Logged in", result.output)
        self.assertIn("dev@example.com", result.output)

    def test_login_with_invalid_token_exits_with_auth_failure(self):
        from tamfis_code.api_client import RemoteAPIError

        fake_client = AsyncMock()
        fake_client.me = AsyncMock(side_effect=RemoteAPIError(401, "bad token"))
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=False)
        with patch("tamfis_code.cli.RemoteAPIClient", return_value=fake_client):
            result = self.runner.invoke(cli, ["login", "--token", "bad"])
        self.assertNotEqual(result.exit_code, 0)


class LogoutCommandTests(_CliConfigIsolationMixin, unittest.TestCase):
    def test_logout_without_prior_login_reports_not_logged_in(self):
        result = self.runner.invoke(cli, ["logout"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Not logged in", result.output)

    def test_logout_after_login_clears_credentials(self):
        from tamfis_code.config import Credentials
        from tamfis_code.api_client import save_secure_credentials

        save_secure_credentials(Credentials(access_token="tok", user_id="u1", email="dev@example.com"))
        fake_client = AsyncMock()
        fake_client.logout = AsyncMock(return_value=None)
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=False)
        with patch("tamfis_code.cli.RemoteAPIClient", return_value=fake_client):
            result = self.runner.invoke(cli, ["logout"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Logged out", result.output)


class UseRemoteHelperTests(unittest.TestCase):
    """The --remote flag always wins; otherwise a paid tenant's persistent
    config.toml `default_backend = "remote"` makes every command use the
    legacy backend without needing --remote on each invocation."""

    def test_flag_true_is_remote_regardless_of_config(self):
        self.assertTrue(_use_remote(Config(default_backend="standalone"), True))

    def test_flag_false_defers_to_config_standalone(self):
        self.assertFalse(_use_remote(Config(default_backend="standalone"), False))

    def test_flag_false_defers_to_config_remote(self):
        self.assertTrue(_use_remote(Config(default_backend="remote"), False))


class WorkspaceGroupCommandTests(_CliConfigIsolationMixin, unittest.TestCase):
    def test_workspace_list_without_known_session_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "workspace", "list"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No known session", result.output)

    def test_workspace_add_and_list_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "primary"
            extra = Path(tmp) / "extra"
            primary.mkdir()
            extra.mkdir()
            state_module.save_session_state(
                1, workspace_root=str(primary.resolve()), primary_workspace=str(primary.resolve()),
                current_working_directory=str(primary.resolve()),
            )
            result = self.runner.invoke(cli, ["--cwd", str(primary), "workspace", "add", str(extra)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("approved", result.output)

            result = self.runner.invoke(cli, ["--cwd", str(primary), "workspace", "list"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn(str(extra.resolve()), result.output)

    def test_workspace_remove_rejects_primary_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp)
            state_module.save_session_state(
                2, workspace_root=str(primary.resolve()), primary_workspace=str(primary.resolve()),
            )
            result = self.runner.invoke(cli, ["--cwd", str(primary), "workspace", "remove", str(primary)])
        self.assertNotEqual(result.exit_code, 0)


class ConfigCommandTests(_CliConfigIsolationMixin, unittest.TestCase):
    def test_config_command_prints_a_table_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "config"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("credential_storage", result.output)


class StandaloneDefaultDispatchTests(_CliConfigIsolationMixin, unittest.TestCase):
    """ask/agent/exec/chat/audit/plan default to the standalone local loop
    (runner_local.py) now -- --remote is required to reach the legacy
    RemoteAPIClient path, and --bg (server-side background execution) only
    makes sense with --remote since a standalone process has no server to
    keep a task alive after it exits."""

    def test_bg_without_remote_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "ask", "do something", "--bg"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--bg requires --remote", result.output)

    def test_execute_plan_bg_without_remote_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "execute-plan", "--bg"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--bg requires --remote", result.output)

    def test_ask_without_remote_never_constructs_a_remote_client(self):
        from tamfis_code.runner import TaskOutcome

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli.RemoteAPIClient") as fake_remote_client, \
                    patch("tamfis_code.runner_local.run_local_agent_turn", new=AsyncMock(return_value=TaskOutcome(status="completed", summary="hi"))):
                result = self.runner.invoke(cli, ["--cwd", tmp, "chat", "hello", "--provider", "nvidia"])
            self.assertEqual(result.exit_code, 0, result.output)
            fake_remote_client.assert_not_called()

    def test_standalone_chat_is_interactive_when_stdin_is_a_real_tty(self):
        """Confirmed live: `interactive` was hardcoded False for every
        one-shot command (agent/ask/chat/audit/plan/run/retry), regardless
        of whether a real human was sitting at an attached terminal -- the
        default "ask" approval policy's (y)es/(n)o/(a)lways prompt could
        never appear at all, no matter what; every risky action was
        silently denied instead. It must now follow sys.stdin.isatty().

        Calls _run_local_ai_command directly rather than through
        CliRunner.invoke() -- CliRunner replaces sys.stdin with its own
        captured-output stream for the duration of invoke(), which would
        make a patched sys.stdin.isatty() irrelevant by the time the
        command body actually runs.
        """
        from tamfis_code.cli import _run_local_ai_command
        from tamfis_code.runner import TaskOutcome

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli.sys.stdin.isatty", return_value=True), \
                    patch(
                        "tamfis_code.runner_local.run_local_agent_turn",
                        new=AsyncMock(return_value=TaskOutcome(status="completed", summary="hi")),
                    ) as fake_turn:
                exit_code = asyncio.run(
                    _run_local_ai_command(Config(), Path(tmp), "hello", "chat", "auto", "nvidia", ())
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(fake_turn.call_args.kwargs.get("interactive"))

    def test_standalone_chat_stays_non_interactive_when_stdin_is_piped(self):
        from tamfis_code.cli import _run_local_ai_command
        from tamfis_code.runner import TaskOutcome

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli.sys.stdin.isatty", return_value=False), \
                    patch(
                        "tamfis_code.runner_local.run_local_agent_turn",
                        new=AsyncMock(return_value=TaskOutcome(status="completed", summary="hi")),
                    ) as fake_turn:
                exit_code = asyncio.run(
                    _run_local_ai_command(Config(), Path(tmp), "hello", "chat", "auto", "nvidia", ())
                )
            self.assertEqual(exit_code, 0)
            self.assertFalse(fake_turn.call_args.kwargs.get("interactive"))

    def test_attach_is_passed_to_standalone_turn_as_an_explicit_read_only_input(self):
        from tamfis_code.runner import TaskOutcome

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.txt"
            probe.write_text("x")
            with patch(
                "tamfis_code.runner_local.run_local_agent_turn",
                new=AsyncMock(return_value=TaskOutcome(status="completed", summary="inspected")),
            ) as fake_turn:
                result = self.runner.invoke(
                    cli, ["--cwd", tmp, "ask", "look at this", "--attach", str(probe), "--provider", "nvidia"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(fake_turn.call_args.kwargs["attachment_paths"], (str(probe.resolve()),))
        messages = fake_turn.call_args.args[3]
        self.assertIn(str(probe.resolve()), messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "look at this"})

    def test_default_backend_remote_in_config_uses_remote_without_the_flag(self):
        # A paid TamfisGPT tenant sets this once instead of typing --remote
        # on every single command.
        config_module.USER_CONFIG_PATH.write_text('default_backend = "remote"\n')
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli._run_ai_command", new=AsyncMock(return_value=0)) as fake_remote_run, \
                    patch("tamfis_code.cli._run_local_ai_command", new=AsyncMock(return_value=0)) as fake_local_run:
                self.runner.invoke(cli, ["--cwd", tmp, "ask", "do something"])
            fake_remote_run.assert_called_once()
            fake_local_run.assert_not_called()


class AgentCmdDelegateTests(_CliConfigIsolationMixin, unittest.TestCase):
    """agent-cmd delegate had zero dedicated CLI test coverage before this
    -- only AgentManager.execute_tasks itself (tests/test_agents_delegation.py)
    was tested, not the CLI command that dispatches to it."""

    def test_delegate_disabled_by_default_exits_with_invalid_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "agent-cmd", "delegate", "--task", "fix the bug"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Subagent delegation is disabled", result.output)

    def test_delegate_with_no_task_shows_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(
                cli, ["--cwd", tmp, "agent-cmd", "delegate"],
                env={"TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION": "1"},
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Please specify at least one --task", result.output)

    def test_delegate_reports_per_task_results(self):
        captured = {}

        async def fake_execute_tasks(self, descriptions, **kwargs):
            captured["descriptions"] = descriptions
            captured["max_concurrency"] = kwargs.get("max_concurrency")
            return [
                {"task_id": "t1", "description": descriptions[0], "status": "completed", "result": {"summary": "did thing one"}},
                {"task_id": "t2", "description": descriptions[1], "status": "failed", "result": {"error": "boom"}},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.agents.AgentManager.execute_tasks", new=fake_execute_tasks):
                result = self.runner.invoke(
                    cli,
                    ["--cwd", tmp, "agent-cmd", "delegate", "--task", "fix bug a", "--task", "write tests for b", "--max-concurrency", "2"],
                    env={"TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION": "1"},
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(captured["descriptions"], ["fix bug a", "write tests for b"])
        self.assertEqual(captured["max_concurrency"], 2)
        self.assertIn("✅ fix bug a", result.output)
        self.assertIn("did thing one", result.output)
        self.assertIn("❌ write tests for b", result.output)
        self.assertIn("boom", result.output)


class AgentCommandNamingCollisionHelpTests(_CliConfigIsolationMixin, unittest.TestCase):
    """`agent-cmd` and `agents` are similarly-named but functionally
    distinct commands -- a real naming footgun, not fixed by renaming
    (would break scripts/muscle memory) but by making each command's
    --help cross-reference the other so the distinction is discoverable."""

    def test_agent_cmd_help_mentions_agents(self):
        result = self.runner.invoke(cli, ["agent-cmd", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("tamfis-code agents", result.output)

    def test_agents_help_mentions_agent_cmd(self):
        result = self.runner.invoke(cli, ["agents", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("tamfis-code agent-cmd", result.output)


class StandaloneInfoCommandTests(_CliConfigIsolationMixin, unittest.TestCase):
    """init/doctor/status/sessions/diffs/diff/revert/agents/run all default
    to a local implementation now (state.py / safety.py-backed), with
    --remote as the explicit opt-out to the legacy backend."""

    def test_init_creates_a_local_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "init"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("standalone, local session", result.output)

    def test_status_shows_local_session_without_remote_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli.RemoteAPIClient") as fake_client:
                result = self.runner.invoke(cli, ["--cwd", tmp, "status"])
            fake_client.assert_not_called()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("standalone, local session", result.output)

    def test_sessions_lists_known_local_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_module.save_session_state(5, workspace_root=str(Path(tmp).resolve()))
            result = self.runner.invoke(cli, ["--cwd", tmp, "sessions"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(str(Path(tmp).resolve()), result.output)

    def test_sessions_hides_swarm_child_sessions_unless_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp).resolve())
            state_module.save_session_state(5, workspace_root=root)
            state_module.save_session_state(6, workspace_root=root, is_swarm_child=True, parent_session_id=5, swarm_label="child")
            result = self.runner.invoke(cli, ["--cwd", tmp, "sessions"])
            result_all = self.runner.invoke(cli, ["--cwd", tmp, "sessions", "--all"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.count(root), 1)
        self.assertEqual(result_all.output.count(root), 2)

    def test_sessions_hides_swarm_child_with_no_known_parent(self):
        # Live-caught regression (agent-cmd delegate has no pre-existing
        # session to record as a parent) -- is_swarm_child alone must still
        # hide it, independent of parent_session_id being None.
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp).resolve())
            state_module.save_session_state(7, workspace_root=root, is_swarm_child=True, parent_session_id=None, swarm_label="no parent")
            result = self.runner.invoke(cli, ["--cwd", tmp, "sessions"])
            result_all = self.runner.invoke(cli, ["--cwd", tmp, "sessions", "--all"])
        self.assertEqual(result.output.count(root), 0)
        self.assertEqual(result_all.output.count(root), 1)

    def test_diffs_reads_local_mutation_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tamfis_code.safety import record_mutation

            record_mutation(1, path=f"{tmp}/x.py", operation="create", original_content=None, new_content="x = 1\n")
            state_module.save_session_state(1, workspace_root=str(Path(tmp).resolve()))
            result = self.runner.invoke(cli, ["--cwd", tmp, "diffs"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("create", result.output)

    def test_diffs_remote_without_auth_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "diffs", "--remote"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Not authenticated", result.output)

    def test_revert_unknown_mutation_reports_error_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "revert", "mut_doesnotexist"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No recorded mutation", result.output)

    def test_revert_turn_id_reverts_every_mutation_from_that_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tamfis_code.safety import record_mutation

            a, b = Path(tmp) / "a.py", Path(tmp) / "b.py"
            a.write_text("new a\n")
            b.write_text("new b\n")
            record_mutation(1, path=str(a), operation="update", original_content="old a\n", new_content="new a\n", transaction_id="turn_cli1")
            record_mutation(1, path=str(b), operation="update", original_content="old b\n", new_content="new b\n", transaction_id="turn_cli1")
            state_module.save_session_state(1, workspace_root=str(Path(tmp).resolve()))
            result = self.runner.invoke(cli, ["--cwd", tmp, "revert", "turn_cli1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Reverted 2 mutation(s)", result.output)
            self.assertEqual(a.read_text(), "old a\n")
            self.assertEqual(b.read_text(), "old b\n")

    def test_revert_unknown_turn_id_reports_error_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "revert", "turn_doesnotexist"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No recorded mutations", result.output)

    def test_diffs_shows_the_turn_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tamfis_code.safety import record_mutation

            record_mutation(1, path=f"{tmp}/x.py", operation="create", original_content=None, new_content="x = 1\n", transaction_id="turn_visible")
            state_module.save_session_state(1, workspace_root=str(Path(tmp).resolve()))
            result = self.runner.invoke(cli, ["--cwd", tmp, "diffs"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("turn_visible", result.output)

    def test_resume_shows_incomplete_plan_step_progress(self):
        # Before this, `tamfis-code resume` showed only a conversation
        # summary -- a plan left mid-execution was invisible once resumed.
        with tempfile.TemporaryDirectory() as tmp:
            state_module.save_session_state(
                7, workspace_root=str(Path(tmp).resolve()),
                saved_plans=[{
                    "id": "plan_cli_resume1", "objective": "fix the login bug",
                    "steps": [
                        {"index": 0, "step": "Read auth.py", "status": "completed"},
                        {"index": 1, "step": "Fix the bug", "status": "in_progress"},
                    ],
                }],
                active_plan_id="plan_cli_resume1",
            )
            with patch("tamfis_code.interactive.run_interactive", new=AsyncMock(return_value=None)):
                result = self.runner.invoke(cli, ["--cwd", tmp, "resume", "7"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Plan in progress", result.output)
        self.assertIn("Read auth.py", result.output)
        self.assertIn("Fix the bug", result.output)

    def test_mcp_server_command_runs_stdio_server_scoped_to_the_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.mcp_stdio_server.run_stdio_server", new=AsyncMock(return_value=None)) as fake_run:
                result = self.runner.invoke(cli, ["--cwd", tmp, "mcp-server"])
        self.assertEqual(result.exit_code, 0, result.output)
        fake_run.assert_called_once()
        self.assertEqual(fake_run.call_args.args[0], str(Path(tmp)))

    def test_run_executes_locally_without_remote_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli.RemoteAPIClient") as fake_client:
                result = self.runner.invoke(cli, ["--cwd", tmp, "--approval", "auto", "run", "echo hi"])
            fake_client.assert_not_called()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("hi", result.output)

    def test_run_bg_without_remote_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "run", "echo hi", "--bg"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--bg requires --remote", result.output)

    def test_doctor_reports_provider_status_without_remote_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli.RemoteAPIClient") as fake_client:
                result = self.runner.invoke(cli, ["--cwd", tmp, "doctor"])
            fake_client.assert_not_called()
        self.assertIn("PROVIDER", result.output)

    def test_doctor_reports_local_session_diagnostics_by_default(self):
        # `doctor` (no --remote) used to stop at the provider connectivity
        # table and never report anything about actual local turns run in
        # this directory, even though state.py already records tool-call
        # outcomes, plan progress, and estimated context usage for exactly
        # this purpose -- run_doctor() (where that reporting lived) was
        # only ever reached via the separate --remote code path.
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tamfis_code.cli.RemoteAPIClient") as fake_client:
                result = self.runner.invoke(cli, ["--cwd", tmp, "doctor"])
            fake_client.assert_not_called()
        self.assertIn("Local session context usage", result.output)
        self.assertIn("Local tool-call success rate", result.output)

    def test_retry_with_no_previous_turn_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "retry"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No previous turn", result.output)

    def test_retry_task_id_without_remote_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.invoke(cli, ["--cwd", tmp, "retry", "some-task-id"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("only applies with --remote", result.output)


if __name__ == "__main__":
    unittest.main()
