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
    cli,
)


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


if __name__ == "__main__":
    unittest.main()
