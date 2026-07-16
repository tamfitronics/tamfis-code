import os
import tempfile
import unittest
from pathlib import Path

from tamfis_code import config as config_module
from tamfis_code import state as state_module


class LoadConfigTests(unittest.TestCase):
    def setUp(self):
        # config.py computes CONFIG_DIR/CREDENTIALS_PATH/USER_CONFIG_PATH
        # once at import time as module-level constants -- patching
        # CONFIG_DIR alone would NOT redirect the other two, since they
        # aren't re-derived from it dynamically. All three must be patched
        # so a test run never touches the real ~/.config/tamfis-code/.
        self._originals = {
            "CONFIG_DIR": config_module.CONFIG_DIR,
            "CREDENTIALS_PATH": config_module.CREDENTIALS_PATH,
            "USER_CONFIG_PATH": config_module.USER_CONFIG_PATH,
        }
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)
        config_module.CONFIG_DIR = tmp_path
        config_module.CREDENTIALS_PATH = tmp_path / "credentials.json"
        config_module.USER_CONFIG_PATH = tmp_path / "config.toml"
        self._env_token = os.environ.pop("TAMFIS_CODE_TOKEN", None)
        self._env_api_base = os.environ.pop("TAMFIS_CODE_API_BASE", None)

    def tearDown(self):
        config_module.CONFIG_DIR = self._originals["CONFIG_DIR"]
        config_module.CREDENTIALS_PATH = self._originals["CREDENTIALS_PATH"]
        config_module.USER_CONFIG_PATH = self._originals["USER_CONFIG_PATH"]
        self.tmpdir.cleanup()
        if self._env_token is not None:
            os.environ["TAMFIS_CODE_TOKEN"] = self._env_token
        if self._env_api_base is not None:
            os.environ["TAMFIS_CODE_API_BASE"] = self._env_api_base

    def test_default_api_base_when_no_config_present(self):
        cfg = config_module.load_config()
        self.assertEqual(cfg.api_base, config_module.DEFAULT_API_BASE)
        self.assertEqual(cfg.sources["api_base"], "default")

    def test_user_config_overrides_default(self):
        config_module.USER_CONFIG_PATH.write_text('api_base = "http://user.invalid"\n')
        cfg = config_module.load_config()
        self.assertEqual(cfg.api_base, "http://user.invalid")
        self.assertEqual(cfg.sources["api_base"], "user config")

    def test_project_config_overrides_user_config(self):
        config_module.USER_CONFIG_PATH.write_text('api_base = "http://user.invalid"\n')
        project_root = Path(self.tmpdir.name) / "proj"
        (project_root / ".tamfis").mkdir(parents=True)
        (project_root / ".tamfis" / "config.toml").write_text('api_base = "http://project.invalid"\n')

        cfg = config_module.load_config(project_root=project_root)
        self.assertEqual(cfg.api_base, "http://project.invalid")

    def test_env_var_overrides_config_files(self):
        config_module.USER_CONFIG_PATH.write_text('api_base = "http://user.invalid"\n')
        os.environ["TAMFIS_CODE_API_BASE"] = "http://env.invalid"
        try:
            cfg = config_module.load_config()
        finally:
            del os.environ["TAMFIS_CODE_API_BASE"]
        self.assertEqual(cfg.api_base, "http://env.invalid")
        self.assertEqual(cfg.sources["api_base"], "env TAMFIS_CODE_API_BASE")

    def test_invalid_approval_policy_in_config_file_is_ignored(self):
        config_module.USER_CONFIG_PATH.write_text('approval_policy = "yolo"\n')
        cfg = config_module.load_config()
        self.assertEqual(cfg.approval_policy, "ask")  # falls back to the built-in default

    def test_credentials_roundtrip(self):
        creds = config_module.Credentials(access_token="tok", refresh_token="ref", user_id="u1", email="a@b.com")
        config_module.save_credentials(creds)

        loaded = config_module.load_credentials()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.access_token, "tok")
        self.assertEqual(loaded.email, "a@b.com")

        self.assertTrue(config_module.clear_credentials())
        self.assertIsNone(config_module.load_credentials())
        self.assertFalse(config_module.clear_credentials())  # idempotent

    def test_credentials_file_written_with_owner_only_permissions(self):
        creds = config_module.Credentials(access_token="tok")
        config_module.save_credentials(creds)
        mode = config_module.CREDENTIALS_PATH.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_token_env_var_overrides_saved_credentials_file(self):
        config_module.save_credentials(config_module.Credentials(access_token="from-file"))
        os.environ["TAMFIS_CODE_TOKEN"] = "from-env"
        try:
            loaded = config_module.load_credentials()
        finally:
            del os.environ["TAMFIS_CODE_TOKEN"]
        self.assertEqual(loaded.access_token, "from-env")


class DurableSessionStateTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_dir = state_module.CONFIG_DIR
        self.original_path = state_module.STATE_PATH
        state_module.CONFIG_DIR = Path(self.tmpdir.name)
        state_module.STATE_PATH = Path(self.tmpdir.name) / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR = self.original_dir
        state_module.STATE_PATH = self.original_path
        self.tmpdir.cleanup()

    def test_extended_session_context_survives_reload(self):
        state_module.save_session_state(
            7, workspace_root="/repo", repository_root="/repo", active_branch="main",
            current_phase="validate", validation_results=[{"command": "pytest", "status": "passed"}],
        )
        loaded = state_module.get_session_state(7)
        self.assertEqual(loaded.repository_root, "/repo")
        self.assertEqual(loaded.active_branch, "main")
        self.assertEqual(loaded.validation_results[0]["status"], "passed")
        self.assertEqual(state_module.STATE_PATH.stat().st_mode & 0o777, 0o600)

    def test_instruction_queue_is_priority_ordered_and_status_is_durable(self):
        later = state_module.enqueue_instruction(8, "later", priority=200)
        sooner = state_module.enqueue_instruction(8, "auth first", classification="reprioritise", priority=10)
        loaded = state_module.get_session_state(8)
        self.assertEqual([item["id"] for item in loaded.queued_user_instructions], [sooner.id, later.id])
        self.assertTrue(state_module.update_instruction(8, sooner.id, "completed"))
        self.assertEqual(state_module.get_session_state(8).queued_user_instructions[0]["status"], "completed")

    def test_action_history_and_checkpoint_survive_reconnection(self):
        action = state_module.start_action(9, action_type="shell_command", purpose="run tests")
        state_module.finish_action(9, action.id, status="completed", summary="exit=0")
        state_module.checkpoint(9, reason="validation_complete", summary="tests passed")
        loaded = state_module.get_session_state(9)
        self.assertIsNone(loaded.running_action)
        self.assertEqual(loaded.completed_actions[-1]["id"], action.id)
        self.assertEqual(loaded.context_checkpoints[-1]["reason"], "validation_complete")

    def test_persisted_queue_and_actions_redact_likely_secrets(self):
        state_module.enqueue_instruction(10, "retry with access_token=super-secret-token")
        state_module.start_action(10, action_type="shell_command", purpose="login", detail="password=hunter2")
        raw = state_module.STATE_PATH.read_text()
        self.assertNotIn("super-secret-token", raw)
        self.assertNotIn("hunter2", raw)
        self.assertIn("[REDACTED]", raw)

    def test_saved_plan_lifecycle_is_durable_and_executable(self):
        state_module.save_session_state(11, workspace_root="/repo")
        saved = state_module.save_plan(
            11, objective="Add validation", content="1. Inspect\n2. Implement\n3. Test",
            source_task_id="task-plan",
        )

        loaded = state_module.get_plan(11, saved.id[:10])
        self.assertEqual(loaded["objective"], "Add validation")
        self.assertEqual(loaded["status"], "ready")
        prompt = state_module.plan_execution_objective(loaded)
        self.assertIn("Do not merely restate the plan", prompt)
        self.assertIn("3. Test", prompt)

        state_module.update_plan(11, saved.id, status="completed", execution_task_id="task-exec")
        reloaded = state_module.get_plan(11, saved.id)
        self.assertEqual(reloaded["status"], "completed")
        self.assertEqual(reloaded["execution_task_id"], "task-exec")


if __name__ == "__main__":
    unittest.main()
