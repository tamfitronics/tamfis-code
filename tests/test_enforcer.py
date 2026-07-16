import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tamfis_code.enforcer import TestEnforcer, run_enforcer


class RunCmdTests(unittest.TestCase):
    def test_successful_command_reports_success(self):
        enforcer = TestEnforcer()
        result = enforcer._run_cmd(["true"], cwd=Path("/tmp"))
        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)

    def test_failing_command_reports_failure(self):
        enforcer = TestEnforcer()
        result = enforcer._run_cmd(["false"], cwd=Path("/tmp"))
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 1)

    def test_timeout_is_reported_as_failure_not_an_exception(self):
        enforcer = TestEnforcer()
        with patch("tamfis_code.enforcer.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            result = enforcer._run_cmd(["sleep", "10"], cwd=Path("/tmp"), timeout=1)
        self.assertFalse(result["success"])
        self.assertIn("Timeout", result["error"])

    def test_unexpected_exception_is_reported_as_failure_not_raised(self):
        enforcer = TestEnforcer()
        with patch("tamfis_code.enforcer.subprocess.run", side_effect=OSError("boom")):
            result = enforcer._run_cmd(["whatever"], cwd=Path("/tmp"))
        self.assertFalse(result["success"])
        self.assertIn("boom", result["error"])


class TestEnforcerInitTests(unittest.TestCase):
    def test_results_scaffold_has_expected_keys(self):
        enforcer = TestEnforcer()
        for key in ("timestamp", "workspace", "backend", "frontend", "python", "node", "frontend_tests"):
            self.assertIn(key, enforcer.results)

    def test_backend_and_frontend_paths_are_siblings_under_base(self):
        enforcer = TestEnforcer()
        self.assertEqual(enforcer.backend, enforcer.base / "tamgpt6")
        self.assertEqual(enforcer.frontend, enforcer.base / "tamfis-frontend")


class RunBackendTestsTests(unittest.TestCase):
    def test_missing_test_directory_is_handled_without_raising(self):
        enforcer = TestEnforcer()
        enforcer.backend = Path("/nonexistent/backend/path")
        enforcer._run_backend_tests()  # must not raise
        self.assertNotIn("total", enforcer.results["python"]["summary"])

    def test_counts_passed_and_failed_test_files(self):
        enforcer = TestEnforcer()
        with patch.object(Path, "glob", return_value=[Path("test_a.py"), Path("test_b.py")]):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(enforcer, "_run_cmd", side_effect=[
                    {"success": True, "elapsed": 0.1, "stdout": "", "stderr": ""},
                    {"success": False, "elapsed": 0.2, "stdout": "FAILED test_b.py::test_x", "stderr": ""},
                ]):
                    enforcer._run_backend_tests()
        self.assertEqual(enforcer.results["python"]["summary"]["total"], 2)
        self.assertEqual(enforcer.results["python"]["summary"]["passed"], 1)


class RunFrontendTestsTests(unittest.TestCase):
    def test_missing_frontend_directory_is_handled_without_raising(self):
        enforcer = TestEnforcer()
        enforcer.frontend = Path("/nonexistent/frontend/path")
        enforcer._run_frontend_tests()  # must not raise

    def test_missing_package_json_sets_no_package_status(self):
        enforcer = TestEnforcer()
        with tempfile_dir() as tmp:
            enforcer.frontend = tmp
            enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "no_package")

    def test_missing_npm_sets_no_npm_status(self):
        enforcer = TestEnforcer()
        with tempfile_dir() as tmp:
            (tmp / "package.json").write_text('{"scripts": {"test": "vitest"}}')
            enforcer.frontend = tmp
            with patch.object(enforcer, "_run_cmd", return_value={"success": False, "stdout": "", "stderr": "not found"}):
                enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "no_npm")

    def test_no_recognized_scripts_sets_no_scripts_status(self):
        enforcer = TestEnforcer()
        with tempfile_dir() as tmp:
            (tmp / "package.json").write_text('{"scripts": {"lint": "eslint ."}}')
            (tmp / "node_modules").mkdir()
            enforcer.frontend = tmp
            with patch.object(enforcer, "_run_cmd", return_value={"success": True, "stdout": "8.0.0", "stderr": ""}):
                enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "no_scripts")

    def test_recognized_scripts_are_run_and_summarized(self):
        enforcer = TestEnforcer()
        with tempfile_dir() as tmp:
            (tmp / "package.json").write_text('{"scripts": {"test": "vitest", "build": "vite build"}}')
            (tmp / "node_modules").mkdir()
            enforcer.frontend = tmp

            def fake_run_cmd(cmd, cwd=None, timeout=120):
                if cmd == ["npm", "--version"]:
                    return {"success": True, "stdout": "8.0.0", "stderr": "", "elapsed": 0.01}
                return {"success": True, "stdout": "", "stderr": "", "elapsed": 0.5}

            with patch.object(enforcer, "_run_cmd", side_effect=fake_run_cmd):
                enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "done")
        self.assertEqual(len(enforcer.results["frontend_tests"]["results"]), 2)
        self.assertTrue(all(r["passed"] for r in enforcer.results["frontend_tests"]["results"]))


class GenerateSummaryTests(unittest.TestCase):
    def test_generate_summary_does_not_raise_with_empty_results(self):
        enforcer = TestEnforcer()
        enforcer._generate_summary()  # must not raise


class RunEnforcerFunctionTests(unittest.TestCase):
    def test_run_enforcer_delegates_to_testenforcer_run(self):
        with patch("tamfis_code.enforcer.TestEnforcer.run", return_value={"ok": True}) as mock_run:
            result = run_enforcer()
        mock_run.assert_called_once()
        self.assertEqual(result, {"ok": True})


def tempfile_dir():
    import tempfile as _tempfile

    class _Ctx:
        def __enter__(self_inner):
            self_inner._tmp = _tempfile.TemporaryDirectory()
            return Path(self_inner._tmp.name)

        def __exit__(self_inner, *exc):
            self_inner._tmp.cleanup()

    return _Ctx()


if __name__ == "__main__":
    unittest.main()
