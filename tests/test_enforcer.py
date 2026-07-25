import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tamfis_code.enforcer import TestEnforcer, run_enforcer


class RunCmdTests(unittest.TestCase):
    def test_successful_command_reports_success(self):
        enforcer = TestEnforcer(Path("/tmp"))
        result = enforcer._run_cmd(["true"], cwd=Path("/tmp"))
        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)

    def test_failing_command_reports_failure(self):
        enforcer = TestEnforcer(Path("/tmp"))
        result = enforcer._run_cmd(["false"], cwd=Path("/tmp"))
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 1)

    def test_timeout_is_reported_as_failure_not_an_exception(self):
        enforcer = TestEnforcer(Path("/tmp"))
        with patch("tamfis_code.enforcer.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            result = enforcer._run_cmd(["sleep", "10"], cwd=Path("/tmp"), timeout=1)
        self.assertFalse(result["success"])
        self.assertIn("Timeout", result["error"])

    def test_unexpected_exception_is_reported_as_failure_not_raised(self):
        enforcer = TestEnforcer(Path("/tmp"))
        with patch("tamfis_code.enforcer.subprocess.run", side_effect=OSError("boom")):
            result = enforcer._run_cmd(["whatever"], cwd=Path("/tmp"))
        self.assertFalse(result["success"])
        self.assertIn("boom", result["error"])


class TestEnforcerInitTests(unittest.TestCase):
    def test_results_scaffold_has_expected_keys(self):
        enforcer = TestEnforcer(Path("/tmp"))
        for key in ("timestamp", "workspace", "python", "frontend_tests"):
            self.assertIn(key, enforcer.results)

    def test_workspace_root_defaults_to_cwd(self):
        with patch("tamfis_code.enforcer.Path.cwd", return_value=Path("/somewhere")):
            enforcer = TestEnforcer()
        self.assertEqual(enforcer.workspace_root, Path("/somewhere"))

    def test_workspace_root_is_scoped_to_the_given_directory_not_a_fixed_sibling_layout(self):
        # Regression: TestEnforcer used to hard-code /home/tamfisgpt/tamgpt6
        # and /home/tamfisgpt/tamfis-frontend regardless of what workspace
        # was actually passed in, so `tamfis-code enforce` silently tested a
        # different, unrelated project on one specific machine instead of
        # the current workspace.
        enforcer = TestEnforcer(Path("/some/other/project"))
        self.assertEqual(enforcer.workspace_root, Path("/some/other/project"))
        self.assertEqual(enforcer.results["workspace"], "/some/other/project")


class RunBackendTestsTests(unittest.TestCase):
    def test_missing_test_directory_falls_back_to_bare_pytest(self):
        with tempfile_dir() as tmp:
            enforcer = TestEnforcer(tmp)
            with patch.object(enforcer, "_run_cmd", return_value={"success": True, "elapsed": 0.1, "stdout": "", "stderr": ""}) as mock_run:
                enforcer._run_backend_tests()
            mock_run.assert_called_once()
            self.assertIn("pytest", mock_run.call_args[0][0])
        self.assertEqual(enforcer.results["python"]["summary"]["total"], 1)

    def test_counts_passed_and_failed_test_files(self):
        with tempfile_dir() as tmp:
            (tmp / "tests").mkdir()
            (tmp / "tests" / "test_a.py").write_text("")
            (tmp / "tests" / "test_b.py").write_text("")
            enforcer = TestEnforcer(tmp)
            with patch.object(enforcer, "_run_cmd", side_effect=[
                {"success": True, "elapsed": 0.1, "stdout": "", "stderr": ""},
                {"success": False, "elapsed": 0.2, "stdout": "FAILED test_b.py::test_x", "stderr": ""},
            ]):
                enforcer._run_backend_tests()
        self.assertEqual(enforcer.results["python"]["summary"]["total"], 2)
        self.assertEqual(enforcer.results["python"]["summary"]["passed"], 1)


class RunFrontendTestsTests(unittest.TestCase):
    def test_empty_directory_is_skipped_not_treated_as_a_js_project(self):
        # _run_frontend_tests now gates on _discover_project_type first (the
        # stack-awareness fix for npm running against non-Node projects) --
        # an empty directory detects as "unknown", which is caught by that
        # gate before ever reaching the package.json-existence check below
        # it, so the status is "skipped", not "no_package" (that status is
        # now only reachable if _discover_project_type ever reports a JS/TS
        # language without package.json actually existing, which it can't --
        # its own JS/TS branch only fires when package.json.is_file() is
        # true in the first place).
        with tempfile_dir() as tmp:
            enforcer = TestEnforcer(tmp)
            enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "skipped")

    def test_missing_npm_sets_no_npm_status(self):
        with tempfile_dir() as tmp:
            (tmp / "package.json").write_text('{"scripts": {"test": "vitest"}}')
            enforcer = TestEnforcer(tmp)
            with patch.object(enforcer, "_run_cmd", return_value={"success": False, "stdout": "", "stderr": "not found"}):
                enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "no_npm")

    def test_no_recognized_scripts_sets_no_scripts_status(self):
        with tempfile_dir() as tmp:
            (tmp / "package.json").write_text('{"scripts": {"lint": "eslint ."}}')
            (tmp / "node_modules").mkdir()
            enforcer = TestEnforcer(tmp)
            with patch.object(enforcer, "_run_cmd", return_value={"success": True, "stdout": "8.0.0", "stderr": ""}):
                enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "no_scripts")

    def test_recognized_scripts_are_run_and_summarized(self):
        with tempfile_dir() as tmp:
            (tmp / "package.json").write_text('{"scripts": {"test": "vitest", "build": "vite build"}}')
            (tmp / "node_modules").mkdir()
            enforcer = TestEnforcer(tmp)

            def fake_run_cmd(cmd, cwd=None, timeout=120):
                if cmd == ["npm", "--version"]:
                    return {"success": True, "stdout": "8.0.0", "stderr": "", "elapsed": 0.01}
                return {"success": True, "stdout": "", "stderr": "", "elapsed": 0.5}

            with patch.object(enforcer, "_run_cmd", side_effect=fake_run_cmd):
                enforcer._run_frontend_tests()
        self.assertEqual(enforcer.results["frontend_tests"]["status"], "done")
        self.assertEqual(len(enforcer.results["frontend_tests"]["results"]), 2)
        self.assertTrue(all(r["passed"] for r in enforcer.results["frontend_tests"]["results"]))


class RunDetectsRightSuitesTests(unittest.TestCase):
    def test_run_only_invokes_suites_it_detected(self):
        with tempfile_dir() as tmp:
            (tmp / "package.json").write_text('{"scripts": {}}')
            enforcer = TestEnforcer(tmp)
            with patch.object(enforcer, "_run_backend_tests") as backend, \
                 patch.object(enforcer, "_run_frontend_tests") as frontend, \
                 patch.object(enforcer, "_run_cargo_tests") as cargo:
                enforcer.run()
        backend.assert_not_called()
        frontend.assert_called_once()
        cargo.assert_not_called()


class GenerateSummaryTests(unittest.TestCase):
    def test_generate_summary_does_not_raise_with_empty_results(self):
        enforcer = TestEnforcer(Path("/tmp"))
        enforcer._generate_summary()  # must not raise


class RunEnforcerFunctionTests(unittest.TestCase):
    def test_run_enforcer_delegates_to_testenforcer_run(self):
        with patch("tamfis_code.enforcer.TestEnforcer.run", return_value={"ok": True}) as mock_run:
            result = run_enforcer(Path("/tmp"))
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
