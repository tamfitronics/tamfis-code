"""workspace.py's detect_verify_command -- the highest-leverage part of the
build/typecheck completion gate (see test_verify_command_gate.py for the
active guard this feeds). Confirmed live against TamfisSEO Pro: its
package.json has "check": "tsc -b" (a real type-check), not the literal
name "typecheck" -- both this function and enforcer.py's own script list
originally only recognized "typecheck", silently missing it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamfis_code.workspace import detect_validation_commands, detect_verify_command


def _write_package_json(root: Path, scripts: dict) -> None:
    (root / "package.json").write_text(json.dumps({"name": "x", "scripts": scripts}), encoding="utf-8")


class DetectVerifyCommandTests(unittest.TestCase):
    def test_detects_typecheck_build_and_tests_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            _write_package_json(root, {"check": "tsc -b", "build": "vite build", "test": "vitest"})
            commands = dict(detect_validation_commands(root))
        self.assertIn("npm run check", commands.values())
        self.assertIn("npm run build", commands.values())
        self.assertIn("npm test", commands.values())
        self.assertIn("git diff --check", commands.values())

    def test_detects_python_checks(self):
        # Regression: this used to hardcode "python -m compileall -q ." --
        # wrong on this box and most modern Linux distros, which have no
        # bare `python` on PATH, only `python3`. Match the same
        # interpreter-detection the implementation uses instead of assuming
        # either name, so this test is correct on both kinds of machine.
        python_bin = "python3" if shutil.which("python3") else "python"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[tool.ruff]\n")
            (root / "tests").mkdir()
            commands = dict(detect_validation_commands(root))
        self.assertIn(f"{python_bin} -m compileall -q .", commands.values())
        self.assertIn("ruff check .", commands.values())
        self.assertIn("pytest -q", commands.values())

    def test_python_syntax_check_uses_python3_when_bare_python_is_absent(self):
        """Live-reproduced: a model correctly ran `python -m compileall -q .`
        (fails: "bash: python: command not found" on this box), self-
        corrected to `python3 -m compileall -q .` (succeeds), but the
        harness's own confirmation check does a plain substring match of the
        *detected* command against the executed one -- "python -m
        compileall -q ." is not a substring of "python3 -m compileall -q ."
        (the "3" breaks it right after "python"). The real, successful,
        self-corrected run was never recognized, and the task hard-failed
        after exhausting its retry budget. Pin both PATH states explicitly
        so this doesn't depend on what happens to be installed wherever the
        test suite runs.
        """
        from unittest.mock import patch
        from tamfis_code.workspace import detect_validation_commands

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text("")
            (root / "tests").mkdir()

            with patch("tamfis_code.workspace.shutil.which", side_effect=lambda name: name == "python3" and "/usr/bin/python3" or None):
                commands = dict(detect_validation_commands(root))
            self.assertIn("python3 -m compileall -q .", commands.values())
            self.assertNotIn("python -m compileall -q .", commands.values())

            with patch("tamfis_code.workspace.shutil.which", side_effect=lambda name: name == "python" and "/usr/bin/python" or None):
                commands = dict(detect_validation_commands(root))
            self.assertIn("python -m compileall -q .", commands.values())

    def test_manifestless_project_container_does_not_inherit_python_compileall(self):
        """A stray root script must not turn sibling WordPress sites into Python projects."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "operator_helper.py").write_text("print('not the project')\n")
            for name in ("site-a", "site-b"):
                site = root / name
                site.mkdir()
                (site / "wp-config.php").write_text("<?php\n")
                (site / "index.php").write_text("<?php\n")

            commands = dict(detect_validation_commands(root))

        self.assertNotIn("python3 -m compileall -q .", commands.values())
        self.assertNotIn("python -m compileall -q .", commands.values())

    def test_detects_cmake_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n")
            commands = dict(detect_validation_commands(root))
        self.assertIn("cmake -S . -B build && cmake --build build", commands.values())

    def test_prefers_check_script_over_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_package_json(root, {"check": "tsc -b", "build": "vite build"})
            self.assertEqual(detect_verify_command(root), ("check", "npm run check"))

    def test_prefers_typecheck_over_build_when_no_check_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_package_json(root, {"typecheck": "tsc --noEmit", "build": "vite build"})
            self.assertEqual(detect_verify_command(root), ("typecheck", "npm run typecheck"))

    def test_falls_back_to_build_when_no_typecheck_script_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_package_json(root, {"build": "vite build", "test": "vitest run"})
            self.assertEqual(detect_verify_command(root), ("build", "npm run build"))

    def test_returns_none_when_no_recognised_script_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_package_json(root, {"start": "node index.js"})
            self.assertIsNone(detect_verify_command(root))

    def test_returns_none_when_no_package_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(detect_verify_command(Path(tmp)))

    def test_returns_none_for_a_python_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            self.assertIsNone(detect_verify_command(root))


if __name__ == "__main__":
    unittest.main()
