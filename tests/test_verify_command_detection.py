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
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamfis_code.workspace import detect_verify_command


def _write_package_json(root: Path, scripts: dict) -> None:
    (root / "package.json").write_text(json.dumps({"name": "x", "scripts": scripts}), encoding="utf-8")


class DetectVerifyCommandTests(unittest.TestCase):
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
