import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import state as state_module
from tamfis_code.workspace import (
    _git,
    _indexable_files,
    _report_title,
    discover_local_repository,
)


class _StatePatchMixin:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        state_module.CONFIG_DIR = root / ".config"
        state_module.STATE_PATH = root / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


class GitHelperTests(unittest.TestCase):
    def test_returns_empty_string_on_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Not a git repo -> git rev-parse fails -> "" not an exception.
            self.assertEqual(_git(Path(tmp), "rev-parse", "HEAD"), "")

    def test_returns_empty_string_on_timeout(self):
        with patch(
            "tamfis_code.workspace.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5),
        ):
            self.assertEqual(_git(Path("/tmp"), "status"), "")

    def test_returns_empty_string_on_oserror(self):
        with patch("tamfis_code.workspace.subprocess.run", side_effect=OSError("no git binary")):
            self.assertEqual(_git(Path("/tmp"), "status"), "")


class IndexableFilesTests(unittest.TestCase):
    def test_falls_back_to_rglob_outside_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x")
            (root / "sub").mkdir()
            (root / "sub" / "b.py").write_text("y")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "c.pyc").write_text("z")
            files = _indexable_files(root)
        names = {f.name for f in files}
        self.assertIn("a.py", names)
        self.assertIn("b.py", names)
        self.assertNotIn("c.pyc", names)

    def test_uses_git_ls_files_inside_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tracked.py").write_text("x")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
            files = _indexable_files(root)
        names = {f.name for f in files}
        self.assertIn("tracked.py", names)


class ReportTitleTests(unittest.TestCase):
    def test_extracts_first_markdown_heading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            path.write_text("intro line\n# Real Title\nbody\n")
            self.assertEqual(_report_title(path), "Real Title")

    def test_falls_back_to_stem_when_no_heading_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "my_cool-report.md"
            path.write_text("no heading here at all\n")
            self.assertEqual(_report_title(path), "my cool report")

    def test_non_markdown_suffix_uses_stem_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status_update.json"
            path.write_text('{"heading-like": "# not a real heading"}')
            self.assertEqual(_report_title(path), "status update")


class DiscoverLocalRepositoryTests(_StatePatchMixin, unittest.TestCase):
    def test_finds_instruction_files_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("# Agents\n")
            (root / "audit_findings_report.md").write_text("# Audit Findings\nStuff.\n")
            (root / "random_notes.md").write_text("nothing relevant\n")
            context = discover_local_repository(9001, root)
        self.assertEqual(len(context["instruction_files"]), 1)
        self.assertTrue(context["instruction_files"][0].endswith("AGENTS.md"))
        state = state_module.get_session_state(9001)
        report_paths = [r["path"] for r in state.discovered_reports]
        self.assertTrue(any("audit_findings_report.md" in p for p in report_paths))
        self.assertFalse(any("random_notes.md" in p for p in report_paths))

    def test_no_head_reports_as_no_head_string_internally_but_none_in_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file.txt").write_text("x")
            context = discover_local_repository(9002, root)
        self.assertIsNone(context["head"])
        self.assertFalse(context["dirty"])


if __name__ == "__main__":
    unittest.main()
