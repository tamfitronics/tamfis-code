"""An unreadable neighbouring directory must never crash the system-prompt build.

CI (GitHub runner, non-root) failed 51 tests with
`PermissionError: /tmp/snap-private-tmp/wp-content`: building the system prompt scans the
workspace's SIBLING directories for other projects, `Path.is_dir()` raises PermissionError on
EACCES (it only swallows ENOENT), and a root-only neighbour in /tmp took the whole prompt down.
It passed on the dev machine only because that runs as root, so the tests here simulate the
denied stat instead of relying on chmod (which root ignores).
"""
import pathlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import workspace
from tamfis_code.workspace import _detect_sibling_projects, _discover_project_type

_REAL_IS_DIR = pathlib.Path.is_dir
_REAL_EXISTS = pathlib.Path.exists
_REAL_IS_FILE = pathlib.Path.is_file


def _deny_inside(locked: Path):
    """Path methods that raise PermissionError for anything INSIDE `locked`, as a real
    non-root user gets for a directory with mode 700 owned by someone else."""
    locked = locked.resolve()

    def guard(real):
        def wrapper(self, *args, **kwargs):
            try:
                inside = locked in Path(self).resolve().parents
            except OSError:
                inside = False
            if inside:
                raise PermissionError(13, "Permission denied", str(self))
            return real(self, *args, **kwargs)
        return wrapper

    return (
        patch.object(pathlib.Path, "is_dir", guard(_REAL_IS_DIR)),
        patch.object(pathlib.Path, "exists", guard(_REAL_EXISTS)),
        patch.object(pathlib.Path, "is_file", guard(_REAL_IS_FILE)),
    )


class UnreadableSiblingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.parent = Path(tmp.name).resolve()
        self.workspace = self.parent / "app"
        self.workspace.mkdir()
        self.locked = self.parent / "snap-private-tmp"
        self.locked.mkdir()
        self.other = self.parent / "site"
        self.other.mkdir()
        (self.other / "package.json").write_text('{"name": "site"}')

    def _denied(self):
        patches = _deny_inside(self.locked)
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_the_simulation_really_raises(self):
        self._denied()
        with self.assertRaises(PermissionError):
            (self.locked / "wp-content").is_dir()

    def test_an_unreadable_directory_is_unknown_not_a_crash(self):
        self._denied()
        self.assertEqual(_discover_project_type(self.locked)["language"], "unknown")

    def test_the_sibling_scan_skips_the_unreadable_neighbour_and_still_finds_the_others(self):
        self._denied()
        found = dict(_detect_sibling_projects(self.parent))
        self.assertIn("site", found)
        self.assertNotIn("snap-private-tmp", found)

    def test_building_the_system_prompt_beside_an_unreadable_directory_works(self):
        self._denied()
        # A workspace ROOT that holds several projects (a shared /home, /tmp, /srv): the prompt
        # build scans its child directories, one of which the user cannot read.
        prompt = workspace.build_system_prompt(1, self.parent)
        self.assertIsInstance(prompt, str)
        self.assertTrue(prompt.strip())

    def test_a_readable_workspace_is_detected_as_before(self):
        self.assertNotEqual(_discover_project_type(self.other)["language"], "unknown")


if __name__ == "__main__":
    unittest.main()
