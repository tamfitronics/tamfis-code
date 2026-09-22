"""Regression coverage for _workspace_roots_related (live-reproduced
2026-08-30): a broad current workspace_root spanning multiple sibling
projects (e.g. /home, needed because a task legitimately touches several
site directories under it) used to be treated as an "ancestor" of every
OTHER unrelated project nested anywhere under that same broad root, no
matter how deep or unrelated. _select_resume_state's cross-session search
then matched a completely unrelated session's stale checkpoint purely
because both workspace roots happened to share a broad common ancestor --
confirmed live: a WordPress-site-audit task (root /home) resumed a stale
checkpoint about fixing a bug in this very CLI's own orchestrator/
planner.py (root /home/tamfisgpt/tamgpt6), a project the user never
mentioned that turn.
"""
import tempfile
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.runner_local import _select_resume_state, _workspace_roots_related


class WorkspaceRootsRelatedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_identical_roots_are_related(self):
        project = self.root / "project"
        project.mkdir()
        self.assertTrue(_workspace_roots_related(str(project), str(project)))

    def test_current_root_nested_inside_candidate_root_is_related(self):
        # The documented legitimate case: the interrupted turn's checkpoint
        # is rooted at the parent project; the new turn restarts from a
        # subdirectory of that same project.
        parent = self.root / "project"
        child = parent / "backend"
        child.mkdir(parents=True)
        self.assertTrue(_workspace_roots_related(str(parent), str(child)))

    def test_candidate_root_nested_inside_a_broad_current_root_is_not_related(self):
        # The bug: current_root is a broad multi-project directory (like
        # /home) and candidate_root is some unrelated, deeper, completely
        # different project nested under it. Must NOT match.
        broad_current = self.root / "home"
        unrelated_project = broad_current / "tamfisgpt" / "tamgpt6"
        unrelated_project.mkdir(parents=True)
        self.assertFalse(_workspace_roots_related(str(unrelated_project), str(broad_current)))

    def test_sibling_projects_under_a_shared_broad_ancestor_are_not_related(self):
        broad = self.root / "home"
        site_a = broad / "finima" / "www"
        site_b = broad / "tamfisgpt" / "tamgpt6"
        site_a.mkdir(parents=True)
        site_b.mkdir(parents=True)
        self.assertFalse(_workspace_roots_related(str(site_b), str(site_a)))

    def test_empty_roots_are_not_related(self):
        self.assertFalse(_workspace_roots_related("", str(self.root)))
        self.assertFalse(_workspace_roots_related(str(self.root), ""))


if __name__ == "__main__":
    unittest.main()


class SessionPinnedResumeTests(unittest.TestCase):
    """A resume-shaped turn must never borrow another terminal's state."""

    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = state_module.CONFIG_DIR / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()

    def test_continue_cannot_select_newer_checkpoint_from_another_session(self):
        root = Path(self.tmp.name) / "project"
        root.mkdir()
        state_module.save_session_state(10, workspace_root=str(root), execution_status="idle")
        state_module.save_session_state(
            11,
            workspace_root=str(root),
            execution_status="interrupted",
            turn_checkpoint={"status": "interrupted", "objective": "other terminal task"},
            conversation_history=[{"role": "user", "content": "other terminal task"}],
        )

        selected = _select_resume_state(10, str(root))

        self.assertEqual(selected.session_id, 10)
        self.assertIsNone(selected.turn_checkpoint)
        self.assertEqual(selected.conversation_history, [])

    def test_current_session_checkpoint_remains_available(self):
        root = Path(self.tmp.name) / "project"
        root.mkdir()
        state_module.save_session_state(
            10,
            workspace_root=str(root),
            execution_status="interrupted",
            turn_checkpoint={"status": "interrupted", "objective": "this terminal task"},
        )
        state_module.save_session_state(
            11,
            workspace_root=str(root),
            execution_status="interrupted",
            turn_checkpoint={"status": "interrupted", "objective": "other terminal task"},
        )

        selected = _select_resume_state(10, str(root))

        self.assertEqual(selected.session_id, 10)
        self.assertEqual(selected.turn_checkpoint["objective"], "this terminal task")
