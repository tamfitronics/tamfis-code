"""Regression tests for checkpoint file-fingerprinting.

Confirmed live: resuming an interrupted task re-ran a full "reading N
files, listing directories" discovery pass regardless of whether anything
on disk had actually changed since the interruption -- indistinguishable
from restarting the task from scratch, and unable to tell the model when a
touched file really had been edited by another coder or process in the
meantime. save_turn_checkpoint now fingerprints (mtime, size) every file a
resumable turn's tool calls touched, and diff_checkpoint_file_fingerprint
lets a resumed run compare that against the file's current state -- see
runner_local.py's resumed_from_checkpoint branch, which turns the result
into an explicit grounding note instead of leaving the model to guess.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.runner_local import _resume_file_status_note


def _read_file_message(path: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
            },
        ],
    }


class _StateDirFixture:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self.workspace = tempfile.TemporaryDirectory()

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self._tmp.cleanup()
        self.workspace.cleanup()


class SaveTurnCheckpointFingerprintTests(_StateDirFixture, unittest.TestCase):
    def test_fingerprints_every_path_a_tool_call_touched(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        touched = Path(self.workspace.name) / "a.py"
        touched.write_text("print('hi')\n")
        state_module.save_turn_checkpoint(
            1, objective="fix a.py", mode="execute",
            messages=[_read_file_message("a.py")],
        )
        fingerprint = state_module.get_session_state(1).turn_checkpoint["file_fingerprint"]
        self.assertIn("a.py", fingerprint)
        self.assertEqual(fingerprint["a.py"]["size"], touched.stat().st_size)

    def test_a_path_a_tool_call_never_touched_is_not_fingerprinted(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        state_module.save_turn_checkpoint(
            1, objective="fix a.py", mode="execute",
            messages=[{"role": "assistant", "content": "no tool calls here"}],
        )
        fingerprint = state_module.get_session_state(1).turn_checkpoint["file_fingerprint"]
        self.assertEqual(fingerprint, {})

    def test_a_touched_path_that_does_not_exist_is_skipped_not_raised(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        state_module.save_turn_checkpoint(
            1, objective="fix missing.py", mode="execute",
            messages=[_read_file_message("missing.py")],
        )
        fingerprint = state_module.get_session_state(1).turn_checkpoint["file_fingerprint"]
        self.assertEqual(fingerprint, {})


class DiffCheckpointFileFingerprintTests(_StateDirFixture, unittest.TestCase):
    def test_reports_no_changes_when_nothing_moved(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        touched = Path(self.workspace.name) / "a.py"
        touched.write_text("print('hi')\n")
        state_module.save_turn_checkpoint(
            1, objective="fix a.py", mode="execute", messages=[_read_file_message("a.py")],
        )
        checkpoint = state_module.get_session_state(1).turn_checkpoint
        result = state_module.diff_checkpoint_file_fingerprint(checkpoint, self.workspace.name)
        self.assertEqual(result, {"changed": [], "missing": []})

    def test_reports_a_file_another_process_edited_as_changed(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        touched = Path(self.workspace.name) / "a.py"
        touched.write_text("print('hi')\n")
        state_module.save_turn_checkpoint(
            1, objective="fix a.py", mode="execute", messages=[_read_file_message("a.py")],
        )
        checkpoint = state_module.get_session_state(1).turn_checkpoint
        time.sleep(0.01)
        touched.write_text("print('a completely different file now')\n")
        result = state_module.diff_checkpoint_file_fingerprint(checkpoint, self.workspace.name)
        self.assertEqual(result["changed"], ["a.py"])
        self.assertEqual(result["missing"], [])

    def test_reports_a_deleted_file_as_missing(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        touched = Path(self.workspace.name) / "a.py"
        touched.write_text("print('hi')\n")
        state_module.save_turn_checkpoint(
            1, objective="fix a.py", mode="execute", messages=[_read_file_message("a.py")],
        )
        checkpoint = state_module.get_session_state(1).turn_checkpoint
        touched.unlink()
        result = state_module.diff_checkpoint_file_fingerprint(checkpoint, self.workspace.name)
        self.assertEqual(result["changed"], [])
        self.assertEqual(result["missing"], ["a.py"])


class ResumeFileStatusNoteTests(_StateDirFixture, unittest.TestCase):
    """runner_local._resume_file_status_note turns the fingerprint diff into
    the actual grounding message spliced into a resumed run's transcript."""

    def test_no_note_when_the_checkpoint_touched_no_files(self):
        checkpoint = {"file_fingerprint": {}}
        self.assertIsNone(_resume_file_status_note(checkpoint, self.workspace.name))

    def test_confirms_nothing_changed_when_fingerprint_still_matches(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        touched = Path(self.workspace.name) / "a.py"
        touched.write_text("print('hi')\n")
        state_module.save_turn_checkpoint(
            1, objective="fix a.py", mode="execute", messages=[_read_file_message("a.py")],
        )
        checkpoint = state_module.get_session_state(1).turn_checkpoint
        note = _resume_file_status_note(checkpoint, self.workspace.name)
        self.assertEqual(note["role"], "system")
        self.assertIn("continue directly", note["content"])

    def test_flags_a_file_changed_by_another_coder(self):
        state_module.save_session_state(1, workspace_root=self.workspace.name)
        touched = Path(self.workspace.name) / "a.py"
        touched.write_text("print('hi')\n")
        state_module.save_turn_checkpoint(
            1, objective="fix a.py", mode="execute", messages=[_read_file_message("a.py")],
        )
        checkpoint = state_module.get_session_state(1).turn_checkpoint
        time.sleep(0.01)
        touched.write_text("someone else's edit\n")
        note = _resume_file_status_note(checkpoint, self.workspace.name)
        self.assertEqual(note["role"], "system")
        self.assertIn("modified outside this session", note["content"])
        self.assertIn("a.py", note["content"])
