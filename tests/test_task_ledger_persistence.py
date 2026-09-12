"""TaskLedger is the compaction-safe anchor thread compression will
rebuild active context from (see runtime/ledger.py's module docstring).
These tests drive the REAL AgentOrchestrator path (begin/checkpoint/
complete/fail) and assert the ledger a session's /status reads is kept in
sync with the run's own plan/route/objective -- the same facts already
tracked elsewhere, not a second parallel copy that can drift.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.orchestrator.engine import AgentOrchestrator
from tamfis_code.runtime import RuntimeBudgets
from tamfis_code.runtime import ledger as ledger_module


class _IsolatedStorage(unittest.TestCase):
    def setUp(self):
        self._orig_config_dir = state_module.CONFIG_DIR
        self._orig_state_path = state_module.STATE_PATH
        self._orig_ledger_dir = ledger_module.LEDGER_DIR
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        ledger_module.LEDGER_DIR = base / ".config" / "ledgers"

    def tearDown(self):
        state_module.CONFIG_DIR = self._orig_config_dir
        state_module.STATE_PATH = self._orig_state_path
        ledger_module.LEDGER_DIR = self._orig_ledger_dir
        self.tmp.cleanup()


def _orchestrator(session_id: int) -> AgentOrchestrator:
    return AgentOrchestrator(
        session_id=session_id, workspace_root="/tmp", emit=lambda event: None,
        budgets=RuntimeBudgets(max_runtime_seconds=900, max_runtime_extensions=100),
    )


class TaskLedgerLifecycleTests(_IsolatedStorage):
    def test_begin_creates_a_running_ledger(self):
        orchestrator = _orchestrator(1)
        run = orchestrator.begin(objective="fix the bug", messages=[], read_only=False)

        loaded = ledger_module.load_ledger(str(1))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.status, "running")
        self.assertEqual(loaded.objective, "fix the bug")
        self.assertEqual(loaded.repo_roots, ["/tmp"])
        # TaskLedger's own default is checkpoint_version=1; save_task_ledger
        # increments on every save, including this first one.
        self.assertEqual(loaded.checkpoint_version, 2)
        self.assertEqual(loaded.integrity_errors(), [])

    def test_epoch_renewal_checkpoints_the_ledger(self):
        orchestrator = _orchestrator(2)
        orchestrator.begin(objective="long audit", messages=[], read_only=False)
        before = ledger_module.load_ledger(str(2))

        orchestrator._checkpoint_before_epoch_renewal()

        after = ledger_module.load_ledger(str(2))
        self.assertEqual(after.status, "checkpointing")
        self.assertGreater(after.checkpoint_version, before.checkpoint_version)
        self.assertIn("renew execution epoch", after.next_action)
        self.assertEqual(after.integrity_errors(), [])

    def test_complete_marks_ledger_completed(self):
        orchestrator = _orchestrator(3)
        orchestrator.begin(objective="add tests", messages=[], read_only=False)
        orchestrator.start_execution()

        orchestrator.complete(final_text="Done: tests added.", any_mutation=True)

        ledger = ledger_module.load_ledger(str(3))
        self.assertEqual(ledger.status, "completed")
        self.assertEqual(ledger.next_action, "none")

    def test_fail_marks_ledger_failed(self):
        orchestrator = _orchestrator(4)
        orchestrator.begin(objective="risky change", messages=[], read_only=False)

        orchestrator.fail("Blocked repeated action: no progress possible.")

        ledger = ledger_module.load_ledger(str(4))
        self.assertEqual(ledger.status, "failed")

    def test_plan_progress_is_reflected_in_ledger(self):
        orchestrator = _orchestrator(5)
        run = orchestrator.begin(objective="multi-step audit", messages=[], read_only=False)
        self.assertIsNotNone(run.plan)
        run.plan.steps[0].status = "completed"

        orchestrator._checkpoint_before_epoch_renewal()

        ledger = ledger_module.load_ledger(str(5))
        self.assertEqual(ledger.plan_steps[0].status, "completed")
        self.assertGreaterEqual(ledger.current_step_index, 1)

    def test_ledger_survives_across_orchestrator_instances(self):
        # A fresh process (or a new AgentOrchestrator for the same session
        # after a crash) must see the same durable ledger -- this is the
        # anchor a real resume would reload from.
        first = _orchestrator(6)
        first.begin(objective="cross-process task", messages=[], read_only=False)
        first._checkpoint_before_epoch_renewal()

        reloaded = ledger_module.load_ledger(str(6))
        self.assertEqual(reloaded.session_id, 6)
        self.assertEqual(reloaded.objective, "cross-process task")


if __name__ == "__main__":
    unittest.main()
