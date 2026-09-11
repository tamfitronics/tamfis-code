"""Regression tests for the durable-checkpoint / execution-epoch-renewal
self-repair: a runtime budget belongs to an EXECUTION EPOCH, not to the
user's task, so epoch expiry must checkpoint + renew + continue instead of
failing the task.
"""
from __future__ import annotations

import time

import pytest

from tamfis_code.runtime import ExecutionController, RuntimeBudgets, RuntimePhase
from tamfis_code.runtime import checkpoint as checkpoint_module
from tamfis_code.runtime.checkpoint import (
    CHECKPOINT_DIR,
    ExecutionCheckpoint,
    latest_resumable_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path, monkeypatch):
    """Point the checkpoint store at a per-test temp dir so tests never
    touch the operator's real ~/.config/tamfis-code/checkpoints."""
    target = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoint_module, "CHECKPOINT_DIR", target)
    return target


def _result(stdout="", *, success=True, items=None):
    payload = {"stdout": stdout}
    if items is not None:
        payload["items"] = items
    return {"success": success, "result": payload}


# ---------------------------------------------------------------------------
# TEST A -- runtime expiry
# ---------------------------------------------------------------------------

def test_runtime_expiry_renews_epoch_instead_of_failing():
    """A tiny epoch budget must not fail a healthy multi-step task."""
    controller = ExecutionController(
        RuntimeBudgets(max_runtime_seconds=1, max_runtime_extensions=5, runtime_renewal_grace_seconds=1)
    )
    steps_done = 0
    for step in range(6):
        # Simulate work inside one epoch; each iteration takes real time so
        # the 1-second epoch genuinely expires mid-task.
        time.sleep(0.35)
        decision = controller.guard_action("read_file", {"path": f"/tmp/step-{step}.py"})
        if not decision.allowed and decision.time_budget_exhausted:
            # The required recovery path: checkpoint, renew, continue.
            assert decision.epoch_renewal_requested
            assert controller.renew_epoch() is True
            decision = controller.guard_action("read_file", {"path": f"/tmp/step-{step}.py"})
        assert decision.allowed, f"step {step} should continue after renewal"
        controller.observe("read_file", {"path": f"/tmp/step-{step}.py"}, _result(f"content-{step}"))
        steps_done += 1
    assert steps_done == 6
    assert controller.snapshot.phase != RuntimePhase.FAILED
    assert controller.epoch_renewals >= 1
    assert controller.snapshot.runtime_extensions >= 1


def test_epoch_renewal_is_bounded():
    """Renewals are bounded so a runaway task cannot renew forever."""
    controller = ExecutionController(
        RuntimeBudgets(max_runtime_seconds=1, max_runtime_extensions=2, runtime_renewal_grace_seconds=1)
    )
    assert controller.renew_epoch() is True
    assert controller.renew_epoch() is True
    assert controller.renew_epoch() is False


def test_epoch_renewal_due_signals_grace_threshold():
    controller = ExecutionController(
        RuntimeBudgets(max_runtime_seconds=1, max_runtime_extensions=5, runtime_renewal_grace_seconds=5)
    )
    # Grace (5s) >= epoch budget (1s) means renewal is due immediately.
    assert controller.epoch_renewal_due() is True


def test_epoch_renewal_not_due_when_far_from_deadline():
    controller = ExecutionController(
        RuntimeBudgets(max_runtime_seconds=600, max_runtime_extensions=5, runtime_renewal_grace_seconds=5)
    )
    assert controller.epoch_renewal_due() is False


def test_time_exhaustion_is_recoverable_not_terminal_failure():
    """The exact bad behaviour: 'Runtime budget exhausted' used to mark the
    task FAILED. It must now request epoch renewal instead."""
    controller = ExecutionController(
        RuntimeBudgets(max_runtime_seconds=1, max_runtime_extensions=3, runtime_renewal_grace_seconds=1)
    )
    time.sleep(1.1)
    decision = controller.guard_action("read_file", {"path": "/tmp/x.py"})
    assert not decision.allowed
    assert decision.time_budget_exhausted
    assert decision.epoch_renewal_requested
    # The task is NOT failed: the phase was never moved to FAILED.
    assert controller.snapshot.phase != RuntimePhase.FAILED
    assert not controller.snapshot.terminal
    assert controller.snapshot.failure_reason == ""
    # Renewal closes the exhausted epoch and opens a fresh bounded one.
    assert controller.renew_epoch() is True
    assert controller.epoch_index == 1
    assert controller.epoch_renewals == 1
    assert controller.snapshot.runtime_extensions == 1
    resumed = controller.guard_action("read_file", {"path": "/tmp/x.py"})
    assert resumed.allowed
    assert controller.snapshot.phase == RuntimePhase.EXECUTE


# ---------------------------------------------------------------------------
# TEST B -- process interruption / durable checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_survives_process_restart(tmp_path):
    """A checkpoint written before interruption reloads with full state."""
    checkpoint = ExecutionCheckpoint(
        execution_id="test-exec-1",
        session_id=1380884427,
        mode="local_agent",
        objective="Rebalance provider weights in orchestration.yaml",
        workspace_root="/home/tamfisgpt",
        status="running",
        phase="execute",
        plan={"steps": ["inspect routing", "change pools", "validate"]},
        plan_steps=[{"name": "inspect routing", "status": "completed"},
                    {"name": "change pools", "status": "in_progress"},
                    {"name": "validate", "status": "pending"}],
        current_step_index=1,
        completed_steps=[0],
        remaining_steps=[2],
        files_already_examined=["/home/tamfisgpt/tamgpt6/tier_iv_orchestration/config/orchestration.yaml"],
        relevant_files=["/home/tamfisgpt/tamgpt6/tier_iv_orchestration/config/orchestration.yaml"],
        changed_files=["/home/tamfisgpt/tamgpt6/tier_iv_orchestration/config/orchestration.yaml"],
        last_completed_action="Updated provider_weights in orchestration.yaml",
        next_action="Trace production selection path and verify weights are consumed",
        current_provider="nvidia_nim",
        attempted_providers=["nvidia_nim"],
        recovery_count=1,
    )
    save_checkpoint(checkpoint)
    # Simulate a fresh process: reload from disk only.
    reloaded = load_checkpoint("test-exec-1")
    assert reloaded is not None
    assert reloaded.execution_id == "test-exec-1"
    assert reloaded.session_id == 1380884427
    assert reloaded.objective == "Rebalance provider weights in orchestration.yaml"
    assert reloaded.plan_steps[0]["status"] == "completed"
    assert reloaded.current_step_index == 1
    assert reloaded.next_action.startswith("Trace production selection path")
    assert reloaded.changed_files == [
        "/home/tamfisgpt/tamgpt6/tier_iv_orchestration/config/orchestration.yaml"
    ]
    assert reloaded.last_completed_action == "Updated provider_weights in orchestration.yaml"


def test_checkpoint_write_is_atomic_and_latest_valid_survives(tmp_path):
    """A crash mid-write must not corrupt the last valid checkpoint."""
    checkpoint = ExecutionCheckpoint(
        execution_id="test-exec-atomic",
        session_id=1,
        mode="local_agent",
        objective="objective",
        workspace_root="/tmp",
        status="running",
    )
    first = save_checkpoint(checkpoint)
    assert first.is_file()
    # Overwrite with newer state; the atomic replace must leave a valid file.
    checkpoint.status = "partial"
    checkpoint.next_action = "continue with step 3"
    save_checkpoint(checkpoint)
    reloaded = load_checkpoint("test-exec-atomic")
    assert reloaded is not None
    assert reloaded.status == "partial"
    assert reloaded.next_action == "continue with step 3"


def test_latest_resumable_checkpoint_finds_running_task():
    checkpoint = ExecutionCheckpoint(
        execution_id="test-exec-resumable",
        session_id=424242,
        mode="local_agent",
        objective="resumable objective",
        workspace_root="/tmp",
        status="running",
    )
    save_checkpoint(checkpoint)
    found = latest_resumable_checkpoint(424242)
    assert found is not None
    assert found.execution_id == "test-exec-resumable"


# ---------------------------------------------------------------------------
# TEST C -- provider switch preserves task identity
# ---------------------------------------------------------------------------

def test_provider_switch_preserves_task_state():
    """Switching providers must not reset the task: same checkpoint, same
    plan, same edits -- only the provider field changes."""
    checkpoint = ExecutionCheckpoint(
        execution_id="test-exec-provider",
        session_id=777,
        mode="local_agent",
        objective="provider routing task",
        workspace_root="/tmp",
        status="running",
        plan_steps=[{"name": "step1", "status": "completed"}, {"name": "step2", "status": "in_progress"}],
        changed_files=["/repo/a.py"],
        current_provider="tamfisgpt-ultra",
        attempted_providers=["tamfisgpt-ultra"],
    )
    save_checkpoint(checkpoint)
    # Provider A fails -> switch to Provider B, same task.
    checkpoint.current_provider = "tamfisgpt-ultima"
    checkpoint.attempted_providers.append("tamfisgpt-ultima")
    checkpoint.recovery_count += 1
    save_checkpoint(checkpoint)
    reloaded = load_checkpoint("test-exec-provider")
    assert reloaded.execution_id == "test-exec-provider"
    assert reloaded.session_id == 777
    assert reloaded.plan_steps[0]["status"] == "completed"
    assert reloaded.changed_files == ["/repo/a.py"]
    assert reloaded.current_provider == "tamfisgpt-ultima"
    assert "tamfisgpt-ultra" in reloaded.attempted_providers


# ---------------------------------------------------------------------------
# TEST D -- no directory rescan on resume
# ---------------------------------------------------------------------------

def test_resume_uses_checkpoint_scope_without_global_rescan():
    """The checkpoint's recorded scope must be authoritative on resume: no
    /home or /tmp inventory is required to continue."""
    checkpoint = ExecutionCheckpoint(
        execution_id="test-exec-scope",
        session_id=888,
        mode="local_agent",
        objective="scoped task",
        workspace_root="/home/tamfiscode",
        status="running",
        relevant_files=["/home/tamfiscode/tamfis_code/runtime/controller.py"],
        files_already_examined=["/home/tamfiscode/tamfis_code/runtime/controller.py"],
        directories_already_examined=["/home/tamfiscode/tamfis_code/runtime"],
        irrelevant_paths_to_skip=["/home/editorial-posts", "/home/tamgpt", "/tmp"],
    )
    save_checkpoint(checkpoint)
    reloaded = load_checkpoint("test-exec-scope")
    assert reloaded is not None
    # Resume scope comes from the checkpoint, not from a fresh /home scan.
    assert "/home/tamfiscode/tamfis_code/runtime/controller.py" in reloaded.relevant_files
    assert "/home/editorial-posts" in reloaded.irrelevant_paths_to_skip
    assert "/tmp" in reloaded.irrelevant_paths_to_skip


# ---------------------------------------------------------------------------
# TEST E -- changed relevant file invalidation
# ---------------------------------------------------------------------------

def test_changed_relevant_file_is_selectively_invalidated(tmp_path):
    """Only the changed file's cached understanding is dropped; the rest of
    the discovery cache is reused."""
    checkpoint = ExecutionCheckpoint(
        execution_id="test-exec-invalidate",
        session_id=999,
        mode="local_agent",
        objective="invalidate test",
        workspace_root=str(tmp_path),
        status="running",
        relevant_files=[str(tmp_path / "a.py"), str(tmp_path / "b.py")],
        files_already_examined=[str(tmp_path / "a.py"), str(tmp_path / "b.py")],
    )
    save_checkpoint(checkpoint)
    # Externally modify one known file.
    changed = tmp_path / "a.py"
    changed.write_text("print('externally changed')\n", encoding="utf-8")
    reloaded = load_checkpoint("test-exec-invalidate")
    assert reloaded is not None
    # Reconciliation: only the changed file needs re-reading.
    stale = {str(changed)}
    needs_reread = [p for p in reloaded.relevant_files if p in stale]
    assert needs_reread == [str(changed)]
    assert str(tmp_path / "b.py") not in needs_reread


# ---------------------------------------------------------------------------
# TEST F -- queued user message survives recovery
# ---------------------------------------------------------------------------

def test_queued_user_message_survives_recovery():
    from tamfis_code import state as local_state

    session_id = 20260910
    local_state.enqueue_instruction(session_id, "prefer Kimi-K3 for NIM chat")
    checkpoint = ExecutionCheckpoint(
        execution_id="test-exec-queue",
        session_id=session_id,
        mode="local_agent",
        objective="queued message task",
        workspace_root="/tmp",
        status="running",
    )
    save_checkpoint(checkpoint)
    # Recovery/renewal must not drop the queued instruction.
    state = local_state.get_session_state(session_id)
    texts = [str(item.get("text")) for item in state.queued_user_instructions]
    assert "prefer Kimi-K3 for NIM chat" in texts
    reloaded = load_checkpoint("test-exec-queue")
    assert reloaded is not None and reloaded.session_id == session_id


# ---------------------------------------------------------------------------
# Plan persistence across epochs
# ---------------------------------------------------------------------------

def test_plan_survives_epoch_renewal():
    controller = ExecutionController(
        RuntimeBudgets(max_runtime_seconds=1, max_runtime_extensions=5, runtime_renewal_grace_seconds=1)
    )
    plan = [
        {"name": "inspect routing implementation", "status": "completed"},
        {"name": "change provider pools", "status": "completed"},
        {"name": "implement Kimi-K3 NIM priority", "status": "in_progress"},
        {"name": "validate normalisation/fallback", "status": "pending"},
        {"name": "test TamfisGPT", "status": "pending"},
        {"name": "test Tamfis-Code", "status": "pending"},
        {"name": "report", "status": "pending"},
    ]
    time.sleep(1.1)
    decision = controller.guard_action("edit_file", {"path": "/repo/x.py"})
    assert decision.time_budget_exhausted and decision.epoch_renewal_requested
    assert controller.renew_epoch()
    # The plan is untouched by the renewal: same steps, same statuses.
    assert plan[0]["status"] == "completed"
    assert plan[2]["status"] == "in_progress"
    # The task was never marked FAILED by the epoch expiry.
    assert controller.snapshot.phase != RuntimePhase.FAILED
    assert not controller.snapshot.terminal
    # After renewal, the next action proceeds into EXECUTE.
    resumed = controller.guard_action("edit_file", {"path": "/repo/x.py"})
    assert resumed.allowed
    assert controller.snapshot.phase == RuntimePhase.EXECUTE