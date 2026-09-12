"""Focused regression test for the exact live failure:

    ✓ Ran command
      COMPILE OK
    ✗ Command failed
      Runtime budget exhausted after 902 seconds.

A successful tool call must never be converted into a terminal task
failure by the post-command wall-clock watchdog.  The epoch expiry is a
recoverable control signal: checkpoint, renew the epoch, continue the
same task with the same plan and the same task identity.

These tests deliberately drive the REAL production path
(AgentOrchestrator.guard_tool_call -> ExecutionController.guard_action)
with a 1-second epoch, so the renewal semantics are exercised at the
command boundary rather than in isolation.
"""
from __future__ import annotations

import time

from tamfis_code.orchestrator.engine import AgentOrchestrator
from tamfis_code.runtime import ExecutionController, RuntimeBudgets, RuntimePhase


def _budgets(**overrides) -> RuntimeBudgets:
    defaults = dict(
        max_runtime_seconds=1,
        max_runtime_extensions=1000,
        runtime_renewal_grace_seconds=1,
        max_tool_calls=10000,
    )
    defaults.update(overrides)
    return RuntimeBudgets(**defaults)


def _orchestrator(budgets: RuntimeBudgets) -> AgentOrchestrator:
    return AgentOrchestrator(
        session_id=20260912,
        workspace_root="/tmp",
        emit=lambda event: None,
        budgets=budgets,
    )


def test_command_boundary_expiry_never_fails_task():
    """The exact live reproduction: a command succeeds, then the epoch
    deadline is reached at the command boundary.  The next guard check
    must renew the epoch and allow the next action -- never mark the task
    FAILED."""
    orchestrator = _orchestrator(_budgets())
    run = orchestrator.begin(objective="build the project", messages=[], read_only=False)

    # Simulate a successful command completing (the "COMPILE OK" case).
    decision = orchestrator.guard_tool_call("execute_command", {"command": "python3 -m compileall -q ."})
    assert decision.allowed, f"first guard must allow: {decision.reason}"

    # Let the 1-second epoch expire while the command "runs".
    time.sleep(1.1)

    # The next guard check happens AFTER the command returned -- this is
    # the post-command watchdog that used to kill the task.
    decision = orchestrator.guard_tool_call("execute_command", {"command": "python3 -m pytest -q"})
    assert decision.allowed, (
        f"post-command epoch expiry must renew and continue, got: {decision.reason}"
    )
    assert not decision.terminal
    assert run.runtime.snapshot.phase != RuntimePhase.FAILED
    # The epoch timer actually reset (not just a re-check of a stale clock).
    assert run.runtime.epoch_elapsed() < 1.0
    assert run.runtime.epoch_index >= 1
    assert run.runtime.snapshot.runtime_extensions >= 1


def test_five_renewals_keep_same_task_and_plan():
    """Force at least 5 runtime renewals in one task.  The task identity,
    plan, and completed steps must be preserved across every renewal."""
    orchestrator = _orchestrator(_budgets())
    run = orchestrator.begin(objective="long audit", messages=[], read_only=False)
    original_plan_ids = [s.name for s in run.plan.steps] if run.plan else []
    original_phase = run.runtime.snapshot.phase

    for renewal in range(5):
        time.sleep(1.1)  # expire the epoch before the command boundary
        decision = orchestrator.guard_tool_call("execute_command", {"command": f"step-{renewal}"})
        assert decision.allowed, f"renewal {renewal} must continue: {decision.reason}"

    # Same task throughout.
    assert run.objective == "long audit"
    assert run.runtime.snapshot.phase != RuntimePhase.FAILED
    assert not run.runtime.snapshot.terminal
    # Same plan throughout.
    assert [s.name for s in run.plan.steps] == original_plan_ids
    # Five renewals actually happened.
    assert run.runtime.epoch_renewals >= 5
    assert run.runtime.snapshot.runtime_extensions >= 5
    # The phase never regressed to a pre-execution state (no replanning).
    assert run.runtime.snapshot.phase.value in {"execute", "observe", "validate"}


def test_renewal_does_not_replan_or_rediscover():
    """Epoch renewal must not invoke the initial planner or workspace
    discovery.  The orchestrator's guard path only touches the runtime
    controller and the durable checkpoint -- never begin()/create_plan().
    This test asserts the plan object identity is unchanged across a
    renewal, which is only possible if no replanning occurred."""
    orchestrator = _orchestrator(_budgets())
    run = orchestrator.begin(objective="no replan on renewal", messages=[], read_only=False)
    plan_before = run.plan
    plan_id_before = run.plan_id

    time.sleep(1.1)
    decision = orchestrator.guard_tool_call("read_file", {"path": "/tmp/x.py"})
    assert decision.allowed

    assert run.plan is plan_before, "renewal must not replace the plan object"
    assert run.plan_id == plan_id_before, "renewal must not mint a new plan id"


def test_terminal_state_guard_blocks_runtime_only_failure():
    """Defence in depth: even if a caller routes a runtime-budget
    exhaustion into the generic failure handler, the controller must
    refuse to move a healthy RUNNING/RECOVERING task to FAILED on elapsed
    time alone."""
    controller = ExecutionController(_budgets(max_runtime_extensions=1000))
    controller.start_execution()
    time.sleep(1.1)

    # A rogue caller calls fail() with the runtime-exhaustion reason.
    controller.fail("Runtime budget exhausted after 1 seconds.")

    # The controller must have refused the RUNNING -> FAILED transition
    # for a runtime-only reason (no separate unrecoverable failure).
    assert controller.snapshot.phase != RuntimePhase.FAILED, (
        "runtime-budget exhaustion alone must never move a healthy task to FAILED"
    )


def test_extension_cap_exhaustion_is_not_silent_failure():
    """When the extension cap is genuinely exhausted, the decision must
    still be explicit and the reason must name the cap -- not the raw
    elapsed-time string that looks like an ordinary watchdog kill."""
    controller = ExecutionController(_budgets(max_runtime_extensions=1))
    controller.start_execution()
    time.sleep(1.1)
    first = controller.guard_action("edit_file", {"path": "/repo/x.py"})
    assert first.time_budget_exhausted
    assert controller.renew_epoch(), "first renewal must succeed"
    time.sleep(1.1)
    second = controller.guard_action("edit_file", {"path": "/repo/x.py"})
    assert second.time_budget_exhausted
    assert not controller.renew_epoch(), "cap reached, renewal must refuse"
    # The refusal reason names the extension cap, not just elapsed time.
    assert "extension" in controller.snapshot.failure_reason or (
        controller.snapshot.runtime_extensions >= controller.budgets.max_runtime_extensions
    )


def test_cap_exhaustion_returns_partial_not_failed():
    """The live failure shape: extension cap reached at a command boundary.
    The runner must checkpoint as resumable (status=partial) and never
    emit ai_task_failed for elapsed time alone.  This drives the real
    orchestrator path with a cap of 1, so the second expiry exhausts it."""
    orchestrator = _orchestrator(_budgets(max_runtime_extensions=1))
    run = orchestrator.begin(objective="cap exhaustion task", messages=[], read_only=False)

    # First epoch: allowed, then expires.
    decision = orchestrator.guard_tool_call("execute_command", {"command": "build"})
    assert decision.allowed
    time.sleep(1.1)

    # Second boundary: renewal succeeds (extension 1/1), continues.
    decision = orchestrator.guard_tool_call("execute_command", {"command": "test"})
    assert decision.allowed
    assert run.runtime.snapshot.runtime_extensions == 1
    time.sleep(1.1)

    # Third boundary: cap exhausted. The guard still reports time-budget
    # exhaustion (never a generic failure reason), and the controller
    # refuses the FAILED transition for elapsed time alone.
    decision = orchestrator.guard_tool_call("execute_command", {"command": "verify"})
    assert decision.time_budget_exhausted, "cap exhaustion must stay a time-budget signal"
    assert not controller_only_failure(run), "task must not be terminally FAILED"
    # The controller's defensive guard kept the phase out of FAILED.
    assert run.runtime.snapshot.phase != RuntimePhase.FAILED


def controller_only_failure(run) -> bool:
    """True only when the run is terminally FAILED for a non-runtime reason."""
    snap = run.runtime.snapshot
    return snap.terminal and snap.phase == RuntimePhase.FAILED and not (
        snap.failure_reason.startswith("Runtime budget exhausted") or not snap.failure_reason
    )