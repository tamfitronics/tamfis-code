from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tamfis_code.orchestrator.approvals import ApprovalAction, ApprovalBatch
from tamfis_code.orchestrator.completion import CompletionStatus, determine_completion
from tamfis_code.orchestrator.planner import ExecutionPlan, PlanStep
from tamfis_code.orchestrator.repair import FailureClass, choose_repair
from tamfis_code.runtime.checkpoint import ExecutionCheckpoint, load_checkpoint, save_checkpoint
from tamfis_code.runtime.unified import ExecutionMode, ExecutionRequest, UnifiedAgentRuntime


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Refactor runtime",
        assumptions=[],
        components=[],
        steps=[PlanStep(1, "Inspect"), PlanStep(2, "Edit")],
        validation_criteria=[],
        risks=[],
    )


def test_plan_is_editable_and_reindexed():
    plan = _plan()
    plan.edit_step(1, name="Inspect runtime", status="completed")
    plan.add_step("Validate", after=2)
    removed = plan.remove_step(2)
    assert removed.name == "Edit"
    assert [(s.index, s.name, s.status) for s in plan.steps] == [
        (1, "Inspect runtime", "completed"),
        (2, "Validate", "pending"),
    ]


def test_grouped_approval_reports_highest_risk():
    batch = ApprovalBatch()
    batch.add(ApprovalAction("read_file", {"path": "a.py"}, "inspect", "read_only"))
    batch.add(ApprovalAction("edit_file", {"path": "a.py"}, "repair", "medium"))
    assert batch.requires_prompt is True
    assert batch.highest_risk == "medium"
    assert len(batch.to_dict()["actions"]) == 2


def test_shell_quoting_repair_forces_native_tool():
    decision = choose_repair(
        tool_name="execute_command",
        result="bash: syntax error near unexpected token ( from printf",
        attempt=0,
    )
    assert decision.failure_class == FailureClass.SHELL_QUOTING_ERROR
    assert decision.force_different_tool is True
    assert "native write_file" in decision.strategy


def test_permission_repair_preserves_canonical_workspace():
    decision = choose_repair(
        tool_name="execute_command",
        result="npm: EACCES: permission denied, mkdir '/workspace/dist'",
        attempt=0,
    )
    assert decision.failure_class == FailureClass.PERMISSION_DENIED
    assert decision.retry_allowed is True
    assert "without sudo" in decision.strategy
    assert "copying the project" in decision.strategy


def test_truthful_completion_statuses():
    assert determine_completion(
        requested_mutation=True, changed_files=["a.py"], validation_passed=True, unresolved=[]
    ) == CompletionStatus.COMPLETED
    assert determine_completion(
        requested_mutation=True, changed_files=["a.py"], validation_passed=False, unresolved=["tests failed"]
    ) == CompletionStatus.PARTIAL
    assert determine_completion(
        requested_mutation=True, changed_files=[], validation_passed=False, unresolved=["permission denied"]
    ) == CompletionStatus.BLOCKED


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    import tamfis_code.runtime.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint, "CHECKPOINT_DIR", tmp_path)
    value = ExecutionCheckpoint(
        execution_id="abc", session_id=7, mode="local_agent", objective="fix",
        workspace_root="/tmp/work", status="partial", changed_files=["a.py"],
        unresolved=["validation pending"],
    )
    save_checkpoint(value)
    loaded = load_checkpoint("abc")
    assert loaded is not None
    assert loaded.changed_files == ["a.py"]
    assert loaded.status == "partial"


@dataclass
class _Outcome:
    status: str = "completed"
    summary: str = "done"
    error: str | None = None


@pytest.mark.asyncio
async def test_unified_runtime_writes_completion_checkpoint(monkeypatch, tmp_path):
    import tamfis_code.runtime.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("tamfis_code.runtime.unified.save_checkpoint", checkpoint.save_checkpoint)
    runtime = UnifiedAgentRuntime()
    request = ExecutionRequest(ExecutionMode.LOCAL_AGENT, session_id=3, objective="phase 3")

    async def operation():
        await asyncio.sleep(0)
        return _Outcome()

    await runtime._run_exclusive(request, operation)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    loaded = load_checkpoint(files[0].stem)
    assert loaded is not None
    assert loaded.status == "completed"
