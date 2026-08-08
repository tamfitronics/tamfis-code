from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tamfis_code.runtime.cognitive import (
    EvidenceGraph,
    EvidenceNode,
    ReplanningEngine,
    RequirementStatus,
    TaskContract,
)
from tamfis_code.runtime.repository_index import RepositoryIndex
from tamfis_code.runtime.reviewer import IndependentReviewer
from tamfis_code.runtime.steering import SteeringIntent, classify_live_input
from tamfis_code.runtime.unified import ExecutionMode, ExecutionRequest, UnifiedAgentRuntime


def test_task_contract_requires_mutation_and_validation_evidence():
    contract = TaskContract.derive("Fix the terminal bug", read_only=False, approval_policy="ask")
    assert contract.requested_mutation is True
    assert {c.criterion_id for c in contract.criteria} >= {"objective", "evidence", "mutation", "validation"}


def test_independent_review_blocks_unsupported_completion():
    contract = TaskContract.derive("Implement a fix", read_only=False, approval_policy="ask")
    graph = EvidenceGraph()
    graph.add(EvidenceNode("c", "completion_summary", "done", "result", ["objective"]))
    graph.add(EvidenceNode("o", "tool_observation", "observed", "tool", ["evidence"]))
    review = IndependentReviewer().review(contract, graph)
    assert review.approved is False
    assert any("changed-file" in warning for warning in review.warnings)


def test_replanning_requires_material_change_and_reason():
    revision = ReplanningEngine().revise(
        revision=1,
        reason="Dispatcher schema, not edit_file, is defective",
        previous_steps=["Repair edit_file", "Run tests"],
        replacement_steps=["Repair dispatcher normalisation", "Run tool tests"],
        evidence_ids=["schema-error"],
    )
    assert revision.revision == 1
    with pytest.raises(ValueError):
        ReplanningEngine().revise(revision=2, reason="", previous_steps=["a"], replacement_steps=["b"])


def test_repository_index_reuses_fingerprint(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\n', encoding="utf-8")
    (tmp_path / "x.py").write_text("print('x')\n", encoding="utf-8")
    index = RepositoryIndex(tmp_path, tmp_path / ".index.json")
    first = index.build()
    loaded = index.load()
    assert loaded is not None
    assert loaded.fingerprint == first.fingerprint
    (tmp_path / "x.py").write_text("print('y')\n", encoding="utf-8")
    second = index.build()
    assert second.fingerprint != first.fingerprint


def test_live_yes_answers_visible_approval():
    assert classify_live_input("yes", approval_visible=True) == SteeringIntent.APPROVAL
    assert classify_live_input("yes", approval_visible=False) == SteeringIntent.FOLLOW_UP
    assert classify_live_input("do not change providers", approval_visible=False) == SteeringIntent.CORRECTION


@dataclass
class Result:
    status: str = "completed"
    summary: str = "changed"
    changed_files: list[str] = field(default_factory=lambda: ["a.py"])
    validations: list[dict] = field(default_factory=lambda: [{"passed": True, "command": "pytest"}])


def test_unified_runtime_records_contract_and_review(tmp_path: Path, monkeypatch):
    runtime = UnifiedAgentRuntime()
    request = ExecutionRequest(
        mode=ExecutionMode.LOCAL_AGENT,
        session_id=1,
        objective="Fix the bug",
        workspace_root=str(tmp_path),
        interactive=False,
        approval_policy="ask",
        read_only=False,
    )
    result = asyncio.run(runtime._run_exclusive(request, lambda: asyncio.sleep(0, result=Result())))
    assert result.status == "completed"
    assert runtime.task_contract is not None
    assert runtime.last_review is not None and runtime.last_review.approved is True


def test_unified_runtime_derives_legacy_runner_evidence_from_session_state(tmp_path: Path, monkeypatch):
    import tamfis_code.state as state
    original = state.CONFIG_DIR, state.STATE_PATH
    state.CONFIG_DIR = tmp_path / ".config"
    state.STATE_PATH = state.CONFIG_DIR / "state.json"
    try:
        runtime = UnifiedAgentRuntime()
        request = ExecutionRequest(
            mode=ExecutionMode.LOCAL_AGENT,
            session_id=22,
            objective="Fix the bug",
            workspace_root=str(tmp_path),
            interactive=False,
            approval_policy="auto",
        )

        async def operation():
            state.save_session_state(
                22,
                workspace_root=str(tmp_path),
                modified_files=[{"mutation_id": "m1", "path": str(tmp_path / "app.py")}],
                validation_results=[{"passed": True, "command": "pytest -q"}],
            )
            return Result(status="completed", summary="changed", changed_files=[], validations=[])

        result = asyncio.run(runtime._run_exclusive(request, operation))
        assert result.status == "completed"
        assert result.changed_files == [str(tmp_path / "app.py")]
        assert runtime.last_review is not None and runtime.last_review.approved is True
    finally:
        state.CONFIG_DIR, state.STATE_PATH = original
