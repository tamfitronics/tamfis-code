from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tamfis_code.runtime.unified import ExecutionMode, ExecutionRequest, UnifiedAgentRuntime


@dataclass
class Outcome:
    status: str = "completed"
    summary: str = "ok"
    error: str | None = None


@pytest.mark.asyncio
async def test_runtime_serialises_execution_and_records_completion():
    runtime = UnifiedAgentRuntime()
    request = ExecutionRequest(ExecutionMode.LOCAL_AGENT, session_id=7, objective="test")
    gate = asyncio.Event()

    async def operation():
        await gate.wait()
        return Outcome()

    first = asyncio.create_task(runtime._run_exclusive(request, operation))
    await asyncio.sleep(0)
    assert runtime.active
    with pytest.raises(RuntimeError, match="already active"):
        await runtime._run_exclusive(request, operation)
    gate.set()
    result = await first
    assert result.status == "completed"
    assert runtime.history[-1].status == "completed"
    assert not runtime.active


@pytest.mark.asyncio
async def test_runtime_cancel_cancels_active_operation():
    runtime = UnifiedAgentRuntime()
    request = ExecutionRequest(ExecutionMode.LOCAL_AGENT, session_id=9, objective="cancel")

    async def operation():
        await asyncio.Event().wait()

    task = asyncio.create_task(runtime._run_exclusive(request, operation))
    await asyncio.sleep(0)
    assert runtime.cancel() is True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.history[-1].status == "cancelled"
    assert not runtime.active


def test_execution_request_modes_are_explicit():
    assert {mode.value for mode in ExecutionMode} == {
        "local_agent", "remote_agent", "local_chat", "local_stream"
    }

@pytest.mark.asyncio
async def test_runtime_persists_failure_journal(monkeypatch, tmp_path):
    import tamfis_code.runtime.journal as journal
    monkeypatch.setattr(journal, "JOURNAL_PATH", tmp_path / "runtime-events.jsonl")
    monkeypatch.setattr("tamfis_code.runtime.unified.append_event", journal.append_event)
    runtime = UnifiedAgentRuntime()
    request = ExecutionRequest(ExecutionMode.LOCAL_AGENT, session_id=12, objective="fail")

    async def operation():
        raise ValueError("classified failure")

    with pytest.raises(ValueError, match="classified failure"):
        await runtime._run_exclusive(request, operation)

    events = journal.read_recent_events()
    assert [event["event"] for event in events] == ["execution_started", "execution_finished"]
    assert events[-1]["status"] == "failed"
    assert "classified failure" in events[-1]["error"]
    assert events[-1]["execution_id"]
