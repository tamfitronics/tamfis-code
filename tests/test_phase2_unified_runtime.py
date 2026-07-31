from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from tamfis_code.runtime.unified import ExecutionMode, ExecutionRequest, UnifiedAgentRuntime


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect state.json/.memory to a throwaway dir so these tests never
    touch a real user's session state."""
    import tamfis_code.state as state

    monkeypatch.setattr(state, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(state, "STATE_PATH", tmp_path / "state.json")
    state._VOLATILE_STATE.clear()
    return state


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


@pytest.mark.asyncio
async def test_runtime_does_not_turn_unavailable_telemetry_into_task_failure(monkeypatch):
    import tamfis_code.runtime.unified as unified

    def unavailable(*args, **kwargs):
        raise OSError("read-only runtime storage")

    monkeypatch.setattr(unified, "append_event", unavailable)
    monkeypatch.setattr(unified, "save_checkpoint", unavailable)
    runtime = UnifiedAgentRuntime()
    result = await runtime._run_exclusive(
        ExecutionRequest(ExecutionMode.LOCAL_AGENT, objective="answer"),
        lambda: asyncio.sleep(0, result=Outcome()),
    )
    assert result.status == "completed"
    assert any("journal unavailable" in warning for warning in runtime.persistence_warnings)
    assert any("checkpoint unavailable" in warning for warning in runtime.persistence_warnings)


def test_session_state_falls_back_to_volatile_process_storage(monkeypatch):
    import tamfis_code.state as state

    original = dict(state._VOLATILE_STATE)
    state._VOLATILE_STATE.clear()
    monkeypatch.setattr(state, "_save_raw", lambda data: (_ for _ in ()).throw(OSError("read-only")))
    try:
        state.save_session_state(404, active_task={"objective": "continue"})
        assert state.get_session_state(404).active_task == {"objective": "continue"}
    finally:
        state._VOLATILE_STATE.clear()
        state._VOLATILE_STATE.update(original)


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


def _remote_kwargs(**overrides):
    base = dict(
        client=AsyncMock(), renderer=None, console=None,
        session_id=50, objective="fix the bug", mode="coding",
        approval_policy="ask", interactive=True, model="auto", provider="auto",
        attachments=None, config=None,
    )
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_execute_remote_falls_back_to_local_on_infra_failure(monkeypatch, isolated_state):
    import tamfis_code.runner as runner
    import tamfis_code.runner_local as runner_local
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    monkeypatch.setattr(
        runner, "_run_ai_task_and_stream_impl",
        AsyncMock(return_value=Outcome(status="failed", error="Remote agent runtime is unavailable")),
    )
    local_impl = AsyncMock(return_value=Outcome(status="completed", summary="done locally"))
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    runtime = UnifiedAgentRuntime()
    result = await runtime.execute_remote(**_remote_kwargs())

    local_impl.assert_awaited_once()
    assert result.status == "completed"
    assert isolated_state.get_session_state(50).remote_degraded_since is not None


@pytest.mark.asyncio
async def test_execute_remote_does_not_fallback_on_ordinary_task_failure(monkeypatch, isolated_state):
    import tamfis_code.runner as runner
    import tamfis_code.runner_local as runner_local
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    monkeypatch.setattr(
        runner, "_run_ai_task_and_stream_impl",
        AsyncMock(return_value=Outcome(status="failed", error="2 of 5 tests failed")),
    )
    local_impl = AsyncMock()
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    runtime = UnifiedAgentRuntime()
    result = await runtime.execute_remote(**_remote_kwargs())

    local_impl.assert_not_called()
    assert result.status == "failed"
    assert isolated_state.get_session_state(50).remote_degraded_since is None


@pytest.mark.asyncio
async def test_execute_remote_falls_back_on_connection_exception(monkeypatch, isolated_state):
    import tamfis_code.runner as runner
    import tamfis_code.runner_local as runner_local
    from tamfis_code.api_client import RemoteAPIError
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    monkeypatch.setattr(
        runner, "_run_ai_task_and_stream_impl",
        AsyncMock(side_effect=RemoteAPIError(503, "bad gateway")),
    )
    local_impl = AsyncMock(return_value=Outcome(status="completed", summary="done locally"))
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    runtime = UnifiedAgentRuntime()
    result = await runtime.execute_remote(**_remote_kwargs())

    local_impl.assert_awaited_once()
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_execute_remote_does_not_fallback_on_auth_error(monkeypatch, isolated_state):
    import tamfis_code.runner as runner
    import tamfis_code.runner_local as runner_local
    from tamfis_code.api_client import AuthRequiredError
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    monkeypatch.setattr(
        runner, "_run_ai_task_and_stream_impl",
        AsyncMock(side_effect=AuthRequiredError(401, "not logged in")),
    )
    local_impl = AsyncMock()
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    runtime = UnifiedAgentRuntime()
    with pytest.raises(AuthRequiredError):
        await runtime.execute_remote(**_remote_kwargs())

    local_impl.assert_not_called()


@pytest.mark.asyncio
async def test_execute_remote_sticky_degraded_skips_remote_until_probe_succeeds(monkeypatch, isolated_state):
    import tamfis_code.runner as runner
    import tamfis_code.runner_local as runner_local
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    isolated_state.mark_remote_degraded(50, reason="prior outage")
    remote_impl = AsyncMock(return_value=Outcome(status="completed"))
    monkeypatch.setattr(runner, "_run_ai_task_and_stream_impl", remote_impl)
    local_impl = AsyncMock(return_value=Outcome(status="completed", summary="done locally"))
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    # Health probe still failing: stays local, remote is never attempted.
    client = AsyncMock()
    client.get_session.side_effect = RuntimeError("still down")
    runtime = UnifiedAgentRuntime()
    result = await runtime.execute_remote(**_remote_kwargs(client=client))
    remote_impl.assert_not_called()
    local_impl.assert_awaited_once()
    assert result.summary == "done locally"
    assert isolated_state.get_session_state(50).remote_degraded_since is not None

    # Health probe now succeeding: clears the flag and resumes Remote.
    client.get_session.side_effect = None
    client.get_session.return_value = {"id": 50, "status": "idle"}
    result = await runtime.execute_remote(**_remote_kwargs(client=client))
    remote_impl.assert_awaited_once()
    assert result.status == "completed"
    assert isolated_state.get_session_state(50).remote_degraded_since is None
