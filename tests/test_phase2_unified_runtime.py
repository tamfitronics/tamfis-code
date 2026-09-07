from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tamfis_code.providers import ProviderType
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


class _FakeChatClient:
    """Minimal stand-in for the AsyncOpenAI client `_run_local_turn_impl` calls."""

    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content
        self.calls = 0

    async def _create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=self._content, tool_calls=None)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeSideManager:
    """Stands in for `/btw`'s own isolated ProviderManager -- just enough
    surface for `_run_local_turn_impl` (get_client, PROVIDERS), independent
    of the real provider pool / API keys."""

    def __init__(self, client: _FakeChatClient):
        self._client = client
        self.PROVIDERS = {
            ProviderType.NVIDIA: SimpleNamespace(
                default_model="fake-model", models=(), free_model=None, context_window=32768,
            ),
        }

    def get_client(self, provider):
        return self._client


@pytest.mark.asyncio
async def test_execute_local_chat_rejects_concurrent_call_while_runtime_active():
    """Documents the bug: the public `run_local_turn` adapter routes through
    `execute_local_chat`, which takes the same exclusive lock as a full
    tool-using agent turn. `/btw` used to call that adapter and would fail
    with exactly this error every time it ran while a task was active --
    the one situation it exists for."""
    runtime = UnifiedAgentRuntime()
    gate = asyncio.Event()

    async def blocking_operation():
        await gate.wait()
        return Outcome()

    holder = asyncio.create_task(
        runtime._run_exclusive(ExecutionRequest(ExecutionMode.LOCAL_AGENT, objective="main task"), blocking_operation)
    )
    await asyncio.sleep(0)
    assert runtime.active

    with pytest.raises(RuntimeError, match="already active"):
        await runtime.execute_local_chat(
            manager=_FakeSideManager(_FakeChatClient("side answer")),
            provider=ProviderType.NVIDIA, messages=[{"role": "user", "content": "hi"}],
            model="fake-model", console=None, use_tools=False,
        )

    gate.set()
    await holder


@pytest.mark.asyncio
async def test_run_local_turn_impl_bypasses_the_runtime_lock_for_side_questions():
    """The fix: `/btw` (interactive.py's `_answer_side_question`) calls
    `local_chat._run_local_turn_impl` directly instead of going through
    `execute_local_chat`, so a side question answers successfully even
    while the main runtime is mid-execution -- it never touches
    `UnifiedAgentRuntime` at all, by design (see that function's
    docstring)."""
    from tamfis_code.local_chat import _run_local_turn_impl

    runtime = UnifiedAgentRuntime()
    gate = asyncio.Event()

    async def blocking_operation():
        await gate.wait()
        return Outcome()

    holder = asyncio.create_task(
        runtime._run_exclusive(ExecutionRequest(ExecutionMode.LOCAL_AGENT, objective="main task"), blocking_operation)
    )
    await asyncio.sleep(0)
    assert runtime.active

    client = _FakeChatClient("side answer")
    answer = await _run_local_turn_impl(
        _FakeSideManager(client), ProviderType.NVIDIA, [{"role": "user", "content": "who created AI?"}],
        "fake-model", None, use_tools=False,
    )

    assert answer == "side answer"
    assert client.calls == 1
    assert runtime.active  # the main task is still untouched, still running

    gate.set()
    await holder


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
async def test_execute_remote_always_runs_the_local_provider(monkeypatch, isolated_state):
    """The local engine (runner_local.py) is the richer, better-tested
    implementation; a --remote task must never delegate reasoning to the
    Remote backend's own agent loop -- see execute_remote's docstring."""
    import tamfis_code.runner as runner
    import tamfis_code.runner_local as runner_local
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    remote_impl = AsyncMock(return_value=Outcome(status="completed"))
    monkeypatch.setattr(runner, "_run_ai_task_and_stream_impl", remote_impl)
    local_impl = AsyncMock(return_value=Outcome(status="completed", summary="done locally"))
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    runtime = UnifiedAgentRuntime()
    result = await runtime.execute_remote(**_remote_kwargs())

    remote_impl.assert_not_called()
    local_impl.assert_awaited_once()
    assert result.summary == "done locally"
    _, kwargs = local_impl.call_args
    assert kwargs["session_id"] == 50
    assert kwargs["messages"] == [{"role": "user", "content": "fix the bug"}]


@pytest.mark.asyncio
async def test_execute_remote_carries_prior_turns_as_conversation_history(monkeypatch, isolated_state):
    """A --remote session now has multi-turn memory the same way a
    standalone session does: _run_local_agent_turn_impl durably appends each
    completed turn via state.remember_conversation_turn, and execute_remote
    reads it back for the next call -- nothing server-side tracks it."""
    import tamfis_code.runner_local as runner_local
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    isolated_state.remember_conversation_turn(50, objective="what does this repo do", answer="It's a CLI.")
    local_impl = AsyncMock(return_value=Outcome(status="completed", summary="ok"))
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    runtime = UnifiedAgentRuntime()
    await runtime.execute_remote(**_remote_kwargs(objective="now add a test for it"))

    _, kwargs = local_impl.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what does this repo do"},
        {"role": "assistant", "content": "It's a CLI."},
        {"role": "user", "content": "now add a test for it"},
    ]


@pytest.mark.asyncio
async def test_execute_remote_read_only_mode_maps_to_local_read_only(monkeypatch, isolated_state):
    import tamfis_code.runner_local as runner_local
    from tamfis_code.runtime.unified import UnifiedAgentRuntime

    local_impl = AsyncMock(return_value=Outcome(status="completed"))
    monkeypatch.setattr(runner_local, "_run_local_agent_turn_impl", local_impl)

    runtime = UnifiedAgentRuntime()
    await runtime.execute_remote(**_remote_kwargs(mode="chat"))

    _, kwargs = local_impl.call_args
    assert kwargs["read_only"] is True
