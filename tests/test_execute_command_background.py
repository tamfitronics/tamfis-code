"""execute_command's Ctrl+B backgrounding (mcp.py's background_signal param
and the module-level _BACKGROUND_JOBS registry / read_background_job_status).

The critical correctness property this guards: a detached command is the
SAME asyncio.subprocess.Process and the SAME in-flight communicate() call
that _execute_command already started -- not a restarted duplicate, and
never awaited twice (a second concurrent proc.communicate() on top of the
first would race it for the same pipes).
"""
from __future__ import annotations

import asyncio
import tempfile

import pytest

from tamfis_code.mcp import MCPServer, read_background_job_status


class TestExecuteCommandBackground:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.server = MCPServer(workspace_root=self.temp_dir, session_id=1)

    @pytest.mark.asyncio
    async def test_no_signal_behaves_exactly_as_before(self):
        result = await self.server.call_tool("execute_command", {"command": "echo hi"})
        assert result["success"] is True
        assert result["result"]["stdout"].strip() == "hi"
        assert result["result"]["success"] is True
        assert "backgrounded" not in result["result"]

    @pytest.mark.asyncio
    async def test_signal_already_set_detaches_immediately(self):
        signal = asyncio.Event()
        signal.set()

        result = await self.server.call_tool(
            "execute_command",
            {"command": "sleep 0.3 && echo done"},
            extra_kwargs={"background_signal": signal},
        )

        assert result["success"] is True
        payload = result["result"]
        assert payload["success"] is True
        assert payload["backgrounded"] is True
        job_id = payload["job_id"]
        assert job_id

        status = read_background_job_status(job_id)
        assert status["status"] == "running"

        # The same detached process keeps running and is eventually reaped
        # by _watch_background_job, not lost or duplicated. Polled with a
        # bounded budget rather than one fixed sleep -- `bash -lc`'s login
        # shell startup overhead alone measured ~0.6s in this environment,
        # well past the command's own 0.3s sleep.
        for _ in range(30):
            status = read_background_job_status(job_id)
            if status["status"] != "running":
                break
            await asyncio.sleep(0.2)
        assert status["status"] == "finished"
        assert status["return_code"] == 0
        assert "done" in status["stdout"]

    @pytest.mark.asyncio
    async def test_signal_never_set_completes_normally_not_backgrounded(self):
        signal = asyncio.Event()

        result = await self.server.call_tool(
            "execute_command", {"command": "echo hi"}, extra_kwargs={"background_signal": signal},
        )

        assert result["success"] is True
        assert result["result"]["stdout"].strip() == "hi"
        assert "backgrounded" not in result["result"]

    @pytest.mark.asyncio
    async def test_signal_set_after_command_already_finished_is_a_no_op(self):
        # The event fires, but only after communicate() has already won the
        # race -- the ordinary result must win, not a spurious background.
        signal = asyncio.Event()

        result = await self.server.call_tool(
            "execute_command", {"command": "echo hi"}, extra_kwargs={"background_signal": signal},
        )
        signal.set()

        assert result["result"]["stdout"].strip() == "hi"
        assert "backgrounded" not in result["result"]

    @pytest.mark.asyncio
    async def test_still_respects_timeout_when_never_backgrounded(self):
        signal = asyncio.Event()

        result = await self.server.call_tool(
            "execute_command",
            {"command": "sleep 2", "timeout": 1},
            extra_kwargs={"background_signal": signal},
        )

        assert result["result"]["success"] is False
        assert "timed out" in result["result"]["error"]

    def test_unknown_job_id_reports_not_found(self):
        status = read_background_job_status("does-not-exist-at-all")
        assert status["success"] is False
        assert "No background job" in status["error"]


class TestReadBackgroundJobToolWiring:
    """runner_local.py's read_background_job dispatch and the
    READ_BACKGROUND_JOB_TOOL_SCHEMA offer -- the model-facing half of this
    feature, distinct from mcp.py's own execute_command/background_signal
    mechanics covered above."""

    def test_schema_is_offered_whenever_any_other_tool_is(self):
        from tamfis_code.runner_local import READ_BACKGROUND_JOB_TOOL_SCHEMA

        assert READ_BACKGROUND_JOB_TOOL_SCHEMA["function"]["name"] == "read_background_job"
        assert "job_id" in READ_BACKGROUND_JOB_TOOL_SCHEMA["function"]["parameters"]["required"]

    def test_end_to_end_background_then_read_back(self):
        """A full turn: execute_command backgrounds (background_requested is
        pre-set, simulating a Ctrl+B press that landed before this round
        started), then a second round calls read_background_job with the
        returned job_id and gets the real, finished result back -- proving
        the wiring from live_input.py's keybinding through render.py's
        Event through runner_local.py's dispatch to mcp.py's registry all
        actually connects, not just each piece in isolation."""
        import asyncio
        import json
        import tempfile

        from tamfis_code.providers import ProviderType
        from tamfis_code.runner_local import run_local_agent_turn

        from test_reasoning_plan import (
            _FakeClient, _FakeManager, _RecordingRenderer, _StatePatchMixin,
            _chunk, _delta, _tool_call_delta,
        )

        class _BackgroundCapableRenderer(_RecordingRenderer):
            def __init__(self):
                super().__init__()
                self.background_requested = asyncio.Event()

        class _Harness(_StatePatchMixin):
            def runTest(self):
                pass

        harness = _Harness()
        harness.setUp()
        try:
            with tempfile.TemporaryDirectory() as ws:
                renderer = _BackgroundCapableRenderer()

                exec_args = json.dumps({"command": "sleep 0.6 && echo from-background"})
                rounds = [
                    [_chunk(_delta(tool_calls=[
                        _tool_call_delta(0, call_id="call_exec", name="execute_command", arguments=exec_args)
                    ]))],
                ]
                client = _FakeClient(rounds)
                manager = _FakeManager(client)

                from tamfis_code.mcp import read_background_job_status

                async def _press_ctrl_b_shortly_after_the_command_starts():
                    # runner_local.py clears background_requested right
                    # before dispatching each execute_command call (so a
                    # stale press from before this command started is never
                    # honored) -- this must fire AFTER that clear(), while
                    # the "sleep 0.6" command above is still genuinely
                    # in flight, to actually exercise the mid-command path.
                    await asyncio.sleep(0.15)
                    renderer.background_requested.set()

                async def _drive():
                    from io import StringIO
                    from rich.console import Console
                    console = Console(file=StringIO(), no_color=True, width=200)
                    asyncio.ensure_future(_press_ctrl_b_shortly_after_the_command_starts())
                    # max_rounds=1 with no final-answer round: the turn ends
                    # via the round-budget path, which is fine -- what's
                    # under test is the tool_output for the execute_command
                    # call itself.
                    await run_local_agent_turn(
                        manager, ProviderType.NVIDIA, None,
                        [{"role": "user", "content": "run something in the background"}],
                        console, renderer,
                        workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                        max_rounds=1,
                    )
                    tool_outputs = [
                        e["payload"]["result"] for e in renderer.events
                        if e["event_type"] == "tool_output" and e["payload"].get("tool") == "execute_command"
                    ]
                    assert tool_outputs, "expected an execute_command tool_output event"
                    backgrounded = tool_outputs[0]
                    assert backgrounded.get("backgrounded") is True
                    job_id = backgrounded["job_id"]

                    # Polled inside THIS same event loop -- the watcher task
                    # _watch_background_job scheduled itself on (see mcp.py)
                    # would otherwise be abandoned the moment this loop
                    # closes, exactly the way it stays alive for real across
                    # an interactive session's one long-lived asyncio.run().
                    status = None
                    for _ in range(30):
                        status = read_background_job_status(job_id)
                        if status["status"] != "running":
                            break
                        await asyncio.sleep(0.2)
                    return status

                status = asyncio.run(_drive())
                assert status["status"] == "finished"
                assert "from-background" in status["stdout"]
        finally:
            harness.tearDown()
