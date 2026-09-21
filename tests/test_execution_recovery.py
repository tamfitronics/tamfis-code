"""Regression tests for the "dead composer / 27-minute wait" incident.

Live report: a task sat on "Waiting for the model's next step…", Enter did
nothing, and Esc / focus events were painted as ``^[^[^[[O^[[I``. Root causes
covered here:

* Ctrl+D / EOF (or any unexpected exit) silently ended the live prompt while the
  task kept running -> cooked tty, echoed keys, dead Enter.
* The stream idle timer reset on EMPTY chunks, so a keepalive-only provider
  never timed out.
* A parallel tool batch used bare gather(): no per-call bound, first exception
  abandoned its siblings.
* Whole-tree scans ran synchronously on the event loop and froze the keyboard.
"""
import asyncio
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser
from rich.console import Console

from tamfis_code import runner_local
from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.live_input import LiveInputListener, strip_terminal_noise
from tamfis_code.mcp import run_blocking_bounded
from tamfis_code.render import StreamRenderer
from tamfis_code.runtime.progress import (
    ExecState, ProgressTracker, StallPolicy, classify_provider_failure,
)
from tamfis_code.terminal_guard import FOCUS_REPORTING_OFF, TerminalGuard, disable_focus_reporting


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _tracker(warn=45.0, abort=240.0):
    clock = _Clock()
    return ProgressTracker(StallPolicy(warn, abort), clock=clock), clock


class ProgressStateMachineTests(unittest.TestCase):
    def test_waiting_provider_becomes_stalled_then_abortable(self):
        tracker, clock = _tracker(warn=45, abort=240)
        tracker.observe("task_started")
        tracker.observe("provider_request_started")
        self.assertEqual(tracker.state(), ExecState.WAITING_PROVIDER)
        clock.now += 46
        self.assertEqual(tracker.state(), ExecState.STALLED)
        self.assertFalse(tracker.should_abort_provider_wait())
        clock.now += 200
        self.assertTrue(tracker.should_abort_provider_wait())

    def test_tokens_are_progress_but_empty_deltas_and_diagnostics_are_not(self):
        tracker, clock = _tracker()
        tracker.observe("provider_request_started")
        clock.now += 30
        tracker.observe("assistant_delta", {"content": ""})
        tracker.observe("diagnostics", {"content": "spinner tick"})
        self.assertGreaterEqual(tracker.idle_seconds(), 30)
        tracker.observe("reasoning_delta", {"content": "hm"})
        self.assertEqual(tracker.idle_seconds(), 0)
        self.assertEqual(tracker.state(), ExecState.RUNNING)

    def test_tool_batch_counts_down_to_running(self):
        tracker, _ = _tracker()
        for _ in range(3):
            tracker.observe("tool_call_requested", {"name": "read_file"})
        self.assertEqual(tracker.state(), ExecState.WAITING_TOOL)
        for _ in range(3):
            tracker.observe("tool_output", {})
        self.assertEqual(tracker.pending_tools, 0)
        self.assertEqual(tracker.state(), ExecState.RUNNING)
        # A long tool never trips the provider abort.
        tracker.observe("tool_call_requested", {})
        tracker._clock.now += 10_000
        self.assertFalse(tracker.should_abort_provider_wait())

    def test_failure_classification_and_recovery(self):
        tracker, _ = _tracker()
        tracker.observe("provider_request_started")
        tracker.note_provider_failure(RuntimeError("HTTP 429 Too Many Requests"))
        self.assertEqual(tracker.state(), ExecState.RATE_LIMITED)
        tracker.note_provider_failure(RuntimeError("insufficient credit / quota exceeded"))
        self.assertEqual(tracker.state(), ExecState.QUOTA_EXHAUSTED)
        tracker.note_provider_failure(TimeoutError("idle"))
        self.assertEqual(tracker.state(), ExecState.RETRYING)
        tracker.observe("assistant_delta", {"content": "ok"})
        self.assertEqual(tracker.state(), ExecState.RUNNING)
        self.assertEqual(classify_provider_failure(SimpleNamespace(status_code=402)), ExecState.QUOTA_EXHAUSTED)

    def test_approval_cancel_and_terminal_states(self):
        tracker, _ = _tracker()
        tracker.observe("approval_required", {})
        self.assertEqual(tracker.state(), ExecState.WAITING_USER)
        tracker.observe("tool_output", {})
        self.assertNotEqual(tracker.state(), ExecState.WAITING_USER)
        tracker.request_cancel()
        self.assertEqual(tracker.state(), ExecState.CANCELLING)
        tracker.finish("completed")
        self.assertEqual(tracker.state(), ExecState.COMPLETED)
        tracker.reset()
        tracker.finish("failed")
        self.assertEqual(tracker.state(), ExecState.FAILED)


class _StaticStream:
    def __init__(self, chunks, then_hang=False):
        self._chunks, self._hang = list(chunks), then_hang

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            await asyncio.sleep(0.01)
            return self._chunks.pop(0)
        if self._hang:
            await asyncio.sleep(3600)
        raise StopAsyncIteration

    async def close(self):
        pass


def _chunk(content=None, tool=None):
    delta = SimpleNamespace(content=content, tool_calls=tool, role=None)
    return SimpleNamespace(
        id="c", model="m", usage=None,
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
    )


class _Client:
    def __init__(self, stream):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream)))


def _renderer():
    return StreamRenderer(Console(file=StringIO(), no_color=True, width=120))


class StreamProgressTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # route_stats persists to the real config dir and providers keeps process-wide
        # route health; a timeout test must not leave either behind for other tests.
        from tamfis_code import providers

        self._patches = [
            patch.object(runner_local.route_stats, "record_failure"),
            patch.object(runner_local.route_stats, "record_latency"),
        ]
        for item in self._patches:
            item.start()
        with providers._HEALTH_LOCK:
            self._health = dict(providers._ROUTE_HEALTH)

    def tearDown(self):
        from tamfis_code import providers

        for item in self._patches:
            item.stop()
        with providers._HEALTH_LOCK:
            providers._ROUTE_HEALTH.clear()
            providers._ROUTE_HEALTH.update(self._health)

    async def test_keepalive_only_stream_times_out(self):
        """Empty deltas used to reset the idle timer forever."""
        empties = [_chunk(content=None) for _ in range(500)]
        client = _Client(_StaticStream(empties))
        started = time.monotonic()
        with patch.object(runner_local, "STREAM_IDLE_TIMEOUT_SECONDS", 0.3):
            with self.assertRaises(asyncio.TimeoutError) as ctx:
                await asyncio.wait_for(runner_local._stream_one_completion_impl(
                    client, model="m", messages=[{"role": "user", "content": "x"}],
                    tools=[], renderer=_renderer(), emit=False,
                ), timeout=10)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn("idle", str(ctx.exception))

    async def test_hung_stream_after_first_event_times_out(self):
        client = _Client(_StaticStream([_chunk(content=None)], then_hang=True))
        with patch.object(runner_local, "STREAM_IDLE_TIMEOUT_SECONDS", 0.3):
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(runner_local._stream_one_completion_impl(
                    client, model="m", messages=[{"role": "user", "content": "x"}],
                    tools=[], renderer=_renderer(), emit=False,
                ), timeout=10)

    async def test_total_deadline_bounds_a_slow_drip_of_real_output(self):
        chunks = [_chunk(content="x") for _ in range(2000)]
        client = _Client(_StaticStream(chunks))
        with patch.object(runner_local, "STREAM_TOTAL_TIMEOUT_SECONDS", 0.3), \
                patch.object(runner_local, "STREAM_IDLE_TIMEOUT_SECONDS", 60.0):
            with self.assertRaises(asyncio.TimeoutError) as ctx:
                await asyncio.wait_for(runner_local._stream_one_completion_impl(
                    client, model="m", messages=[{"role": "user", "content": "x"}],
                    tools=[], renderer=_renderer(), emit=False,
                ), timeout=10)
        self.assertIn("total", str(ctx.exception))

    async def test_provider_failure_is_classified_on_the_renderer(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(side_effect=RuntimeError("HTTP 429 rate limit")))))
        renderer = _renderer()
        with self.assertRaises(RuntimeError):
            await runner_local._stream_one_completion(
                client, model="m", messages=[{"role": "user", "content": "x"}],
                tools=[], renderer=renderer, emit=False,
            )
        self.assertEqual(renderer.progress.state(), ExecState.RATE_LIMITED)


class ToolBarrierTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_failure_and_timeout_all_resolve_the_barrier(self):
        async def call_tool(name, args, extra_kwargs=None):
            if name == "read_file":
                return {"success": True, "result": "ok", "tool": name}
            if name == "list_directory":
                raise RuntimeError("boom")
            await asyncio.sleep(3600)  # search_code hangs

        server = SimpleNamespace(call_tool=call_tool)
        started = time.monotonic()
        results = await asyncio.wait_for(asyncio.gather(*(
            runner_local._bounded_tool_call(server, n, {}, timeout=0.3)
            for n in ("read_file", "list_directory", "search_code")
        )), timeout=10)
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(results[0]["success"])
        self.assertFalse(results[1]["success"])
        self.assertIn("boom", results[1]["error"])
        self.assertTrue(results[2]["timed_out"])

    async def test_exempt_tools_are_not_time_limited(self):
        async def call_tool(name, args, extra_kwargs=None):
            await asyncio.sleep(0.2)
            return {"success": True}

        result = await runner_local._bounded_tool_call(
            SimpleNamespace(call_tool=call_tool), "execute_command", {}, timeout=0.01,
        )
        self.assertTrue(result["success"])

    async def test_cancellation_still_propagates(self):
        async def call_tool(name, args, extra_kwargs=None):
            await asyncio.sleep(3600)

        task = asyncio.create_task(runner_local._bounded_tool_call(
            SimpleNamespace(call_tool=call_tool), "read_file", {}, timeout=60,
        ))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


class EventLoopStaysResponsiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_work_does_not_freeze_the_loop(self):
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        result = await run_blocking_bounded(lambda: (time.sleep(0.5), "done")[1], timeout=5)
        task.cancel()
        self.assertEqual(result, "done")
        self.assertGreater(ticks, 10)  # the loop kept running during the 0.5s block

    async def test_blocking_work_is_bounded(self):
        with self.assertRaises(asyncio.TimeoutError):
            await run_blocking_bounded(lambda: time.sleep(2), timeout=0.1)

    async def test_blocking_exceptions_propagate(self):
        def boom():
            raise ValueError("bad")

        with self.assertRaises(ValueError):
            await run_blocking_bounded(boom, timeout=5)


class KeyDecodingTests(unittest.TestCase):
    def _feed(self, data):
        keys = []
        parser = Vt100Parser(lambda kp: keys.append(kp))
        parser.feed(data)
        parser.flush()
        return keys

    def test_known_sequences_decode_to_single_keys(self):
        for seq, name in (("\x1b[A", "up"), ("\x1b[B", "down"), ("\x1b[Z", "s-tab"), ("\x1bOA", "up")):
            keys = self._feed(seq)
            self.assertEqual([k.key.value if hasattr(k.key, "value") else k.key for k in keys], [name], seq)

    def test_bracketed_paste_is_one_event(self):
        keys = self._feed("\x1b[200~line1\nline2\x1b[201~")
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0].data, "line1\nline2")

    def test_focus_sequences_are_bound_so_they_are_consumed(self):
        # ptk decodes ESC [ I / ESC [ O as Escape, '[', 'I'/'O'. The live composer
        # binds both full sequences so the bare-Esc "cancel" binding never fires
        # and the letters are never inserted (see LiveInputListener._input_loop).
        keys = [k.data for k in self._feed("\x1b[O\x1b[I")]
        self.assertEqual(keys, ["\x1b", "[", "O", "\x1b", "[", "I"])
        self.assertNotIn("\x1b[I", ANSI_SEQUENCES)

    def test_strip_terminal_noise(self):
        self.assertEqual(strip_terminal_noise("hi\x1b[O\x1b[I there"), "hi there")
        self.assertEqual(strip_terminal_noise("\x1b\x1b\x1bhello"), "hello")
        self.assertEqual(strip_terminal_noise("a\x1b[Zb\x1b[A"), "ab")
        self.assertEqual(strip_terminal_noise("keep [O and\ttabs\nnewlines"), "keep [O and\ttabs\nnewlines")


class TerminalGuardTests(unittest.TestCase):
    def test_disable_focus_reporting_writes_mode_off_only_to_a_tty(self):
        tty = SimpleNamespace(isatty=lambda: True, written=[], flush=lambda: None)
        tty.write = tty.written.append
        disable_focus_reporting(tty)
        self.assertEqual(tty.written, [FOCUS_REPORTING_OFF])
        pipe = SimpleNamespace(isatty=lambda: False, write=lambda s: self.fail("wrote to a pipe"), flush=lambda: None)
        disable_focus_reporting(pipe)

    def test_guard_mutes_and_restores_echo_on_a_real_pty(self):
        import os
        import pty
        import termios

        master, slave = pty.openpty()
        try:
            guard = TerminalGuard(fd=slave)
            guard.snapshot()
            self.assertTrue(termios.tcgetattr(slave)[3] & termios.ECHO)
            self.assertTrue(guard.mute_echo())
            self.assertFalse(termios.tcgetattr(slave)[3] & termios.ECHO)
            self.assertTrue(termios.tcgetattr(slave)[3] & termios.ISIG)  # Ctrl+C still works
            guard.restore()
            self.assertTrue(termios.tcgetattr(slave)[3] & termios.ECHO)
        finally:
            os.close(master)
            os.close(slave)

    def test_guard_is_a_noop_without_a_tty(self):
        guard = TerminalGuard(fd=None)
        with patch("sys.stdin", SimpleNamespace(fileno=lambda: (_ for _ in ()).throw(OSError()))):
            self.assertFalse(guard.mute_echo())
            guard.restore()


class _StatePatch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._orig
        self.tmp.cleanup()


def _listener(session_id=7):
    cfg = Config.__new__(Config)
    cfg.approval_policy = "ask"
    renderer = _renderer()
    return LiveInputListener(session_id=session_id, renderer=renderer, cli_config=cfg), renderer


class ListenerRecoveryTests(_StatePatch):
    async def test_supervisor_restarts_a_composer_that_died_unexpectedly(self):
        listener, _ = _listener()
        listener._active = True
        runs = []

        async def flaky():
            runs.append(1)
            if len(runs) < 3:
                return  # e.g. old `except EOFError: return`
            listener._input_exit_expected = True

        listener._input_loop = flaky
        await asyncio.wait_for(listener._supervised_input_loop(), timeout=5)
        self.assertEqual(len(runs), 3)

    async def test_supervisor_survives_exceptions_and_gives_up_loudly(self):
        listener, renderer = _listener()
        listener._active = True

        async def broken():
            raise RuntimeError("renderer exploded")

        listener._input_loop = broken
        with patch.object(listener._terminal, "mute_echo", return_value=True) as mute:
            await asyncio.wait_for(listener._supervised_input_loop(), timeout=10)
        mute.assert_called_once()
        self.assertIn("message box stopped working", renderer.console.file.getvalue())

    async def test_expected_exit_is_not_restarted(self):
        listener, _ = _listener()
        listener._active = True
        runs = []

        async def once():
            runs.append(1)
            listener._input_exit_expected = True

        listener._input_loop = once
        await asyncio.wait_for(listener._supervised_input_loop(), timeout=5)
        self.assertEqual(runs, [1])

    async def test_single_eof_keeps_the_prompt_but_hangup_stops_the_task(self):
        listener, _ = _listener()
        stopped = []
        listener._interrupt_callback = stopped.append
        with patch.object(listener, "_stdin_closed", return_value=False):
            self.assertFalse(listener._handle_input_eof())
            self.assertFalse(listener._handle_input_eof())
            self.assertTrue(listener._handle_input_eof())  # 3 in 3s = the tty is gone
        self.assertEqual(stopped, ["exit"])

        listener2, _ = _listener(8)
        listener2._interrupt_callback = stopped.append
        with patch.object(listener2, "_stdin_closed", return_value=True):
            self.assertTrue(listener2._handle_input_eof())

    async def test_stall_classification_is_not_persisted_as_an_instruction(self):
        listener, _ = _listener(9)
        seen = []
        listener._interrupt_callback = seen.append
        listener._request_interrupt("stall")
        self.assertEqual(seen, ["stall"])
        self.assertEqual(state_module.get_session_state(9).queued_user_instructions, [])

    async def test_watchdog_aborts_only_a_dead_provider_wait(self):
        listener, renderer = _listener(10)
        clock = _Clock()
        renderer.progress = ProgressTracker(StallPolicy(5, 20), clock=clock)
        seen = []
        listener._interrupt_callback = seen.append
        renderer.progress.observe("provider_request_started")
        clock.now += 6
        listener._watchdog_check()
        self.assertEqual(seen, [])
        self.assertIn("has not responded", renderer.console.file.getvalue())
        clock.now += 30
        listener._watchdog_check()
        self.assertEqual(seen, ["stall"])
        self.assertIn("type `continue`", " ".join(renderer.console.file.getvalue().split()))

    async def test_watchdog_leaves_a_running_tool_alone(self):
        listener, renderer = _listener(11)
        clock = _Clock()
        renderer.progress = ProgressTracker(StallPolicy(5, 20), clock=clock)
        seen = []
        listener._interrupt_callback = seen.append
        renderer.progress.observe("tool_call_requested", {"name": "execute_command"})
        clock.now += 3600
        listener._watchdog_check()
        self.assertEqual(seen, [])


class FollowUpTests(_StatePatch):
    async def test_enter_acknowledges_persists_and_scopes_to_the_session(self):
        listener, renderer = _listener(21)
        other, _ = _listener(22)
        listener._enqueue("Also inspect the gateway.")
        out = renderer.console.file.getvalue()
        self.assertIn("Follow-up queued", out)
        self.assertIn("Also inspect the gateway.", out)
        queued = state_module.get_session_state(21).queued_user_instructions
        self.assertEqual([i["text"] for i in queued], ["Also inspect the gateway."])
        self.assertEqual(queued[0]["classification"], "follow_up")
        self.assertEqual(state_module.get_session_state(22).queued_user_instructions, [])
        self.assertTrue(renderer.has_pending_steering())  # wakes the active stream
        del other

    async def test_control_sequences_never_reach_the_queue(self):
        listener, _ = _listener(23)
        listener._enqueue("\x1b\x1b[Ohello\x1b[I")
        queued = state_module.get_session_state(23).queued_user_instructions
        self.assertEqual(queued[0]["text"], "hello")

    async def test_status_answers_locally_and_is_not_queued(self):
        listener, renderer = _listener(24)
        renderer.progress.observe("provider_request_started")
        listener._enqueue("/status")
        out = renderer.console.file.getvalue()
        for field in ("Current step:", "State:", "Provider:", "Last progress:", "Pending tools:", "Queued follow-ups:"):
            self.assertIn(field, out)
        self.assertEqual(state_module.get_session_state(24).queued_user_instructions, [])
        self.assertNotIn("api_key", out.lower())

    async def test_queue_indicator_and_headline_state(self):
        listener, renderer = _listener(25)
        renderer.progress.observe("provider_request_started")
        renderer.progress._clock = lambda: time.monotonic() + 100  # 100s of silence
        headline = renderer.live_input_headline("✽")
        self.assertIn("Provider not responding", headline)
        renderer.request_steering()
        self.assertEqual(renderer.steering_revision() - renderer._steering_handled_revision, 1)


if __name__ == "__main__":
    unittest.main()
