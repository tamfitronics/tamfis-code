import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.config import Config, next_mode_in_cycle
from tamfis_code.live_input import LiveInputListener, _CTRL_T, _SHIFT_TAB
from tamfis_code.render import StreamRenderer


class _StatePatchMixin:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200)


def _config(approval_policy: str = "ask") -> Config:
    cfg = Config.__new__(Config)
    cfg.approval_policy = approval_policy
    return cfg


class ShiftTabCyclesModeTests(unittest.TestCase):
    def test_dispatch_cycles_approval_policy_and_emits_diagnostic(self):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
        expected = next_mode_in_cycle("ask")

        listener._buf = bytearray(_SHIFT_TAB)
        listener._dispatch()

        self.assertEqual(cfg.approval_policy, expected)
        self.assertEqual(bytes(listener._buf), b"")

    def test_incomplete_escape_sequence_is_not_dropped_prematurely(self):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)

        listener._buf = bytearray(b"\x1b")
        listener._dispatch()
        self.assertEqual(bytes(listener._buf), b"\x1b")  # still waiting

        listener._buf.extend(b"[")
        listener._dispatch()
        self.assertEqual(bytes(listener._buf), b"\x1b[")  # still waiting

        listener._buf.extend(b"Z")
        listener._dispatch()
        self.assertEqual(bytes(listener._buf), b"")  # consumed as Shift+Tab
        self.assertEqual(cfg.approval_policy, next_mode_in_cycle("ask"))

    def test_unrecognised_byte_is_dropped_silently(self):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)

        listener._buf = bytearray(b"x")
        listener._dispatch()

        self.assertEqual(bytes(listener._buf), b"")
        self.assertEqual(cfg.approval_policy, "ask")  # unchanged


class CtrlTInjectsFollowUpTests(_StatePatchMixin, unittest.IsolatedAsyncioTestCase):
    @patch("prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, return_value="also check the login page")
    async def test_interject_enqueues_a_follow_up_instruction(self, _mock_prompt_async):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=42, renderer=renderer, cli_config=cfg)

        await listener._interject()

        queued = state_module.get_session_state(42).queued_user_instructions
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["text"], "also check the login page")
        self.assertEqual(queued[0]["classification"], "follow_up")
        rendered = renderer.console.file.getvalue()
        self.assertIn("Queued next instruction", rendered)
        self.assertIn("also check the login page", rendered)

    @patch("prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, return_value="   ")
    async def test_blank_interject_queues_nothing(self, _mock_prompt_async):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=43, renderer=renderer, cli_config=cfg)

        await listener._interject()

        queued = state_module.get_session_state(43).queued_user_instructions
        self.assertEqual(queued, [])

    async def test_interject_wraps_the_prompt_in_patch_stdout(self):
        # Locks in the actual fix for the "other output can corrupt the
        # in-progress typed line" rough edge: the prompt must run inside
        # prompt_toolkit's patch_stdout(), which is what safely coalesces
        # any concurrent console output above the active input line instead
        # of interleaving with it.
        import sys as _sys

        from prompt_toolkit.patch_stdout import StdoutProxy

        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=44, renderer=renderer, cli_config=cfg)
        seen_patched = False

        async def _check_patched(*_args, **_kwargs):
            nonlocal seen_patched
            seen_patched = isinstance(_sys.stdout, StdoutProxy)
            return ""

        with patch("prompt_toolkit.PromptSession.prompt_async", side_effect=_check_patched):
            await listener._interject()
        self.assertTrue(seen_patched, "prompt_async did not run under patch_stdout()")
        self.assertNotIsInstance(_sys.stdout, StdoutProxy)  # restored afterward

    def test_ctrl_t_byte_schedules_an_interject_task(self):
        # _dispatch() itself must not block -- it only needs to *schedule*
        # the interject coroutine (asyncio.ensure_future), not run it
        # inline, since running it inline would call the blocking `input()`
        # helper synchronously from inside the fd-readable callback.
        async def _noop():
            return None

        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            with patch.object(listener, "_interject", side_effect=_noop) as mocked:
                listener._buf = bytearray(_CTRL_T)
                listener._dispatch()
                self.assertEqual(bytes(listener._buf), b"")
                await asyncio.sleep(0)  # let the scheduled task actually run
                mocked.assert_called_once()

    def test_repeated_ctrl_t_does_not_open_competing_editors(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            gate = asyncio.Event()

            async def _blocked():
                await gate.wait()

            with patch.object(listener, "_interject", side_effect=_blocked) as mocked:
                listener._buf = bytearray(_CTRL_T)
                listener._dispatch()
                await asyncio.sleep(0)
                listener._buf = bytearray(_CTRL_T)
                listener._dispatch()
                await asyncio.sleep(0)
                self.assertEqual(mocked.call_count, 1)
                gate.set()
                await asyncio.sleep(0)

        import asyncio
        asyncio.run(_run())

    def test_ctrl_t_is_detected_when_read_with_adjacent_terminal_bytes(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            with patch.object(listener, "_interject", new_callable=AsyncMock) as mocked:
                listener._buf = bytearray(b"x" + _CTRL_T + b"y")
                listener._dispatch()
                await asyncio.sleep(0)
                mocked.assert_awaited_once()

        import asyncio
        asyncio.run(_run())

        import asyncio
        asyncio.run(_run())


class PauseResumeAreSafeOffTtyTests(unittest.TestCase):
    def test_pause_resume_start_stop_are_no_ops_without_a_real_tty(self):
        # Test processes' stdin is never a TTY -- this locks in that every
        # method degrades to a safe no-op rather than raising, matching
        # every other TTY-gated feature in render.py/interactive.py.
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
        self.assertFalse(listener._is_tty)

        listener.start()
        listener.pause()
        listener.resume()
        listener.stop()  # must not raise


class RendererSuspendResumeTouchesListenerTests(unittest.TestCase):
    def test_suspend_and_resume_are_safe_when_no_listener_attached(self):
        renderer = StreamRenderer(_console())
        renderer.suspend_live()
        renderer.resume_live()  # must not raise; live_input_listener is None

    def test_suspend_and_resume_delegate_to_an_attached_listener(self):
        renderer = StreamRenderer(_console())
        calls = []
        renderer.live_input_listener = SimpleNamespaceListener(calls)
        renderer.suspend_live()
        renderer.resume_live()
        self.assertEqual(calls, ["pause", "resume"])


class SimpleNamespaceListener:
    def __init__(self, calls):
        self._calls = calls

    def pause(self):
        self._calls.append("pause")

    def resume(self):
        self._calls.append("resume")


if __name__ == "__main__":
    unittest.main()
