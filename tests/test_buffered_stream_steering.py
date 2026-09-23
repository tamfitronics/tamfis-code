"""Steering must wake a buffered (`emit=False`) hung stream.

Regression for the 2026-09-23 live report: during a tool task the agent
buffers its streamed answer (`emit=False`), and the in-flight steering
watcher was gated on `emit` -- so a follow-up typed while the provider
request hung was silently ignored (no wake, no next request) until the
watchdog stall abort, or forever when the request outlived it. The watcher
must exist on every streamed request regardless of `emit`.
"""
import asyncio
import time
import unittest
from types import SimpleNamespace

from unittest.mock import patch

from tamfis_code import runner_local
from tamfis_code.providers import ProviderType
from tamfis_code.render import StreamRenderer


def _fake_renderer() -> StreamRenderer:
    renderer = StreamRenderer.__new__(StreamRenderer)
    renderer.background_requested = asyncio.Event()
    renderer.steering_requested = asyncio.Event()
    renderer._steering_revision = 0
    renderer._steering_handled_revision = 0
    renderer.events = []
    renderer.handle_event = renderer.events.append
    return renderer


class _HungStream:
    """A stream that emits one empty delta then never yields again."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise StopAsyncIteration  # pragma: no cover - never reached


class _FakeClient:
    """chat.completions.create returns a stream that hangs after creation."""

    def __init__(self):
        completions = SimpleNamespace(
            create=lambda **kwargs: _hang_forever_stream_coro(),
        )
        self.chat = SimpleNamespace(completions=completions)


async def _hang_forever_stream_coro():
    await asyncio.sleep(0)
    return _HungStream()


class BufferedStreamSteeringTest(unittest.TestCase):
    def test_hung_buffered_stream_wakes_on_steering(self):
        renderer = _fake_renderer()

        async def scenario():
            stream = _HungStream()

            async def hang():
                await asyncio.sleep(3600)

            real_anext = stream.__anext__

            async def anext():
                return await real_anext()

            stream.__anext__ = anext

            async def victim():
                return await runner_local._stream_one_completion_impl(
                    _FakeClient(), model="m", messages=[], tools=[],
                    renderer=renderer, emit=False,
                )

            task = asyncio.create_task(victim())
            await asyncio.sleep(0.1)
            self.assertFalse(task.done())
            # The user submits a live follow-up: revision bump + event set.
            renderer._steering_revision += 1
            renderer.steering_requested.set()
            start = time.monotonic()
            content, calls, finish_reason = await asyncio.wait_for(task, timeout=5)
            elapsed = time.monotonic() - start
            return finish_reason, elapsed
        finish_reason, elapsed = asyncio.run(scenario())
        self.assertEqual(finish_reason, "live_steering")
        self.assertLess(elapsed, 5.0)

    def test_unbuffered_stream_wakes_on_steering(self):
        renderer = _fake_renderer()

        async def scenario():
            task = asyncio.create_task(runner_local._stream_one_completion_impl(
                _FakeClient(), model="m", messages=[], tools=[],
                renderer=renderer, emit=True,
            ))
            await asyncio.sleep(0.1)
            self.assertFalse(task.done())
            renderer._steering_revision += 1
            renderer.steering_requested.set()
            content, calls, finish_reason = await asyncio.wait_for(task, timeout=5)
            return finish_reason

        self.assertEqual(asyncio.run(scenario()), "live_steering")

    def test_no_steering_keeps_stream_watching(self):
        """Without steering the hung stream must still be interruptible, not
        silently completed -- verify the watcher task exists alongside the
        chunk task while waiting."""
        renderer = _fake_renderer()

        async def scenario():
            task = asyncio.create_task(runner_local._stream_one_completion_impl(
                _FakeClient(), model="m", messages=[], tools=[],
                renderer=renderer, emit=False,
            ))
            await asyncio.sleep(0.1)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
