import asyncio
import unittest
from unittest.mock import patch

from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import _stream_completion_with_reconnect


class _FakeManager:
    """Always says the failure is a transient, same-route-retryable one."""

    def is_retryable_provider_error(self, exc: Exception) -> bool:
        return True

    def provider_error_status(self, exc: Exception):
        return 503


class _EventCollectingRenderer:
    def __init__(self, *, debug: bool):
        self.debug = debug
        self.events: list[dict] = []

    def handle_event(self, event: dict) -> None:
        self.events.append(event)


class StreamReconnectDiagnosticsTests(unittest.TestCase):
    """A dropped stream is retried and stitched back together silently --
    the "Stream interrupted; reconnecting" line used to print on every
    single retry regardless of outcome, which announced normal, working
    continuation as if it were a repeating failure. It should only ever be
    visible under debug."""

    def _run_one_retry_then_succeed(self, *, debug: bool) -> _EventCollectingRenderer:
        renderer = _EventCollectingRenderer(debug=debug)
        attempts = {"count": 0}

        async def fake_stream_one_completion(client, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionError("dropped mid-stream")
            return "final answer", [], "stop"

        async def no_sleep(_seconds):
            return None

        async def run():
            with patch(
                "tamfis_code.runner_local._stream_one_completion",
                side_effect=fake_stream_one_completion,
            ), patch("asyncio.sleep", side_effect=no_sleep):
                return await _stream_completion_with_reconnect(
                    _FakeManager(), client=object(),
                    provider=ProviderType.OLLAMA_CLOUD, model="test-model",
                    messages=[{"role": "user", "content": "hi"}], tools=[],
                    renderer=renderer,
                )

        content, calls, finish_reason = asyncio.run(run())
        self.assertEqual(content, "final answer")
        self.assertEqual(attempts["count"], 2)
        return renderer

    def test_retry_is_silent_by_default(self):
        renderer = self._run_one_retry_then_succeed(debug=False)
        reconnect_events = [
            event for event in renderer.events
            if "interrupted" in str(event.get("payload", {}).get("content", "")).lower()
        ]
        self.assertEqual(reconnect_events, [])

    def test_retry_is_visible_under_debug(self):
        renderer = self._run_one_retry_then_succeed(debug=True)
        reconnect_events = [
            event for event in renderer.events
            if "interrupted" in str(event.get("payload", {}).get("content", "")).lower()
        ]
        self.assertEqual(len(reconnect_events), 1)
        self.assertIn("reconnecting", reconnect_events[0]["payload"]["content"])


if __name__ == "__main__":
    unittest.main()
