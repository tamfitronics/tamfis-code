"""runner_local.py's stuck-loop recovery path.

Live-reported (tamfis-code repo, kimi-k2.7-code via ollama_cloud): a model
ran the same execute_command wp-cli cleanup several times in a row across
four real databases (each call actually succeeded), got flagged as stuck,
got one nudge, stayed stuck, and the tools-disabled recovery completion
also came back with empty content -- so the whole turn hard-failed with
"nothing further to try", discarding four real, successful actions. The
fix: when the tools-disabled recovery answer is also empty, reconstruct a
plain-text summary directly from the tool calls/results already recorded
this turn instead of failing outright.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import run_local_agent_turn

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _FakeStream,
    _FallbackCapableManager,
    _RecordingRenderer,
    _StatePatchMixin,
    _chunk,
    _delta,
    _tool_call_delta,
)


class _RoundsThenRateLimitedClient:
    """Serves the given rounds normally, then raises a retryable rate-limit
    error on every call after -- simulating the stuck-loop recovery's own
    completion call landing on the same exhausted free-tier route that
    already served every earlier round in the turn."""

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        if self._rounds:
            return _FakeStream(self._rounds.pop(0))
        raise RuntimeError(
            "Error code: 429 - {'error': {'message': 'Rate limit exceeded: "
            "free-models-per-day. Add 10 credits to unlock 1000 free model "
            "requests per day', 'code': 429}}"
        )


class StuckLoopRecoveryTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def _read_round(self, index, path):
        args = json.dumps({"path": str(path)})
        return [_chunk(_delta(tool_calls=[
            _tool_call_delta(0, call_id=f"call_{index}", name="read_file", arguments=args)
        ]))]

    def test_empty_recovery_answer_falls_back_to_a_reconstructed_summary(self):
        with tempfile.TemporaryDirectory() as ws:
            path = Path(ws) / "file_0.py"
            path.write_text("# real content\n")

            # 5 identical read_file rounds: rounds 0-1 execute for real,
            # round 2 trips the stuck guard and is refused (nudge given),
            # round 3 executes for real again, round 4 trips the guard a
            # second time -- nudge budget (1) is exhausted, so this goes
            # straight to the tools-disabled recovery completion, which
            # returns an empty chunk to simulate the reported failure.
            rounds = [self._read_round(i, path) for i in range(5)]
            rounds.append([_chunk(_delta(content=""))])
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "read file_0.py repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("read_file", outcome.summary)
            self.assertIn("done", outcome.summary)
            # The reconstructed "name(args) -> done" lines are themselves
            # paren-style text that _looks_like_fake_tool_call would flag if
            # re-checked -- this summary reports real, already-executed
            # calls and must not be second-guessed with that caveat.
            self.assertNotIn("unexecuted tool call", outcome.summary)
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(any("reconstructing a summary" in d for d in diagnostics))

    def test_fake_tool_call_recovery_answer_falls_back_to_a_reconstructed_summary(self):
        """Live-reported (tamfis-code, provider-fallback route "TamfisGPT
        Ultima"): once tools are disabled for the recovery completion, a
        weak model can still write out a well-formed <tool_call> block as
        plain text instead of the requested prose answer. Nothing executes
        it -- tools really are off -- so the turn used to finalize as
        `completed` with the unexecuted tool call plus a fake-tool-call
        caveat as its entire visible output, even though the user's actual
        request (list a directory) was never fulfilled. It must be treated
        like an empty recovery answer: reconstruct a summary from the real
        tool evidence already gathered this turn instead of accepting the
        garbage text as a real answer."""
        with tempfile.TemporaryDirectory() as ws:
            path = Path(ws) / "file_0.py"
            path.write_text("# real content\n")

            rounds = [self._read_round(i, path) for i in range(5)]
            rounds.append([_chunk(_delta(
                content="<tool_call><function=list_directory><parameter=path>.</parameter></tool_call>"
            ))])
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "read file_0.py repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("read_file", outcome.summary)
            self.assertIn("done", outcome.summary)
            self.assertNotIn("<tool_call>", outcome.summary)
            # Same false-positive risk as the empty-answer case above: the
            # reconstructed "read_file(...) -> done" lines must not trip
            # _looks_like_fake_tool_call a second time on the way out.
            self.assertNotIn("unexecuted tool call", outcome.summary)
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(any("reconstructing a summary" in d for d in diagnostics))

    def test_stuck_loop_does_not_switch_providers_when_one_was_explicitly_pinned(self):
        """FIX: _handle_stuck_loop's own provider-fallback branch used to
        switch to a different provider on a detected stall regardless of
        whether the caller explicitly pinned one -- the only escalation
        path in runner_local.py that didn't check `provider ==
        ProviderType.AUTO` first (every other fallback site does). An
        explicit provider selection must be respected the same way here:
        a stall should exhaust the nudge budget and fall through to the
        tools-disabled recovery completion on the SAME pinned provider,
        never silently hop to a different one the user didn't choose."""
        with tempfile.TemporaryDirectory() as ws:
            path = Path(ws) / "file_0.py"
            path.write_text("# real content\n")

            rounds = [self._read_round(i, path) for i in range(5)]
            rounds.append([_chunk(_delta(content="Final answer from the pinned provider."))])
            pinned_client = _FakeClient(rounds)
            other_client = _FakeClient([[_chunk(_delta(content="Should never be called."))]])
            manager = _FallbackCapableManager(
                {ProviderType.NVIDIA: pinned_client, ProviderType.OPENROUTER: other_client},
                fallback_order=[ProviderType.OPENROUTER],
            )
            for config in manager.PROVIDERS.values():
                config.context_window = 32768
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "read file_0.py repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("Final answer from the pinned provider.", outcome.summary)
            self.assertEqual(len(other_client.calls), 0)
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(any("disabling tools for one final answer" in d for d in diagnostics))
            self.assertFalse(any("continuing on another compatible route" in d for d in diagnostics))

    def test_recovery_answer_falls_over_to_another_provider_on_rate_limit(self):
        """Live-reported: the tools-disabled recovery completion hit a 429
        on an exhausted free-tier OpenRouter route and hard-failed the whole
        turn ("nothing further to try this turn"), even though a different,
        healthy provider was configured. This call must retry across
        fallback_candidates the same way the main answer path and
        _attempt_reasoning_plan already do, instead of giving up on the
        first failure."""
        with tempfile.TemporaryDirectory() as ws:
            path = Path(ws) / "file_0.py"
            path.write_text("# real content\n")

            rounds = [self._read_round(i, path) for i in range(5)]
            failing_client = _RoundsThenRateLimitedClient(rounds)
            working_client = _FakeClient([[_chunk(_delta(content="Recovered answer after fallback."))]])
            manager = _FallbackCapableManager(
                {ProviderType.NVIDIA: failing_client, ProviderType.OPENROUTER: working_client},
                fallback_order=[ProviderType.OPENROUTER],
            )
            for config in manager.PROVIDERS.values():
                config.context_window = 32768
            # Isolate the fix under test (the tools-disabled recovery call's
            # own retry-with-fallback) from the separate, earlier
            # switch-provider-and-continue-the-round-loop mechanism, which
            # would otherwise also trigger on this same stuck detection and
            # reach the fallback provider before the code path this test
            # targets ever runs.
            manager.auto_fallback_enabled = lambda: False
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "read file_0.py repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("Recovered answer after fallback.", outcome.summary)
            self.assertEqual(len(working_client.calls), 1)
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(any("retrying with a different provider" in d for d in diagnostics))
            self.assertFalse(any("nothing further to try this turn" in d for d in diagnostics))


if __name__ == "__main__":
    unittest.main()
