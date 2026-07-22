"""Context-budget defenses beyond in-place compaction: internal context
rollover (durable evidence outside the provider prompt + a fresh minimal
continuation for the SAME task), evidence retrieval, larger-context
provider fallback, and repeating-cycle loop detection.

Regression coverage for spec test scenarios: a 300,000-char tool argument
(#1), a tool result of hundreds of thousands of characters (#2), many
completed tool cycles (#4), fallback to a larger-context provider (#5),
internal context rollover (#6), retrieval of evidence from a previous
segment (#7), and repeated-tool-loop detection (#14).
"""
from __future__ import annotations

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from tamfis_code import evidence as evidence_module
from tamfis_code import state as state_module
from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import (
    _compact_tool_arguments,
    _perform_context_rollover,
    _trim_tool_outputs,
    run_local_agent_turn,
)


class _StatePatchMixin:
    def setUp(self):
        self._state_originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._evidence_original = evidence_module.EVIDENCE_DIR
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        evidence_module.EVIDENCE_DIR = base / ".config" / "evidence"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._state_originals
        evidence_module.EVIDENCE_DIR = self._evidence_original
        self.tmp.cleanup()


class CompactionKeepsValidJsonTests(_StatePatchMixin, unittest.TestCase):
    """These compaction primitives already existed; this locks in the two
    largest raw-input scenarios explicitly called out in the spec."""

    def test_300k_char_tool_call_argument_stays_valid_json_after_compaction(self):
        huge_content = "x" * 300_000
        arguments = json.dumps({"path": "src/example.py", "content": huge_content})

        compacted = _compact_tool_arguments(arguments, head=200, tail=100)

        parsed = json.loads(compacted)  # must not raise -- this is the core guarantee
        self.assertEqual(parsed["path"], "src/example.py")
        self.assertLess(len(parsed["content"]), len(huge_content))
        self.assertIn("_tamfis_compacted", json.dumps(parsed))
        self.assertLess(len(compacted), len(arguments))

    def test_300k_char_tool_result_gets_compacted_below_budget(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read the huge file"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "big.txt"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y" * 300_000},
        ]
        changed = _trim_tool_outputs(messages, target_tokens=500)
        self.assertTrue(changed)
        total_chars = sum(len(str(m.get("content") or "")) for m in messages)
        self.assertLess(total_chars, 300_000)
        # tool_call_id linkage must survive compaction -- protocol integrity.
        self.assertEqual(messages[3]["tool_call_id"], "call_1")

    def test_many_completed_cycles_get_evicted_under_budget(self):
        messages = [{"role": "system", "content": "sys"}]
        for i in range(60):
            messages.append({"role": "user" if i == 0 else "assistant", "content": f"turn {i}", "tool_calls": [
                {"id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f"f_{i}.py"})}}
            ]} if i > 0 else {"role": "user", "content": "start"})
            if i > 0:
                messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"contents of file {i}" * 50})
        messages.append({"role": "user", "content": "final question"})

        before = len(messages)
        _trim_tool_outputs(messages, target_tokens=200, keep_recent=6)
        self.assertLess(len(messages), before)
        # Every remaining assistant tool_calls[].function.arguments must
        # still be valid JSON after compaction/eviction.
        for message in messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    json.loads(call["function"]["arguments"])


class ContextRolloverUnitTests(_StatePatchMixin, unittest.TestCase):
    def test_rollover_persists_full_segment_and_builds_minimal_continuation(self):
        working_messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "z" * 300_000},
        ]
        scope_message = {"role": "system", "content": "WORKSPACE SCOPE (authoritative for this turn): ..."}

        continuation = _perform_context_rollover(
            working_messages, objective="do the thing", scope_roots=[Path("/tmp/proj")],
            scope_message=scope_message, session_id=1,
        )

        # Minimum viable context: leading system, scope, rollover note, user.
        self.assertEqual(len(continuation), 4)
        self.assertEqual(continuation[1], scope_message)
        self.assertIn("CONTEXT ROLLOVER", continuation[2]["content"])
        self.assertIn("do not tell the user to start over", continuation[2]["content"].lower())
        match = re.search(r"evidence_id=(evidence_\w+)", continuation[2]["content"])
        self.assertIsNotNone(match)

        # Nothing was lost -- the full segment is retrievable from durable
        # storage outside the provider prompt.
        segment = evidence_module.load_segment(1, match.group(1))
        self.assertIsNotNone(segment)
        self.assertEqual(segment["message_count"], 4)
        self.assertEqual(len(segment["messages"][3]["content"]), 300_000)

    def test_rollover_bounds_a_huge_objective_instead_of_reproducing_the_same_size(self):
        """Confirmed live: re-embedding the full objective twice (system
        note + new user message) made the rollover's own continuation
        roughly as large as what was just rolled over, so the very next
        budget check failed again immediately -- rollover looked like it
        ran but never actually shrank anything. The full text must still be
        recoverable from evidence, just not re-embedded in full."""
        huge_objective = "y" * 300_000
        working_messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": huge_objective},
        ]
        scope_message = {"role": "system", "content": "WORKSPACE SCOPE (authoritative for this turn): ..."}

        continuation = _perform_context_rollover(
            working_messages, objective=huge_objective, scope_roots=[Path("/tmp/proj")],
            scope_message=scope_message, session_id=1,
        )

        rendered_size = sum(len(str(m.get("content") or "")) for m in continuation)
        self.assertLess(rendered_size, 10_000)
        # The new user message is bounded too, not just the system note.
        new_user_message = continuation[3]
        self.assertEqual(new_user_message["role"], "user")
        self.assertLess(len(new_user_message["content"]), len(huge_objective))

        # The exact, untruncated original is still retrievable, not lost.
        match = re.search(r"evidence_id=(evidence_\w+)", continuation[2]["content"])
        segment = evidence_module.load_segment(1, match.group(1))
        self.assertEqual(segment["objective"], huge_objective)


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call_delta(index, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


class _FakeClient:
    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        chunks = self._rounds.pop(0)
        return _FakeStream(chunks)


class _RecordingRenderer:
    def __init__(self):
        self.events = []

    def handle_event(self, event):
        self.events.append(event)


class RetrieveEvidenceToolTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def test_retrieve_evidence_tool_returns_the_persisted_segment(self):
        with tempfile.TemporaryDirectory() as ws:
            evidence_id = evidence_module.store_segment(
                1, objective="earlier task", summary="found the bug in calc.py",
                messages=[{"role": "user", "content": "earlier task"}],
            )
            retrieve_args = json.dumps({"evidence_id": evidence_id})
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="retrieve_evidence", arguments=retrieve_args)]))],
                [_chunk(_delta(content="Found it in the earlier segment."))],
            ]
            client = _FakeClient(rounds)
            manager = SimpleNamespace(
                PROVIDERS={ProviderType.NVIDIA: SimpleNamespace(default_model="fake-model", context_window=32768)},
                get_client=lambda provider: client,
            )
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what did we find earlier? check evidence"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            tool_outputs = [e for e in renderer.events if e["event_type"] == "tool_output" and e["payload"]["tool"] == "retrieve_evidence"]
            self.assertEqual(len(tool_outputs), 1)
            self.assertIn("found the bug in calc.py", json.dumps(tool_outputs[0]["payload"]["result"]))

            offered_names = {t["function"]["name"] for t in client.calls[0]["tools"]}
            self.assertIn("retrieve_evidence", offered_names)


class LargerContextProviderFallbackTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def test_context_overflow_falls_forward_to_a_larger_context_provider(self):
        with tempfile.TemporaryDirectory() as ws:
            small_client = _FakeClient([])  # must never be called
            large_client = _FakeClient([[_chunk(_delta(content="Handled on the larger-context provider."))]])

            manager = SimpleNamespace(
                PROVIDERS={
                    ProviderType.NVIDIA: SimpleNamespace(default_model="small-model", context_window=100, tool_calling=True),
                    ProviderType.OPENROUTER: SimpleNamespace(default_model="large-model", context_window=200_000, tool_calling=True),
                },
                clients={ProviderType.NVIDIA: small_client, ProviderType.OPENROUTER: large_client},
                resolve_route=lambda provider, task_profile=None, quality_mode="quality": (
                    ProviderType.NVIDIA, manager.PROVIDERS[ProviderType.NVIDIA]
                ),
                get_client=lambda provider: manager.clients.get(provider),
                fallback_candidates=lambda current, task_profile=None: [ProviderType.OPENROUTER],
            )
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.AUTO, None,
                [{"role": "user", "content": "a" * 2000}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.summary, "Handled on the larger-context provider.")
            self.assertEqual(len(small_client.calls), 0)
            self.assertEqual(len(large_client.calls), 1)
            diagnostics = [e["payload"].get("content", "") for e in renderer.events if e["event_type"] == "diagnostics"]
            self.assertTrue(any("switching to openrouter" in d for d in diagnostics))


class CyclingLoopDetectionTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def test_alternating_two_call_cycle_is_detected_before_max_rounds(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "a.txt").write_text("a")
            (Path(ws) / "b.txt").write_text("b")
            read_a = json.dumps({"path": str(Path(ws) / "a.txt")})
            read_b = json.dumps({"path": str(Path(ws) / "b.txt")})
            # A, B, A, B, ... -- never identical two rounds in a row, so the
            # period-1 consecutive-identical guard alone would never fire.
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id=f"call_{i}", name="read_file", arguments=(read_a if i % 2 == 0 else read_b))]))]
                for i in range(30)
            ]
            client = _FakeClient(rounds)
            manager = SimpleNamespace(
                PROVIDERS={ProviderType.NVIDIA: SimpleNamespace(default_model="fake-model", context_window=32768)},
                get_client=lambda provider: client,
            )
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "check both files repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False, max_rounds=30,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("repeating cycle", outcome.error)
            self.assertLess(len(client.calls), 15)


if __name__ == "__main__":
    unittest.main()
