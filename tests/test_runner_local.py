import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import _tool_output_for_render, run_local_agent_turn


class ToolOutputForRenderTests(unittest.TestCase):
    """MCPServer.call_tool() nests its actual return value under a "result"
    key; render.py's tool_output handler only looks at top-level keys like
    content/stdout/exit_code. Without flattening, a real successful call
    could render nothing at all -- these lock in the three shapes that
    actually come back from mcp.py's built-in tools."""

    def test_string_result_becomes_content(self):
        flattened = _tool_output_for_render({"result": "file contents", "tool": "read_file", "success": True})
        self.assertEqual(flattened["content"], "file contents")

    def test_list_result_becomes_a_summary_count(self):
        flattened = _tool_output_for_render({
            "result": [{"name": "a.py"}, {"name": "b.py"}], "tool": "list_directory", "success": True,
        })
        self.assertEqual(flattened["content"], "2 item(s)")

    def test_empty_list_result_is_not_treated_as_missing(self):
        flattened = _tool_output_for_render({"result": [], "tool": "list_directory", "success": True})
        self.assertEqual(flattened["content"], "(empty)")

    def test_execute_command_dict_maps_return_code_to_exit_code(self):
        flattened = _tool_output_for_render({
            "result": {"stdout": "hi\n", "stderr": "", "return_code": 0, "success": True},
            "tool": "execute_command", "success": True,
        })
        self.assertEqual(flattened["stdout"], "hi\n")
        self.assertEqual(flattened["exit_code"], 0)
        self.assertNotIn("return_code", flattened)


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call_delta(index, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStream:
    """Async-iterable over a fixed list of chunks, mirroring what
    `await client.chat.completions.create(stream=True, ...)` returns."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


class _FakeClient:
    def __init__(self, rounds: list[list]):
        """`rounds` is a list of chunk-lists, one per completion call."""
        self._rounds = list(rounds)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        chunks = self._rounds.pop(0)
        return _FakeStream(chunks)


class _FakeManager:
    def __init__(self, client):
        self._client = client
        self.PROVIDERS = {
            ProviderType.OLLAMA: SimpleNamespace(default_model="fake-model", context_window=32768)
        }

    def get_client(self, provider):
        return self._client


class _RecordingRenderer:
    def __init__(self):
        self.events = []

    def handle_event(self, event):
        self.events.append(event)


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


class RunLocalAgentTurnTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def test_single_tool_call_round_then_completion(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "app.py")
            write_args = json.dumps({"path": target, "content": "x = 1\n"})
            rounds = [
                # Round 1: model streams a write_file tool call.
                [
                    _chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)])),
                ],
                # Round 2: model streams a final plain-text answer, no tool_calls.
                [
                    _chunk(_delta(content="Done, ")),
                    _chunk(_delta(content="wrote the file.")),
                ],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None, [{"role": "user", "content": "add a file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.summary, "Done, wrote the file.")
            self.assertEqual(Path(target).read_text(), "x = 1\n")

            event_types = [e["event_type"] for e in renderer.events]
            self.assertIn("tool_call_requested", event_types)
            self.assertIn("tool_output", event_types)
            self.assertIn("file_mutation", event_types)
            self.assertIn("ai_task_completed", event_types)

            mutation_events = [e for e in renderer.events if e["event_type"] == "file_mutation"]
            self.assertEqual(mutation_events[0]["payload"]["path"], str(Path(target).resolve()))

    def test_denied_tool_call_does_not_execute(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "app.py")
            write_args = json.dumps({"path": target, "content": "should not be written"})
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="OK, I won't write it."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None, [{"role": "user", "content": "add a file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="never", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertFalse(Path(target).exists())
            self.assertNotIn("file_mutation", [e["event_type"] for e in renderer.events])

    def test_read_only_tool_call_needs_no_approval_event(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "existing.txt").write_text("hello\n")
            read_args = json.dumps({"path": str(Path(ws) / "existing.txt")})
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="read_file", arguments=read_args)]))],
                [_chunk(_delta(content="The file says hello."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None, [{"role": "user", "content": "read the file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="never", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertNotIn("approval_required", [e["event_type"] for e in renderer.events])

    def test_round_limit_terminates_instead_of_looping_forever(self):
        with tempfile.TemporaryDirectory() as ws:
            # Every round returns another tool call, never plain content, and
            # each round's arguments differ (a distinct path) so the
            # identical-repetition guard never fires -- this isolates the
            # max_rounds safety valve from that separate guard, which has its
            # own test below.
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id=f"call_{i}", name="list_directory", arguments=f'{{"path": "dir_{i}"}}')]))]
                for i in range(3)
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None, [{"role": "user", "content": "loop forever"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False, max_rounds=3,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("3 tool-call rounds", outcome.error)
            self.assertEqual(len(client.calls), 3)

    def test_identical_tool_call_repeated_stops_before_max_rounds(self):
        with tempfile.TemporaryDirectory() as ws:
            # Same tool + same arguments every round: the model is stuck
            # repeating itself (e.g. polling a health check that never
            # changes), not making progress -- must stop well before
            # max_rounds, unlike the varied-arguments case above.
            read_args = '{"path": "."}'
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id=f"call_{i}", name="list_directory", arguments=read_args)]))]
                for i in range(50)
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None, [{"role": "user", "content": "check repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False, max_rounds=50,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("stuck repeating", outcome.error)
            # 3 identical rounds is enough to trip the guard -- nowhere near
            # the 50-round cap.
            self.assertLess(len(client.calls), 10)

    def test_context_budget_exceeded_stops_before_request(self):
        with tempfile.TemporaryDirectory() as ws:
            # A provider config with a tiny context window: even the system
            # prompt + one user message should already exceed it, so this
            # must abort before ever calling the client -- never guaranteed
            # to blow up with a real 400 from the provider.
            client = _FakeClient(rounds=[])
            manager = _FakeManager(client)
            manager.PROVIDERS[ProviderType.OLLAMA].context_window = 10
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None,
                [{"role": "user", "content": "a" * 2000}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("context window", outcome.error)
            self.assertEqual(len(client.calls), 0)

    def test_read_only_mode_refuses_mutating_tool_even_if_offered(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "app.py")
            write_args = json.dumps({"path": target, "content": "x = 1\n"})
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="Sorry, can't write in read-only mode."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None, [{"role": "user", "content": "write a file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False, read_only=True,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertFalse(Path(target).exists())
            self.assertNotIn("approval_required", [e["event_type"] for e in renderer.events])
            # tools offered to the model must exclude write_file/edit_file/execute_command
            create_call = client.calls[0]
            offered_names = {t["function"]["name"] for t in create_call["tools"]}
            self.assertNotIn("write_file", offered_names)
            self.assertIn("read_file", offered_names)

    def test_provider_unavailable_returns_failed_outcome(self):
        manager = SimpleNamespace(get_client=lambda provider: None, PROVIDERS={})
        renderer = _RecordingRenderer()
        with tempfile.TemporaryDirectory() as ws:
            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA, None, [{"role": "user", "content": "hi"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))
        self.assertEqual(outcome.status, "failed")
        self.assertIn("not available", outcome.error)


if __name__ == "__main__":
    unittest.main()
