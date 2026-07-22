import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rich.console import Console

from tamfis_code import evidence as evidence_module
from tamfis_code import state as state_module
from tamfis_code.providers import ProviderType
from tamfis_code.mcp import MCPServer
from tamfis_code.runner_local import (
    _checkpoint_resume_objective,
    _close_interrupted_tool_calls,
    _corrupted_lexical_stream_index,
    _is_real_resume_objective,
    _legacy_resume_messages,
    _messages_with_vision_content,
    _novel_continuation,
    _parse_swarm_tasks,
    _preview_diff_for_tool_call,
    _tool_output_for_render,
    build_vision_content_blocks,
    is_vision_image_path,
    run_local_agent_turn,
)


class StreamQualityTests(unittest.TestCase):
    def test_detects_recombined_provider_token_gibberish(self):
        corrupted = " ".join([
            "MistDotblankativityurpunite",
            "XCTurpMargueriteurptenhamftyabbDowurpennessfwinistrege",
            "LogoWebkiturpWonderBVPrompturp",
        ] * 30)
        self.assertIsNotNone(_corrupted_lexical_stream_index(corrupted))

    def test_does_not_reject_long_normal_prose(self):
        prose = (
            "The service configuration must be inspected before reporting a port, "
            "and the exact source should be cited in the answer. "
        ) * 30
        self.assertIsNone(_corrupted_lexical_stream_index(prose))

    def test_does_not_scan_inside_open_fenced_code(self):
        identifier_soup = " ".join(["LongCamelCaseIdentifierWithRepeatedFragment"] * 40)
        self.assertIsNone(_corrupted_lexical_stream_index("```javascript\n" + identifier_soup))


class VisionContentTests(unittest.TestCase):
    """Real image attachments (--attach a.png) used to only ever reach the
    model as a plain-text file path with a "use read_file" instruction --
    read_file can't decode binary image bytes into anything meaningful, so
    the model never actually saw the picture. These lock in the splice
    helper that builds real multipart content without ever mutating the
    canonical, always-plain-text working_messages list itself (every
    resume/dedup/checkpoint helper assumes message["content"] is a str)."""

    def test_is_vision_image_path_recognises_common_formats(self):
        for name in ("a.png", "B.JPG", "c.jpeg", "d.gif", "e.webp"):
            self.assertTrue(is_vision_image_path(name), name)

    def test_is_vision_image_path_rejects_non_images(self):
        for name in ("a.pdf", "b.zip", "c.py", "d.txt", "noextension"):
            self.assertFalse(is_vision_image_path(name), name)

    def test_build_vision_content_blocks_reads_and_encodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\nfakepixels")
            blocks = build_vision_content_blocks([str(path)])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "image_url")
        self.assertTrue(blocks[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_build_vision_content_blocks_skips_non_image_and_missing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "notes.txt"
            text_path.write_text("hello")
            blocks = build_vision_content_blocks([str(text_path), str(Path(tmp) / "missing.png")])
        self.assertEqual(blocks, [])

    def test_messages_with_vision_content_splices_without_mutating_original(self):
        original = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "describe this screenshot"},
        ]
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        patched = _messages_with_vision_content(original, 1, blocks)
        # Original list and message are untouched -- working_messages must
        # stay plain-text for the rest of the turn's resume/dedup logic.
        self.assertEqual(original[1]["content"], "describe this screenshot")
        self.assertIsInstance(original[1]["content"], str)
        # The returned copy carries real multipart content for the API call.
        self.assertEqual(patched[1]["content"][0], {"type": "text", "text": "describe this screenshot"})
        self.assertEqual(patched[1]["content"][1], blocks[0])
        self.assertIsNot(patched, original)
        self.assertIsNot(patched[1], original[1])

    def test_messages_with_vision_content_noop_without_blocks_or_index(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertIs(_messages_with_vision_content(messages, None, [{"type": "image_url"}]), messages)
        self.assertIs(_messages_with_vision_content(messages, 0, None), messages)
        self.assertIs(_messages_with_vision_content(messages, 5, [{"type": "image_url"}]), messages)

    def test_messages_with_vision_content_ignores_non_user_target(self):
        messages = [{"role": "assistant", "content": "ok"}]
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        self.assertIs(_messages_with_vision_content(messages, 0, blocks), messages)


class ParseSwarmTasksTests(unittest.TestCase):
    """delegate_parallel_tasks's `tasks` argument accepts either plain
    strings (backward compatible) or {"objective", "agent_type"} objects
    (SWARM_TOOL_SCHEMA's anyOf) -- this is the shared parsing both shapes
    go through before reaching run_swarm/execute_tasks."""

    def test_plain_strings_have_no_agent_type(self):
        objectives, agent_types = _parse_swarm_tasks(["task a", "task b"])
        self.assertEqual(objectives, ["task a", "task b"])
        self.assertEqual(agent_types, [None, None])

    def test_object_items_carry_their_agent_type(self):
        objectives, agent_types = _parse_swarm_tasks([
            {"objective": "review this", "agent_type": "reviewer"},
            {"objective": "plan that"},
        ])
        self.assertEqual(objectives, ["review this", "plan that"])
        self.assertEqual(agent_types, ["reviewer", None])

    def test_mixed_plain_and_object_items(self):
        objectives, agent_types = _parse_swarm_tasks([
            "plain task", {"objective": "typed task", "agent_type": "planner"},
        ])
        self.assertEqual(objectives, ["plain task", "typed task"])
        self.assertEqual(agent_types, [None, "planner"])

    def test_blank_objective_is_skipped(self):
        objectives, agent_types = _parse_swarm_tasks(["  ", {"objective": ""}, "real task"])
        self.assertEqual(objectives, ["real task"])
        self.assertEqual(agent_types, [None])


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


class PreviewDiffForToolCallTests(unittest.TestCase):
    """Unit coverage for the read-only diff-preview helper used by the
    approval panel -- must never write anything to disk itself."""

    def test_write_file_new_file_shows_a_pure_addition_diff(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "new.py")
            server = MCPServer(workspace_root=ws)
            diff = _preview_diff_for_tool_call(server, "write_file", {"path": target, "content": "x = 1\n"})
            self.assertIn("+x = 1", diff)
            self.assertFalse(Path(target).exists(), "preview must not write the file")

    def test_write_file_existing_file_shows_removed_and_added_lines(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "existing.py")
            Path(target).write_text("x = 1\n")
            server = MCPServer(workspace_root=ws)
            diff = _preview_diff_for_tool_call(server, "write_file", {"path": target, "content": "x = 2\n"})
            self.assertIn("-x = 1", diff)
            self.assertIn("+x = 2", diff)
            self.assertEqual(Path(target).read_text(), "x = 1\n", "preview must not modify the file")

    def test_edit_file_unique_match_shows_the_replacement(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "existing.py")
            Path(target).write_text("def old_name():\n    pass\n")
            server = MCPServer(workspace_root=ws)
            diff = _preview_diff_for_tool_call(
                server, "edit_file", {"path": target, "old_string": "old_name", "new_string": "new_name"},
            )
            self.assertIn("-def old_name():", diff)
            self.assertIn("+def new_name():", diff)

    def test_edit_file_ambiguous_match_returns_none_not_a_wrong_diff(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "existing.py")
            Path(target).write_text("x = 1\nx = 1\n")
            server = MCPServer(workspace_root=ws)
            diff = _preview_diff_for_tool_call(
                server, "edit_file", {"path": target, "old_string": "x = 1", "new_string": "x = 2"},
            )
            self.assertIsNone(diff)

    def test_edit_file_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as ws:
            server = MCPServer(workspace_root=ws)
            diff = _preview_diff_for_tool_call(
                server, "edit_file", {"path": str(Path(ws) / "nope.py"), "old_string": "a", "new_string": "b"},
            )
            self.assertIsNone(diff)

    def test_other_tools_return_none(self):
        with tempfile.TemporaryDirectory() as ws:
            server = MCPServer(workspace_root=ws)
            self.assertIsNone(_preview_diff_for_tool_call(server, "execute_command", {"command": "ls"}))
            self.assertIsNone(_preview_diff_for_tool_call(server, "read_file", {"path": "x"}))


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call_delta(index, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


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
            ProviderType.NVIDIA: SimpleNamespace(default_model="fake-model", context_window=32768)
        }

    def get_client(self, provider):
        return self._client


class _RecordingRenderer:
    def __init__(self):
        self.events = []

    def handle_event(self, event):
        self.events.append(event)

    def suspend_live(self):
        self.events.append({"event_type": "_live_suspended"})

    def resume_live(self):
        self.events.append({"event_type": "_live_resumed"})


class _StatePatchMixin:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._evidence_original = evidence_module.EVIDENCE_DIR
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        evidence_module.EVIDENCE_DIR = base / ".config" / "evidence"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        evidence_module.EVIDENCE_DIR = self._evidence_original
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
                # "add a file" classifies as plan-worthy -- the initial
                # reasoning-plan request consumes the first round, and the
                # one-time post-evidence replan (triggered right after the
                # first round with real tool_calls) consumes a third round,
                # both sandwiched around the two real tool-loop rounds below.
                # Non-JSON content makes each fall back to whatever plan was
                # already in effect.
                [_chunk(_delta(content="not a plan"))],
                # Round 1: model streams a write_file tool call.
                [
                    _chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)])),
                ],
                [_chunk(_delta(content="not a plan"))],
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
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "add a file"}],
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

    def test_truncated_final_answer_is_continued_not_accepted_as_complete(self):
        """Confirmed live: a long, reasoning-heavy final answer (e.g. a
        full-stack audit) can hit the provider's max-output-tokens limit
        mid-sentence. finish_reason=="length" was computed by
        provider_protocols.py but never read anywhere -- a truncated,
        trailing-off partial with no tool_calls was indistinguishable from
        a genuinely complete answer, so it was accepted as-is instead of
        being continued."""
        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(content="This is the first half of a long report, cut off"), finish_reason="length")],
                [_chunk(_delta(content=" and here is the rest of it, finishing naturally."), finish_reason="stop")],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "summarize the incident at length"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(
                outcome.summary,
                "This is the first half of a long report, cut off"
                " and here is the rest of it, finishing naturally.",
            )
            self.assertNotIn("still incomplete", outcome.summary)
            diagnostics = [e["payload"].get("content", "") for e in renderer.events if e["event_type"] == "diagnostics"]
            self.assertFalse(any("cut off" in d for d in diagnostics))
            # The continuation call must not offer tools -- it's purely
            # "keep writing text", never a new tool-call opportunity.
            self.assertEqual(client.calls[-1].get("tools"), None)

    def test_truncation_overlap_is_not_rendered_twice(self):
        self.assertEqual(
            _novel_continuation("alpha beta gamma", "beta gamma delta"),
            " delta",
        )

    def test_xml_textual_execute_command_is_hidden_and_executed(self):
        """OpenAI-compatible models sometimes stream XML-ish calls as text."""
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "benchmark").write_text("ok\n")
            rounds = [
                [
                    _chunk(_delta(content="<tool_")),
                    _chunk(_delta(content=(
                        "call> <function=execute_command> <parameter=command> "
                        "find . -type f -name benchmark 2>/dev/null </tool_call>"
                    ))),
                ],
                [_chunk(_delta(content="Found the benchmark file."))],
            ]
            client = _FakeClient(rounds)
            renderer = _RecordingRenderer()
            outcome = asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": "inspect the repository for benchmark files"}],
                self._console(), renderer, workspace_root=ws, session_id=1,
                approval_policy="never", interactive=False, read_only=True,
            ))

            self.assertEqual(outcome.status, "completed")
            requested = [
                event for event in renderer.events
                if event["event_type"] == "tool_call_requested"
            ]
            self.assertTrue(any(event["payload"]["name"] == "execute_command" for event in requested))
            streamed = "".join(
                event["payload"].get("content", "")
                for event in renderer.events if event["event_type"] == "assistant_delta"
            )
            self.assertNotIn("<tool_call", streamed)
            self.assertNotIn("<function=", streamed)

    def test_resume_rehydrates_interrupted_turn_and_clears_checkpoint(self):
        with tempfile.TemporaryDirectory() as ws:
            state_module.save_turn_checkpoint(
                1,
                objective="inspect the repository and list the three gaps",
                mode="read_only",
                messages=[
                    {"role": "user", "content": "inspect the repository and list the three gaps"},
                    {"role": "assistant", "content": "1. API\n2. CLI\n3. Tests"},
                ],
                status="interrupted",
            )
            client = _FakeClient([[_chunk(_delta(content="Completed steps 1, 2, and 3."))]])
            outcome = asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": "proceed with 1, 2, and 3"}],
                self._console(), _RecordingRenderer(), workspace_root=ws,
                session_id=1, approval_policy="never", interactive=False,
                read_only=True,
            ))

            self.assertEqual(outcome.status, "completed")
            sent = json.dumps(client.calls[0]["messages"])
            self.assertIn("inspect the repository and list the three gaps", sent)
            self.assertIn("proceed with 1, 2, and 3", sent)
            self.assertIsNone(state_module.get_session_state(1).turn_checkpoint)
            memory = state_module.CONFIG_DIR / ".memory" / "session-1.json"
            self.assertTrue(memory.is_file())
            memory_payload = json.loads(memory.read_text())
            self.assertIsNone(memory_payload["turn_checkpoint"])
            self.assertIn("Completed steps 1, 2, and 3.", memory_payload["conversation_summary"])

    def test_resume_recovers_legacy_progress_from_related_parent_workspace(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            child = root / "backend"
            child.mkdir()
            state_module.save_session_state(
                1,
                workspace_root=str(root),
                primary_workspace=str(root),
                completed_actions=[{
                    "type": "tool",
                    "purpose": "Execute execute_command for: please execute benchmark.py",
                    "status": "completed",
                    "success": True,
                    "stdout": "benchmark.py started; provider timed out after 60 seconds",
                }],
                modified_files=[{"path": "unrelated/other-conversation.txt"}],
                conversation_summary="The benchmark run reached the provider timeout.",
            )
            state_module.save_session_state(
                2, workspace_root=str(child), primary_workspace=str(child),
            )
            client = _FakeClient([[_chunk(_delta(content="Resumed the benchmark from its recorded timeout."))]])

            outcome = asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": "continue from where you stopped"}],
                self._console(), _RecordingRenderer(), workspace_root=str(child),
                session_id=2, approval_policy="never", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            sent = json.dumps(client.calls[0]["messages"])
            self.assertIn("please execute benchmark.py", sent)
            self.assertIn("provider timed out after 60 seconds", sent)
            self.assertNotIn("I don't have context", sent)
            self.assertNotIn("other-conversation.txt", sent)

    def test_legacy_resume_reclassifies_on_the_fresh_instruction_not_the_stale_one(self):
        # Confirmed live: a legacy resume's inferred_objective can be a
        # low-complexity leftover (e.g. a plain question) with no bearing on
        # what the user is asking for THIS turn. Before this fix, `objective`
        # was set to that stale text alone, so classify_task saw none of the
        # user's actual "fix everything" instruction -- task_profile came
        # back QUESTION (requires_tools=False), which silently disabled the
        # give-up/capitulation guard (it gates on task_profile.requires_tools)
        # even though real tool-driven work was clearly being asked for. A
        # model that then gives up in prose with zero tool calls must be
        # caught and retried/failed, not accepted as a "completed" answer.
        state_module.save_session_state(
            1,
            workspace_root="/workspace",
            primary_workspace="/workspace",
            conversation_history=[
                {"role": "user", "content": "what does the login page do?"},
                {"role": "assistant", "content": "It renders a form and posts credentials."},
            ],
        )
        # Reclassifying to DEBUG/high-complexity makes this plan-worthy, so
        # the very first completion call is a reasoning-plan request, not
        # the main round -- give it enough identical "give up" rounds to
        # cover that plus the guard's one retry.
        stuck_chunks = [_chunk(_delta(
            content="The task is stuck due to the lack of a clear next step."
        ), finish_reason="stop")]
        client = _FakeClient([stuck_chunks, stuck_chunks, stuck_chunks])

        outcome = asyncio.run(run_local_agent_turn(
            _FakeManager(client), ProviderType.NVIDIA, None,
            [{"role": "user", "content": "continue until you fix everything, don't ask me for confirmation, just go ahead"}],
            self._console(), _RecordingRenderer(), workspace_root="/workspace",
            session_id=1, approval_policy="never", interactive=False,
        ))

        # The guard must have engaged (a retry, then failure) rather than
        # accepting the give-up text as a completed answer.
        self.assertGreaterEqual(len(client.calls), 2)
        self.assertEqual(outcome.status, "failed")
        self.assertIn("gave up", outcome.error or "")

    def test_unmatched_tool_call_is_closed_as_interrupted_before_resume(self):
        repaired = _close_interrupted_tool_calls([{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "execute_command", "arguments": '{"command":"touch x"}'},
            }],
        }])
        self.assertEqual(repaired[-1]["role"], "tool")
        self.assertEqual(repaired[-1]["tool_call_id"], "call_1")
        self.assertIn("Do not blindly repeat", repaired[-1]["content"])

    def test_legacy_resume_rejects_internal_plan_checkpoint_and_uses_latest_real_user(self):
        state_module.save_session_state(
            1,
            workspace_root="/workspace",
            primary_workspace="/workspace",
            turn_checkpoint={
                "objective": "active_plan=plan_deadbeef",
                "messages": [{"role": "user", "content": "active_plan=plan_deadbeef"}],
            },
            conversation_history=[
                {"role": "user", "content": "continue from where you stopped"},
                {"role": "assistant", "content": "I don't have context from a previous turn in this conversation."},
                {"role": "user", "content": "active_plan=plan_deadbeef"},
                {"role": "assistant", "content": "I searched for the internal plan but found nothing."},
                {"role": "user", "content": "check STATUS.md, README.md, and the saved memory"},
            ],
        )
        state = state_module.get_session_state(1)
        messages, objective = _legacy_resume_messages(state, "continue")
        serialized = json.dumps(messages)

        self.assertFalse(_is_real_resume_objective("active_plan=plan_deadbeef"))
        self.assertEqual(objective, "check STATUS.md, README.md, and the saved memory")
        self.assertNotIn("don't have context", serialized.lower())
        self.assertNotIn("active_plan=", serialized)

    def test_legacy_resume_combines_unfinished_task_with_clarification(self):
        state_module.save_session_state(
            1,
            workspace_root="/workspace",
            primary_workspace="/workspace",
            conversation_history=[
                {"role": "user", "content": "unrelated earlier request"},
                {"role": "assistant", "content": "That earlier answer is complete."},
                {"role": "user", "content": "check your previous status and continue"},
                {"role": "assistant", "content": "Let me examine the status files."},
                {"role": "user", "content": "backend is tamgpt6 and frontend is tamfis-frontend"},
                {"role": "assistant", "content": "I'll start by checking both projects."},
            ],
        )

        messages, objective = _legacy_resume_messages(
            state_module.get_session_state(1), "continue"
        )
        serialized = json.dumps(messages)

        self.assertTrue(objective.startswith("check your previous status and continue"))
        self.assertIn("backend is tamgpt6", objective)
        self.assertNotIn("unrelated earlier request", objective)
        self.assertIn("tamfis-frontend", serialized)

    def test_truncated_final_answer_gives_up_after_the_continuation_cap(self):
        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(content=f"part {i} "), finish_reason="length")]
                for i in range(8)  # initial round + MAX_TRUNCATION_CONTINUATIONS(6) all truncated, plus one spare
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "write a very long report"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("still incomplete", outcome.summary)
            self.assertIn("part 0", outcome.summary)
            self.assertIn("part 6", outcome.summary)  # initial + 6 continuations = 7 parts used

    def test_live_queued_instruction_is_spliced_into_the_running_turn(self):
        """User-requested: a standalone task that's already running should
        be reachable from a SECOND terminal's `tamfis-code queue "..."`
        against the same session -- before this, run_local_agent_turn never
        looked at the queue at all (cli.py's own queue command said so:
        "a standalone local turn is always synchronous ... there's nothing
        'live' to push into"). Checked at the top of every round."""
        with tempfile.TemporaryDirectory() as ws:
            state_module.get_session_state(1)  # create session 1 first
            item = state_module.enqueue_instruction(
                1, "actually focus only on the auth module", classification="follow_up",
            )
            rounds = [[_chunk(_delta(content="Summary of the incident."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "summarize the incident"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            sent_messages = client.calls[0]["messages"]
            self.assertTrue(any(
                "actually focus only on the auth module" in str(m.get("content", ""))
                for m in sent_messages
            ))
            queued_after = state_module.get_session_state(1).queued_user_instructions
            consumed = next(entry for entry in queued_after if entry["id"] == item.id)
            self.assertEqual(consumed["status"], "completed")
            diagnostics = [e["payload"].get("content", "") for e in renderer.events if e["event_type"] == "diagnostics"]
            self.assertTrue(any("Live instruction received mid-task" in d for d in diagnostics))

    def test_live_cancel_instruction_stops_the_turn_before_any_completion_call(self):
        with tempfile.TemporaryDirectory() as ws:
            state_module.get_session_state(1)
            state_module.enqueue_instruction(1, "stop, wrong target", classification="cancel")
            client = _FakeClient([[_chunk(_delta(content="should never be reached"))]])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "summarize the incident"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "cancelled")
            self.assertIn("stop, wrong target", outcome.error)
            self.assertEqual(client.calls, [])  # never even reached the provider

    def test_non_live_classification_is_left_queued_for_the_next_turn(self):
        """"reprioritise" only makes sense against a not-yet-started
        backlog -- it must not be claimed/consumed by an already-running
        turn's round loop the way append/follow_up/cancel/pause are."""
        with tempfile.TemporaryDirectory() as ws:
            state_module.get_session_state(1)
            item = state_module.enqueue_instruction(1, "do X first", classification="reprioritise")
            rounds = [[_chunk(_delta(content="Summary of the incident."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "summarize the incident"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            queued_after = state_module.get_session_state(1).queued_user_instructions
            consumed = next(entry for entry in queued_after if entry["id"] == item.id)
            self.assertEqual(consumed["status"], "queued")
            sent_messages = client.calls[0]["messages"]
            self.assertFalse(any("do X first" in str(m.get("content", "")) for m in sent_messages))

    def test_degenerate_repetition_stops_generation_early_instead_of_looping_forever(self):
        """Confirmed live: NVIDIA nemotron got stuck repeating the exact same
        short phrase thousands of times instead of producing real output or
        a normal stop. Before this fix, that garbage ran all the way to the
        provider's token cap, finish_reason=="length" told the truncation-
        continuation loop to ask for MORE of the same, and it kept re-
        feeding the growing garbage back to the model for up to 6 more
        rounds until the process was OOM-killed. The fix detects the repeat
        mid-stream and aborts that completion call immediately, so the round
        finishes with a single, truncated, captioned answer -- no
        continuation call is ever made for a stream that never legitimately
        finished."""
        with tempfile.TemporaryDirectory() as ws:
            looping_phrase = "We have execute_command? Not listed. "
            rounds = [
                [_chunk(_delta(content=looping_phrase)) for _ in range(50)]
                + [_chunk(_delta(content=""), finish_reason="length")],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "summarize the incident at length"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("degenerate repetition loop", outcome.error)
            # Only the one, aborted completion call happened -- the
            # truncation-continuation loop must never fire for a stream that
            # was stopped for looping, not for hitting a real length cap.
            self.assertEqual(len(client.calls), 1)
            diagnostics = [e["payload"].get("content", "") for e in renderer.events if e["event_type"] == "diagnostics"]
            self.assertTrue(any("repeating itself in a loop" in d for d in diagnostics))
            visible = "".join(
                e["payload"].get("content", "") for e in renderer.events
                if e["event_type"] == "assistant_delta"
            )
            self.assertNotIn(looping_phrase, visible)

    def test_repeated_conversation_transcript_is_stopped_early(self):
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[
                _chunk(_delta(content=(
                    'Then the user said "audit the site". Then the assistant responded '
                    "with a long analysis. " if index % 2 else
                    'Then the user said "continue until fixed". Then the assistant responded '
                    "with another long analysis. "
                ))) for index in range(20)
            ]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()
            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "summarize the site"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))
            self.assertEqual(outcome.status, "failed")
            self.assertIn("repeated conversation transcript", outcome.error)
            self.assertEqual(len(client.calls), 1)

    def test_change_request_completed_with_no_mutation_gets_a_caveat(self):
        """Confirmed live: a weak model can narrate "I'll fix this" -- complete
        with a fabricated "corrected" code block -- without ever calling
        write_file/edit_file, and the turn still completes normally since it
        simply stopped requesting tool calls. This must not read as an
        unqualified success when the objective clearly asked for a change."""
        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                # "fix the bug in calc.py" classifies as plan-worthy -- the
                # reasoning-plan request consumes the first round.
                [_chunk(_delta(content="not a plan"))],
                [
                    _chunk(_delta(content="I've fixed the bug by changing n + 2 to n + 1.")),
                ],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "fix the bug in calc.py"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("No files were changed", outcome.summary)

    def test_read_only_question_with_no_mutation_gets_no_caveat(self):
        """The same zero-mutation completion is completely normal for a
        read-only question -- the caveat must not fire just because nothing
        changed, only when the objective asked for a change and nothing did."""
        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(content="This project is a standalone coding agent CLI."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what does this project do?"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertNotIn("No files were changed", outcome.summary)

    def test_image_content_blocks_reach_the_provider_when_vision_is_supported(self):
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[_chunk(_delta(content="I can see the screenshot."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            manager.PROVIDERS[ProviderType.NVIDIA].vision_supported = True
            renderer = _RecordingRenderer()
            blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what does this screenshot show?"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                image_content_blocks=blocks,
            ))

            self.assertEqual(outcome.status, "completed")
            sent_messages = client.calls[0]["messages"]
            user_messages = [m for m in sent_messages if m.get("role") == "user"]
            self.assertEqual(
                user_messages[-1]["content"][0],
                {"type": "text", "text": "what does this screenshot show?"},
            )
            self.assertEqual(user_messages[-1]["content"][1], blocks[0])

    def test_image_content_blocks_are_not_sent_when_provider_lacks_vision(self):
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[_chunk(_delta(content="I can't see images."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)  # default fake has no vision_supported -> treated as False
            renderer = _RecordingRenderer()
            blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what does this screenshot show?"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                image_content_blocks=blocks,
            ))

            self.assertEqual(outcome.status, "completed")
            sent_messages = client.calls[0]["messages"]
            user_messages = [m for m in sent_messages if m.get("role") == "user"]
            self.assertIsInstance(user_messages[-1]["content"], str)

    def test_pre_tool_use_hook_blocks_the_tool_call(self):
        # Real Claude-Code-style PreToolUse parity: a project .tamfis/hooks.toml
        # can veto a tool call before mcp.py ever executes it.
        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[pre_tool_use]]\nmatcher = "write_file"\ncommand = "echo \'no writes in tests\' 1>&2; exit 2"\n'
            )
            write_args = json.dumps({"path": str(Path(ws) / "app.py"), "content": "x = 1\n"})
            rounds = [
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(content="Blocked, as expected."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "add a file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertFalse((Path(ws) / "app.py").exists())
            tool_events = [e for e in renderer.events if e["event_type"] == "tool_output"]
            self.assertTrue(any("Blocked by hook" in str(e["payload"]["result"]) for e in tool_events))

    def test_post_tool_use_hook_feedback_is_added_to_the_conversation(self):
        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[post_tool_use]]\nmatcher = "write_file"\ncommand = "echo \'formatted by hook\' 1>&2"\n'
            )
            target = str(Path(ws) / "app.py")
            write_args = json.dumps({"path": target, "content": "x = 1\n"})
            rounds = [
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "add a file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(Path(target).read_text(), "x = 1\n")
            sent_messages = client.calls[-1]["messages"]
            self.assertTrue(any(
                "formatted by hook" in str(m.get("content"))
                for m in sent_messages if m.get("role") == "system"
            ))

    def test_fake_tool_call_in_text_gets_a_caveat(self):
        """Confirmed live on a plain "audit/check" objective (no change-request
        verb, so the mutation-based caveat above wouldn't fire): the model wrote
        out a fenced ```execute_command(...)``` block in its own prose instead of
        actually calling the tool, and the turn completed with zero real
        tool_calls. This is a much more specific signal than a verb heuristic --
        our own tool names essentially never legitimately appear as a literal
        function call in real prose."""
        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(content='Let\'s run it:\n```python\nexecute_command("pytest")\n```'))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "check if the tests pass"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("unexecuted tool call", outcome.summary)

    def test_fake_json_shaped_tool_call_in_text_gets_a_caveat(self):
        """Confirmed live against nvidia/nemotron-3-super-120b: instead of a
        paren-style fake call (the shape covered above), the model narrated a
        {"tool": "read_file", "argument": {"path": ...}} JSON object in plain
        prose -- no real tool_calls either round, and it kept repeating this
        same JSON blob turn after turn since nothing about it was ever
        flagged as fake. The paren-only regex never matches valid JSON (no
        tool-name-immediately-followed-by-open-paren anywhere in it)."""
        with tempfile.TemporaryDirectory() as ws:
            fake_json = (
                '{\n  "tool": "read_file",\n  "argument": {\n'
                '    "path": "/home/finima/www/wp-config.php"\n  }\n}'
            )
            rounds = [[_chunk(_delta(content=fake_json))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "check if the tests pass"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("unexecuted tool call", outcome.summary)

    def test_future_tool_narration_is_retried_as_a_real_tool_call(self):
        """A promise to inspect files must not be accepted as completion."""
        with tempfile.TemporaryDirectory() as ws:
            status = Path(ws) / "STATUS.md"
            status.write_text("ready\n")
            read_args = json.dumps({"path": str(status)})
            rounds = [
                [_chunk(_delta(content="Let me examine STATUS.md to understand the current state."))],
                [_chunk(_delta(tool_calls=[
                    _tool_call_delta(0, call_id="call_1", name="read_file", arguments=read_args)
                ]))],
                [_chunk(_delta(content="STATUS.md reports ready."))],
            ]
            client = _FakeClient(rounds)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": "inspect STATUS.md"}],
                self._console(), renderer, workspace_root=ws, session_id=1,
                approval_policy="never", interactive=False, read_only=True,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(len(client.calls), 3)
            self.assertIn("issued no registered tool call", json.dumps(client.calls[1]["messages"]))
            self.assertTrue(any(
                event["event_type"] == "tool_call_requested"
                and event["payload"]["name"] == "read_file"
                for event in renderer.events
            ))

    def test_repeated_future_tool_narration_is_checkpointed_not_completed(self):
        with tempfile.TemporaryDirectory() as ws:
            client = _FakeClient([
                [_chunk(_delta(content="Let me check the status files."))],
                [_chunk(_delta(content="Now let me examine the README files."))],
            ])
            outcome = asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": "inspect the repository status"}],
                self._console(), _RecordingRenderer(), workspace_root=ws, session_id=1,
                approval_policy="never", interactive=False, read_only=True,
            ))

            self.assertEqual(outcome.status, "failed")
            checkpoint = state_module.get_session_state(1).turn_checkpoint
            self.assertEqual(checkpoint["status"], "interrupted")
            self.assertIn("without issuing a registered tool call", checkpoint["last_error"])

    def test_resume_objective_keeps_original_task_and_later_clarification(self):
        checkpoint = {
            "objective": "TamfisGPT backend is tamgpt6 and frontend is tamfis-frontend",
            "messages": [
                {"role": "user", "content": "unrelated earlier conversation about billing"},
                {"role": "assistant", "content": "The billing answer is complete."},
                {"role": "user", "content": "check your previous status and continue"},
                {"role": "assistant", "content": "Let me examine the workspace."},
                {"role": "user", "content": "TamfisGPT backend is tamgpt6 and frontend is tamfis-frontend"},
                {"role": "user", "content": "continue"},
            ],
        }

        objective = _checkpoint_resume_objective(checkpoint)
        self.assertTrue(objective.startswith("check your previous status and continue"))
        self.assertIn("backend is tamgpt6", objective)
        self.assertNotIn("billing", objective)
        self.assertNotEqual(objective, "continue")

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
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "add a file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="never", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertFalse(Path(target).exists())
            self.assertNotIn("file_mutation", [e["event_type"] for e in renderer.events])

    def test_live_status_line_is_suspended_before_the_approval_panel_prints(self):
        """Confirmed live via a pty capture: suspending the live status line
        only right before the blocking input prompt (rather than before the
        approval_required panel itself) still let a stray spinner frame
        render between the panel and the prompt, because Live's background
        refresh thread redraws on its own timer independent of anything else
        writing to the console. Suspend must happen before the panel event
        is even emitted, and resume only after the decision is made."""
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "app.py")
            write_args = json.dumps({"path": target, "content": "x = 1\n"})
            rounds = [
                # "add a file" classifies as plan-worthy -- the initial plan
                # request and the one-time post-evidence replan (triggered
                # right after the first round with tool_calls) each consume
                # a round of their own, sandwiching the real tool-loop rounds.
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "add a file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="ask", interactive=False,
            ))

            event_types = [e["event_type"] for e in renderer.events]
            suspend_index = event_types.index("_live_suspended")
            approval_index = event_types.index("approval_required")
            resume_index = event_types.index("_live_resumed")
            self.assertLess(
                suspend_index, approval_index,
                "the live status line must be suspended before the approval panel prints",
            )
            self.assertGreater(
                resume_index, approval_index,
                "the live status line must resume only after the approval decision is made",
            )

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
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "read the file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="never", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertNotIn("approval_required", [e["event_type"] for e in renderer.events])

    def test_service_restart_before_any_mutation_gets_a_warning_in_the_approval_reason(self):
        """Confirmed live: a weak model can claim a fix was applied (in prose
        only, never having called write_file/edit_file) and then restart the
        service that was supposed to pick it up -- the restart succeeds, so
        the turn reads as clean even though nothing on disk changed. The
        end-of-turn no-mutation caveat only fires after that restart already
        ran. The approval panel is the last point a human can catch this
        before it happens, so it must carry the same warning."""
        with tempfile.TemporaryDirectory() as ws:
            restart_args = json.dumps({"command": "systemctl restart caddy"})
            rounds = [
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="execute_command", arguments=restart_args)]))],
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "fix the theme css"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="ask", interactive=False,
            ))

            approval_events = [e for e in renderer.events if e["event_type"] == "approval_required"]
            self.assertEqual(len(approval_events), 1)
            self.assertIn("No files have been changed yet", approval_events[0]["payload"]["reason"])

    def test_service_restart_after_a_real_mutation_gets_no_warning(self):
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "style.css")
            write_args = json.dumps({"path": target, "content": "body { max-width: none; }\n"})
            restart_args = json.dumps({"command": "systemctl restart caddy"})
            rounds = [
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_2", name="execute_command", arguments=restart_args)]))],
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "fix the theme css"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="safe", interactive=False,
            ))

            self.assertTrue(Path(target).exists(), "the write_file call must have actually executed")
            approval_events = [e for e in renderer.events if e["event_type"] == "approval_required"]
            restart_approval = next(e for e in approval_events if "systemctl" in e["payload"]["command"])
            self.assertNotIn("No files have been changed yet", restart_approval["payload"]["reason"])

    def test_write_file_approval_carries_a_real_diff_not_raw_json_content(self):
        # Before this, the approval panel for write_file/edit_file rendered
        # the raw tool-call arguments -- for write_file, the ENTIRE proposed
        # new file content as a JSON string -- instead of a diff. The
        # approval event payload must now carry a real unified diff, and
        # the fallback "command" text must no longer duplicate the full
        # file content since the diff panel already shows the change.
        with tempfile.TemporaryDirectory() as ws:
            target = str(Path(ws) / "style.css")
            Path(target).write_text("body { max-width: 800px; }\n")
            write_args = json.dumps({"path": target, "content": "body { max-width: none; }\n"})
            rounds = [
                [_chunk(_delta(content="not a plan"))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="not a plan"))],  # consumed by the post-evidence replan attempt
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "fix the theme css"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="safe", interactive=False,
            ))

            approval_events = [e for e in renderer.events if e["event_type"] == "approval_required"]
            self.assertEqual(len(approval_events), 1)
            payload = approval_events[0]["payload"]
            diff = payload["diff"]
            self.assertIn("-body { max-width: 800px; }", diff)
            self.assertIn("+body { max-width: none; }", diff)
            self.assertNotIn("max-width: none", payload["command"])
            self.assertIn("write_file(path=", payload["command"])

    def test_nested_semantic_failure_inside_a_transport_success_envelope_is_reported_as_failed(self):
        """MCPServer.call_tool() returns success=True merely because the
        Python tool function returned normally -- mcp.py's `_read_file`
        encodes a missing file as the *string* "Error: File '...' not
        found", not a raised exception. Both the human-facing tool_output
        event and the model-facing tool message must see this as a failure,
        not a transport-success envelope."""
        with tempfile.TemporaryDirectory() as ws:
            missing = str(Path(ws) / "does_not_exist.txt")
            read_args = json.dumps({"path": missing})
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="read_file", arguments=read_args)]))],
                [_chunk(_delta(content="The file does not exist."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "read the missing file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="never", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            tool_outputs = [e for e in renderer.events if e["event_type"] == "tool_output"]
            self.assertEqual(len(tool_outputs), 1)
            rendered_result = tool_outputs[0]["payload"]["result"]
            self.assertFalse(rendered_result.get("success"))
            self.assertIn("not found", str(rendered_result.get("error", "")).lower())

            # The model itself must also receive the truthful, normalised
            # result -- not just the human-facing render.
            tool_messages = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
            self.assertEqual(len(tool_messages), 1)
            sent_to_model = json.loads(tool_messages[0]["content"])
            self.assertFalse(sent_to_model.get("success"))

    def test_empty_provider_continuation_after_a_tool_round_is_recovered_not_treated_as_done(self):
        """Confirmed live: some providers occasionally return a completion
        with neither content nor tool_calls right after consuming tool
        results. That is a provider continuation failure, not a completed
        task -- _recover_empty_continuation must retry the same route before
        the turn is allowed to finish."""
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "existing.txt").write_text("hello\n")
            read_args = json.dumps({"path": str(Path(ws) / "existing.txt")})
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="read_file", arguments=read_args)]))],
                [_chunk(_delta(content=None, tool_calls=None))],  # empty post-tool continuation
                [_chunk(_delta(content="The file says hello."))],  # recovered on retry
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "read the file"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="never", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.summary, "The file says hello.")
            diagnostics = [e["payload"].get("content", "") for e in renderer.events if e["event_type"] == "diagnostics"]
            self.assertTrue(any("empty continuation" in d for d in diagnostics))
            # This successful recovery must be tracked as a real repair
            # attempt (see engine.py's mark_repair/_advance_plan_step) --
            # before this, it was invisible to repair_attempts/AgentPhase.REPAIR.
            self.assertTrue(any(e["event_type"] == "orchestrator_repair" for e in renderer.events))

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
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "loop forever"}],
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
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "check repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False, max_rounds=50,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("stuck repeating", outcome.error)
            # Live-reported: this fires most often on a broad, unscoped
            # request ("audit the entire system") -- the failure should
            # actively suggest narrowing it, not just report the guard
            # tripped with no next step for the user.
            self.assertIn("narrowing", outcome.error)
            # 3 identical rounds is enough to trip the guard -- nowhere near
            # the 50-round cap.
            self.assertLess(len(client.calls), 10)

    def test_huge_single_objective_is_compacted_instead_of_failing_round_one(self):
        """Confirmed live: a large pasted objective (a long log/diff as the
        request) could blow the token budget on round 1, before any tool or
        assistant message exists yet -- compaction used to never touch
        role=="user" content at all, and the rollover gate required prior
        tool history that round 1 doesn't have yet either, so the turn
        failed immediately with a context-window error despite all that
        machinery existing. This must now be compacted and actually
        proceed, not "cut short"."""
        with tempfile.TemporaryDirectory() as ws:
            huge_objective = "read the incident log below and summarize it:\n" + ("x" * 300_000)
            rounds = [[_chunk(_delta(content="Summary of the incident."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": huge_objective}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(len(client.calls), 1)
            sent_user_content = next(
                m["content"] for m in client.calls[0]["messages"] if m.get("role") == "user"
            )
            self.assertLess(len(sent_user_content), len(huge_objective))
            diagnostics = [e["payload"].get("content", "") for e in renderer.events if e["event_type"] == "diagnostics"]
            self.assertTrue(any("Context compacted" in d for d in diagnostics))

    def test_context_budget_exceeded_stops_before_request(self):
        with tempfile.TemporaryDirectory() as ws:
            # A provider config with a tiny context window: even the system
            # prompt + one user message should already exceed it, so this
            # must abort before ever calling the client -- never guaranteed
            # to blow up with a real 400 from the provider.
            client = _FakeClient(rounds=[])
            manager = _FakeManager(client)
            manager.PROVIDERS[ProviderType.NVIDIA].context_window = 10
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "a" * 2000}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("context window", outcome.error)
            self.assertEqual(len(client.calls), 0)
            # An impossibly tiny window (budget stays negative even after
            # compaction) must still genuinely fail -- but it must have
            # actually TRIED an internal context rollover first, not skipped
            # it just because round 1 has no prior tool/assistant history
            # yet (that gate used to block rollover from ever being
            # reachable in exactly this round-1 scenario).
            event_types = [e["event_type"] for e in renderer.events]
            self.assertIn("context_rollover", event_types)

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
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "write a file"}],
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
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "hi"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))
        self.assertEqual(outcome.status, "failed")
        self.assertIn("not available", outcome.error)


class SwarmToolSchemaGatingTests(_StatePatchMixin, unittest.TestCase):
    """delegate_parallel_tasks (SWARM_TOOL_SCHEMA) must only ever be
    offered to the model at a real top-level call (allow_swarm_tool=True),
    never read-only, and never when subagent delegation itself is
    disabled -- these are the three independent gates that keep a
    delegated sub-task's own turn from recursively offering it again."""

    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def _tools_offered(self, client) -> set:
        names = set()
        for call in client.calls:
            for tool in call.get("tools") or []:
                names.add(tool.get("function", {}).get("name"))
        return names

    def test_offered_when_allowed_and_enabled_and_not_read_only(self):
        from tamfis_code.config import Config

        cfg = Config()
        cfg.enable_subagent_delegation = True
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "existing.txt").write_text("hello\n")
            rounds = [[_chunk(_delta(content="Two unrelated things, I'll answer directly."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "look into two separate issues"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                cli_config=cfg, allow_swarm_tool=True,
            ))
        self.assertIn("delegate_parallel_tasks", self._tools_offered(client))

    def test_not_offered_without_allow_swarm_tool(self):
        from tamfis_code.config import Config

        cfg = Config()
        cfg.enable_subagent_delegation = True
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[_chunk(_delta(content="Answering directly."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "look into two separate issues"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                cli_config=cfg,
            ))
        self.assertNotIn("delegate_parallel_tasks", self._tools_offered(client))

    def test_not_offered_when_delegation_disabled(self):
        from tamfis_code.config import Config

        cfg = Config()
        cfg.enable_subagent_delegation = False
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[_chunk(_delta(content="Answering directly."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "look into two separate issues"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                cli_config=cfg, allow_swarm_tool=True,
            ))
        self.assertNotIn("delegate_parallel_tasks", self._tools_offered(client))

    def test_not_offered_in_read_only_mode_even_if_allowed(self):
        from tamfis_code.config import Config

        cfg = Config()
        cfg.enable_subagent_delegation = True
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[_chunk(_delta(content="Answering directly."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "look into two separate issues"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                read_only=True, cli_config=cfg, allow_swarm_tool=True,
            ))
        self.assertNotIn("delegate_parallel_tasks", self._tools_offered(client))

    def test_delegated_subagent_turn_never_offers_swarm_tool(self):
        """DelegatedCodingAgent.execute() (agents.py) never passes
        allow_swarm_tool -- this is the structural depth-1 recursion cap,
        confirmed here by driving run_local_agent_turn exactly the way it
        does (interactive=False, no allow_swarm_tool kwarg at all)."""
        from tamfis_code.config import Config

        cfg = Config()
        cfg.enable_subagent_delegation = True
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[_chunk(_delta(content="Answering directly."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "sub-task objective"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="ask", interactive=False,
            ))
        self.assertNotIn("delegate_parallel_tasks", self._tools_offered(client))


class SwarmToolDispatchTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def test_calling_delegate_parallel_tasks_invokes_run_swarm_and_appends_a_tool_result(self):
        from tamfis_code.config import Config
        from unittest.mock import patch

        cfg = Config()
        cfg.enable_subagent_delegation = True
        swarm_args = json.dumps({"tasks": ["look at a", "look at b"], "mutate": False})

        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="delegate_parallel_tasks", arguments=swarm_args)]))],
                [_chunk(_delta(content="Both sub-tasks are done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            fake_swarm_results = [
                {"task_id": "t1", "description": "look at a", "status": "completed", "result": {"summary": "found nothing"}},
                {"task_id": "t2", "description": "look at b", "status": "completed", "result": {"summary": "found something"}},
            ]

            async def fake_run_swarm(tasks, **kwargs):
                return fake_swarm_results

            with patch("tamfis_code.swarm.run_swarm", new=fake_run_swarm):
                outcome = asyncio.run(run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "look into two separate issues"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                    cli_config=cfg, allow_swarm_tool=True,
                ))

        self.assertEqual(outcome.status, "completed")
        tool_call_events = [e for e in renderer.events if e["event_type"] == "tool_call_requested"]
        self.assertTrue(any(e["payload"]["name"] == "delegate_parallel_tasks" for e in tool_call_events))
        tool_output_events = [e for e in renderer.events if e["event_type"] == "tool_output" and e["payload"]["tool"] == "delegate_parallel_tasks"]
        self.assertEqual(len(tool_output_events), 1)

    def test_fewer_than_two_tasks_is_rejected_without_calling_run_swarm(self):
        from tamfis_code.config import Config
        from unittest.mock import patch

        cfg = Config()
        cfg.enable_subagent_delegation = True
        swarm_args = json.dumps({"tasks": ["only one task"]})

        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="delegate_parallel_tasks", arguments=swarm_args)]))],
                [_chunk(_delta(content="Okay, doing it myself."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            with patch("tamfis_code.swarm.run_swarm") as fake_run_swarm:
                outcome = asyncio.run(run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "look into two separate issues"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                    cli_config=cfg, allow_swarm_tool=True,
                ))

        self.assertEqual(outcome.status, "completed")
        fake_run_swarm.assert_not_called()

    def test_mutation_gate_refusal_from_run_swarm_becomes_a_tool_error_not_a_crash(self):
        from tamfis_code.config import Config
        from unittest.mock import patch

        cfg = Config()
        cfg.enable_subagent_delegation = True
        swarm_args = json.dumps({"tasks": ["fix a", "fix b"], "mutate": True})

        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="delegate_parallel_tasks", arguments=swarm_args)]))],
                [_chunk(_delta(content="I could not run that as a mutating swarm under this policy."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            async def fake_run_swarm(tasks, **kwargs):
                raise ValueError("Swarm sub-tasks run non-interactively and cannot prompt for approval")

            with patch("tamfis_code.swarm.run_swarm", new=fake_run_swarm):
                outcome = asyncio.run(run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None, [{"role": "user", "content": "look into two separate issues"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="ask", interactive=False,
                    cli_config=cfg, allow_swarm_tool=True,
                ))

        self.assertEqual(outcome.status, "completed")
        tool_output_events = [e for e in renderer.events if e["event_type"] == "tool_output" and e["payload"]["tool"] == "delegate_parallel_tasks"]
        self.assertEqual(len(tool_output_events), 1)
        self.assertFalse(tool_output_events[0]["payload"]["result"].get("success", True))


if __name__ == "__main__":
    unittest.main()


class _HTTP402Error(Exception):
    status_code = 402


class _AlwaysFailClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        raise _HTTP402Error("Insufficient credits")


class _FallbackManager:
    def __init__(self, failing, fallback):
        self.clients = {
            ProviderType.OPENROUTER: failing,
            ProviderType.NVIDIA: fallback,
        }
        self.PROVIDERS = {
            ProviderType.OPENROUTER: SimpleNamespace(
                default_model="google/gemini-2.5-flash", context_window=128000,
                tool_calling=True,
            ),
            ProviderType.NVIDIA: SimpleNamespace(
                default_model="llama3.2:3b", context_window=8192,
                tool_calling=False,
            ),
        }

    def resolve_route(self, provider, task_profile=None, quality_mode="quality"):
        return ProviderType.OPENROUTER, self.PROVIDERS[ProviderType.OPENROUTER]

    def get_client(self, provider):
        return self.clients.get(provider)

    @staticmethod
    def is_retryable_provider_error(exc):
        return getattr(exc, "status_code", None) == 402

    @staticmethod
    def provider_error_status(exc):
        return getattr(exc, "status_code", None)

    def fallback_candidates(self, current, task_profile=None):
        return [ProviderType.NVIDIA]


class ProviderFallbackTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def test_openrouter_402_falls_back_to_nvidia_in_auto_mode(self):
        with tempfile.TemporaryDirectory() as ws:
            fallback = _FakeClient([[ _chunk(_delta(content="Hello from fallback")) ]])
            manager = _FallbackManager(_AlwaysFailClient(), fallback)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.AUTO, None,
                [{"role": "user", "content": "hello"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1,
                approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.summary, "Hello from fallback")
            diagnostics = [
                e["payload"].get("content", "") for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(any("HTTP 402" in item and "falling back to nvidia" in item for item in diagnostics))

    def test_interrupted_stream_reconnects_and_continues_without_duplication(self):
        class BrokenStream:
            def __aiter__(self):
                return self._gen()

            async def _gen(self):
                yield _chunk(_delta(content="First clean half. "))
                raise ConnectionError("stream disconnected during service restart")

        class ReconnectingClient:
            def __init__(self):
                self.calls = []
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            async def _create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return BrokenStream()
                return _FakeStream([_chunk(_delta(content="Second clean half."), finish_reason="stop")])

        client = ReconnectingClient()

        class ReconnectManager:
            PROVIDERS = {
                ProviderType.NVIDIA: SimpleNamespace(
                    default_model="moonshotai/kimi-k2.6", context_window=128000,
                    tool_calling=True,
                ),
            }

            def resolve_route(self, provider, task_profile=None, quality_mode="quality"):
                return ProviderType.NVIDIA, self.PROVIDERS[ProviderType.NVIDIA]

            def get_client(self, provider):
                return client

            @staticmethod
            def is_retryable_provider_error(exc):
                return "stream disconnected" in str(exc)

            @staticmethod
            def provider_error_status(exc):
                return None

            @staticmethod
            def fallback_candidates(current, task_profile=None):
                return []

        with tempfile.TemporaryDirectory() as ws:
            renderer = _RecordingRenderer()
            from unittest.mock import patch
            with patch("tamfis_code.runner_local.STREAM_RECONNECT_BACKOFF_SECONDS", (0.0,)):
                outcome = asyncio.run(run_local_agent_turn(
                    ReconnectManager(), ProviderType.AUTO, None,
                    [{"role": "user", "content": "hello"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=4,
                    approval_policy="auto", interactive=False,
                ))

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.summary, "First clean half. Second clean half.")
        visible = "".join(
            event["payload"].get("content", "") for event in renderer.events
            if event["event_type"] == "assistant_delta"
        )
        self.assertEqual(visible, outcome.summary)
        retry_messages = client.calls[1]["messages"]
        self.assertTrue(any(
            message.get("role") == "assistant" and message.get("content") == "First clean half. "
            for message in retry_messages
        ))
        diagnostics = [
            event["payload"].get("content", "") for event in renderer.events
            if event["event_type"] == "diagnostics"
        ]
        self.assertTrue(any("keeping this task alive" in item for item in diagnostics))

    def test_no_output_disconnect_reports_one_error_only_after_backoff_budget(self):
        class AlwaysDisconnectClient:
            def __init__(self):
                self.calls = 0
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            async def _create(self, **kwargs):
                self.calls += 1
                raise ConnectionError("stream disconnected before first token")

        client = AlwaysDisconnectClient()

        class DisconnectManager:
            PROVIDERS = {
                ProviderType.NVIDIA: SimpleNamespace(
                    default_model="moonshotai/kimi-k2.6", context_window=128000,
                    tool_calling=True,
                ),
            }

            def get_client(self, provider):
                return client

            @staticmethod
            def is_retryable_provider_error(exc):
                return "stream disconnected" in str(exc)

            @staticmethod
            def provider_error_status(exc):
                return None

        with tempfile.TemporaryDirectory() as ws:
            renderer = _RecordingRenderer()
            from unittest.mock import patch
            with patch("tamfis_code.runner_local.STREAM_RECONNECT_BACKOFF_SECONDS", (0.0, 0.0)):
                outcome = asyncio.run(run_local_agent_turn(
                    DisconnectManager(), ProviderType.NVIDIA, None,
                    [{"role": "user", "content": "hello"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=5,
                    approval_policy="auto", interactive=False,
                ))

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(client.calls, 3)
        self.assertEqual(
            len([event for event in renderer.events if event["event_type"] == "ai_task_failed"]),
            1,
        )
        self.assertEqual(
            len([
                event for event in renderer.events
                if event["event_type"] == "diagnostics"
                and "keeping this task alive" in event["payload"].get("content", "")
            ]),
            2,
        )

    def test_successful_fallback_is_tracked_as_a_real_repair_attempt(self):
        # Before this, mark_repair() (and AgentPhase.REPAIR) only ever
        # fired once, immediately before giving up entirely -- a fallback
        # that SUCCEEDED was invisible to repair_attempts/AgentPhase.REPAIR,
        # even though it's exactly the kind of real recovery "repair" is
        # supposed to represent.
        with tempfile.TemporaryDirectory() as ws:
            fallback = _FakeClient([[_chunk(_delta(content="Hello from fallback"))]])
            manager = _FallbackManager(_AlwaysFailClient(), fallback)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.AUTO, None,
                [{"role": "user", "content": "hello"}],
                self._console(), renderer,
                workspace_root=ws, session_id=2,
                approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertTrue(any(e["event_type"] == "orchestrator_repair" for e in renderer.events))

    def test_corrupted_external_stream_is_hidden_and_retried_on_clean_external_route(self):
        bad_chunks = [
            _chunk(_delta(content=(
                f"Mist{i}Dotblankativityurpunite "
                f"XCT{i}urpMargueriteurptenhamftyabbDowurpennessfwinistrege "
                f"Logo{i}WebkiturpWonderBVPrompturp "
            )))
            for i in range(40)
        ]
        failing = _FakeClient([bad_chunks])
        fallback = _FakeClient([[_chunk(_delta(content="Clean answer from OpenRouter."))]])

        class QualityFallbackManager:
            clients = {ProviderType.NVIDIA: failing, ProviderType.OPENROUTER: fallback}
            PROVIDERS = {
                ProviderType.NVIDIA: SimpleNamespace(
                    default_model="moonshotai/kimi-k2.6", context_window=128000,
                    tool_calling=True,
                ),
                ProviderType.OPENROUTER: SimpleNamespace(
                    default_model="qwen/qwen3-coder", context_window=128000,
                    tool_calling=True,
                ),
            }

            def resolve_route(self, provider, task_profile=None, quality_mode="quality"):
                return ProviderType.NVIDIA, self.PROVIDERS[ProviderType.NVIDIA]

            def get_client(self, provider):
                return self.clients.get(provider)

            def fallback_candidates(self, current, task_profile=None):
                return [ProviderType.OPENROUTER]

        with tempfile.TemporaryDirectory() as ws:
            renderer = _RecordingRenderer()
            outcome = asyncio.run(run_local_agent_turn(
                QualityFallbackManager(), ProviderType.AUTO, None,
                [{"role": "user", "content": "summarize the incident"}],
                self._console(), renderer,
                workspace_root=ws, session_id=3,
                approval_policy="auto", interactive=False,
            ))

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.summary, "Clean answer from OpenRouter.")
        visible = "".join(
            e["payload"].get("content", "") for e in renderer.events
            if e["event_type"] == "assistant_delta"
        )
        self.assertEqual(visible, "Clean answer from OpenRouter.")
        selected = [
            e["payload"].get("provider") for e in renderer.events
            if e["event_type"] == "model_selected"
        ]
        self.assertIn("openrouter", selected)


class StandaloneBoundaryRegressionTests(unittest.TestCase):
    def test_fallback_chain_never_mentions_tier_iv_for_legacy_manager_double(self):
        from tamfis_code.runner_local import _standalone_fallback_chain_names

        class LegacyManager:
            pass

        chain = _standalone_fallback_chain_names(LegacyManager(), ProviderType.NVIDIA)
        self.assertNotIn("tier_iv", chain)
        self.assertEqual(chain, ["hf", "openrouter"])
