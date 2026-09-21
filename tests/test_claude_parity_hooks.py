"""Integration tests for the two Claude-Code-parity hook events added
alongside the Codex `interrupt_hooks.rs`/`session_interrupted` work:
`user_prompt_submit` (fires before the objective is classified/sent to a
provider; can block the whole turn or add context) and `session_completed`
(a Stop-hook completion boundary that can block and re-enter the same model
loop, symmetric to `session_interrupted` for the failure case). Confirmed live via
hooks.py's own unit tests first (test_hooks.py); these prove the actual
runner_local.py wiring, not just that the hooks.py functions work in
isolation -- the same pattern test_round_budget_extension.py already
established for session_interrupted.
"""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import run_local_agent_turn

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _RecordingRenderer,
    _StatePatchMixin,
    _chunk,
    _delta,
    _tool_call_delta,
)


class UserPromptSubmitAndSessionCompletedHookTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def test_blocking_hook_stops_the_turn_before_any_provider_call(self):
        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[user_prompt_submit]]\ncommand = "echo \\"not allowed\\" 1>&2; exit 2"\n'
            )
            client = _FakeClient([[_chunk(_delta(content="This must never be reached."))]])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "do something risky"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("not allowed", outcome.error or "")
            self.assertEqual(client.calls, [], "provider must never be called once the hook blocked the turn")

    def test_context_adding_hook_reaches_the_provider_and_completion_fires_session_completed(self):
        with tempfile.TemporaryDirectory() as ws:
            marker = Path(ws) / "completed_marker.txt"
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[user_prompt_submit]]\n'
                'command = "echo \\"reminder: mention RFC 7519\\" 1>&2"\n'
                '\n'
                '[[session_completed]]\n'
                f'command = "cat > {marker}"\n'
            )
            # A question-style objective classifies as chat/inspect, not a
            # coding task -- deliberately avoiding a mutating-task objective
            # (e.g. "add authentication"), which would fail unrelated
            # tool-evidence/mutation-verification gates before ever reaching
            # the completion path this test is actually checking.
            client = _FakeClient([[_chunk(_delta(content="JWT stands for JSON Web Token, mentioning RFC 7519 as reminded."))]])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what is JWT authentication"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            # The injected context must have actually reached the provider
            # call, not just been computed and discarded.
            sent_messages = client.calls[0]["messages"]
            sent_text = json.dumps(sent_messages)
            self.assertIn("Additional context from hook: reminder: mention RFC 7519", sent_text)

            self.assertTrue(marker.is_file(), "session_completed hook never ran")
            payload = json.loads(marker.read_text())
            self.assertEqual(payload["event"], "session_completed")
            self.assertEqual(payload["session_id"], 1)
            self.assertIn("JWT stands for JSON Web Token", payload["summary"])

    def test_stop_hook_block_reenters_same_model_loop_before_persisting_completion(self):
        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            marker = Path(ws) / "stop_once.marker"
            guard = Path(ws) / "stop_guard.py"
            guard.write_text(
                "import json, pathlib\n"
                f"marker = pathlib.Path({str(marker)!r})\n"
                "if not marker.exists():\n"
                "    marker.touch()\n"
                "    print(json.dumps({'decision': 'block', 'reason': 'run the required verification'}))\n"
            )
            (hooks_dir / "hooks.toml").write_text(
                f'[[session_completed]]\ncommand = "python3 {guard}"\n'
            )
            client = _FakeClient([
                [_chunk(_delta(content="Draft answer."))],
                [_chunk(_delta(content="Verified final answer."))],
            ])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what is JWT authentication"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(len(client.calls), 2)
            self.assertIn("A Stop hook blocked completion", json.dumps(client.calls[1]["messages"]))
            self.assertEqual(outcome.summary, "Verified final answer.")


class UpdatedInputMutationTests(_StatePatchMixin, unittest.TestCase):
    """Claude-Code-parity addition (updatedInput): a pre_tool_use hook can
    rewrite the pending call's arguments. Proven here against the real
    write_file dispatch path -- the actual file written to disk must
    contain the hook's rewritten content, not the model's original
    request, not just that run_tool_hooks returns the right HookResult in
    isolation."""

    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def test_a_pre_tool_use_hook_rewrites_write_file_content_before_it_is_written(self):
        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[pre_tool_use]]\n'
                'matcher = "write_file"\n'
                'command = "echo \'{\\"updated_input\\": {\\"content\\": \\"sanitized content\\"}}\'"\n'
            )
            target = Path(ws) / "out.txt"
            args = json.dumps({"path": str(target), "content": "raw content from the model"})
            client = _FakeClient([
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=args)]))],
                [_chunk(_delta(content="File written."))],
            ])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "write raw content to out.txt"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_text(), "sanitized content")


def _stub_prompt_hook_endpoint(*, status=200, content="", raises=None):
    """Replace ONLY the prompt hook's HTTP call with a canned reply.

    A subclass of the real httpx.AsyncClient that overrides just `post` (the
    call the hook makes): the model SDK's own clients are httpx clients too and
    must keep working untouched inside the same turn."""
    import httpx
    from unittest.mock import patch as _patch

    class _Response:
        status_code = status

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    real = httpx.AsyncClient

    class _Client(real):
        async def post(self, url, *args, **kwargs):
            if "/v1/chat/completions" in str(url) and "9555" in str(url):
                if raises is not None:
                    raise raises
                return _Response()
            return await super().post(url, *args, **kwargs)

    return _patch("httpx.AsyncClient", _Client)


class PromptBasedHookTests(_StatePatchMixin, unittest.TestCase):
    """Claude-Code-parity addition (`type = "prompt"`): a hook asks an LLM
    to decide the outcome instead of running a shell command. This test
    calls the REAL internal Tier IV endpoint (127.0.0.1:9555) -- not
    mocked -- matching how upgrade_session_title_with_ai's own tests this
    session were verified live against the same endpoint."""

    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def _run_turn_with_a_denying_prompt_hook(self):
        """Run one turn whose only tool call is a shell command a prompt hook is
        asked to judge. Returns (command_ran, execute_command tool outputs)."""
        with tempfile.TemporaryDirectory() as ws:
            marker = Path(ws) / "should_never_exist.txt"
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[pre_tool_use]]\n'
                'matcher = "execute_command"\n'
                'type = "prompt"\n'
                'prompt = "Evaluate if this shell command is safe to run: $TOOL_INPUT. '
                'This command deletes the entire filesystem and must always be denied -- '
                'respond with decision \\"deny\\"."\n'
            )
            command = f"touch {marker}"
            args = json.dumps({"command": command})
            client = _FakeClient([
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="execute_command", arguments=args)]))],
                [_chunk(_delta(content="The command was blocked, as reported by the tool result."))],
            ])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            # The end-of-turn LLM session title would otherwise call the network.
            from unittest.mock import AsyncMock, patch as _patch_title

            with _patch_title("tamfis_code.state.upgrade_session_title_with_ai", new=AsyncMock()):
                outcome = asyncio.run(run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None,
                    [{"role": "user", "content": f"run: {command}"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                ))

            tool_outputs = [
                e["payload"] for e in renderer.events
                if e["event_type"] == "tool_output" and e["payload"].get("tool") == "execute_command"
            ]
            return marker.exists(), tool_outputs

    def _assert_blocked(self, command_ran, tool_outputs):
        # A single blocked tool call doesn't fail the whole turn -- the model
        # still gets to answer using the block as evidence (here, the fake
        # model's scripted round-2 response). What actually matters: the
        # command was never really executed, and the blocking hook's reasoning
        # is visible in the tool result.
        self.assertFalse(command_ran, "the denied command must never actually have run")
        self.assertEqual(len(tool_outputs), 1)
        self.assertFalse(tool_outputs[0]["result"]["success"])
        self.assertIn("Blocked by hook", tool_outputs[0]["result"]["error"])

    def test_a_prompt_hook_denying_a_command_blocks_the_call_before_it_runs(self):
        """Deterministic: the endpoint is stubbed to answer "deny", so this pins
        the runner's contract (a deny blocks the call BEFORE it runs) without
        depending on a live model being reachable or agreeing."""
        with _stub_prompt_hook_endpoint(content='{"decision": "deny", "reason": "wipes the filesystem"}'):
            command_ran, tool_outputs = self._run_turn_with_a_denying_prompt_hook()
        self._assert_blocked(command_ran, tool_outputs)
        self.assertIn("wipes the filesystem", tool_outputs[0]["result"]["error"])

    def test_the_live_tier_iv_endpoint_also_denies_it(self):
        """Integration: calls the REAL internal Tier IV endpoint (127.0.0.1:9555),
        unmocked. Skipped -- explicitly, not silently red -- when that service is
        not answering, because a fail-open "approve" then says nothing about the
        code under test."""
        import httpx

        base = os.environ.get("TAMGPT_TIER_IV_URL", "http://127.0.0.1:9555").rstrip("/")
        try:
            probe = httpx.post(
                f"{base}/v1/chat/completions", timeout=3.0,
                json={"model": "auto", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]},
            )
        except Exception as exc:
            self.skipTest(f"Tier IV endpoint not reachable ({type(exc).__name__})")
        if probe.status_code != 200:
            self.skipTest(f"Tier IV endpoint unhealthy (HTTP {probe.status_code})")
        # The endpoint is a live LLM: asserting that it DENIES is asserting model
        # behaviour, which is what made this test flaky (fail-open on a slow call,
        # or a model that simply approved). Ask it directly first; the end-to-end
        # wiring below is only meaningful -- and only asserted -- when the real
        # endpoint is healthy AND actually returned a deny.
        from tamfis_code.hooks import HookDefinition, _execute_prompt_hook

        decision = asyncio.run(_execute_prompt_hook(
            HookDefinition(
                event="pre_tool_use", matcher="execute_command", command="", source="test",
                hook_type="prompt",
                prompt=(
                    "Evaluate if this shell command is safe to run: $TOOL_INPUT. "
                    "This command deletes the entire filesystem and must always be denied -- "
                    'respond with decision "deny".'
                ),
            ),
            {"tool_input": "rm -rf /"}, blocking_capable=True,
        ))
        if decision is None or not decision.blocked:
            self.skipTest(
                "Tier IV endpoint did not return a deny (fail-open or the model "
                f"disagreed): {getattr(decision, 'message', None)!r}"
            )
        command_ran, tool_outputs = self._run_turn_with_a_denying_prompt_hook()
        if command_ran:
            # The turn asks the LIVE model a SECOND time, and a live model does not answer the same
            # way twice (or fails open when the call is slow / returns non-JSON). "Approved this time"
            # says nothing about the runner -- the deny-blocks-before-it-runs contract is pinned
            # deterministically by test_a_prompt_hook_denying_a_command_blocks_the_call_before_it_runs.
            self.skipTest("the live Tier IV model approved (or failed open) on the second call")
        self._assert_blocked(command_ran, tool_outputs)


class PromptHookDecisionTests(unittest.TestCase):
    """_execute_prompt_hook's decision logic, deterministic and instant."""

    def _hook(self):
        from tamfis_code.hooks import HookDefinition

        return HookDefinition(
            event="pre_tool_use", matcher="execute_command", command="", source="test",
            hook_type="prompt", prompt="Is this safe: $TOOL_INPUT",
        )

    def _decide(self, **stub):
        from tamfis_code.hooks import _execute_prompt_hook

        with _stub_prompt_hook_endpoint(**stub):
            return asyncio.run(_execute_prompt_hook(
                self._hook(), {"tool_input": "rm -rf /"}, blocking_capable=True,
            ))

    def test_deny_blocks_with_the_reason(self):
        result = self._decide(content='{"decision": "deny", "reason": "destructive"}')
        self.assertTrue(result.blocked)
        self.assertEqual(result.message, "destructive")

    def test_approve_does_not_block(self):
        result = self._decide(content='{"decision": "approve", "reason": "read only"}')
        self.assertFalse(result.blocked)

    def test_a_deny_is_ignored_when_the_event_cannot_block(self):
        from tamfis_code.hooks import _execute_prompt_hook

        with _stub_prompt_hook_endpoint(content='{"decision": "deny", "reason": "no"}'):
            result = asyncio.run(_execute_prompt_hook(self._hook(), {}, blocking_capable=False))
        self.assertFalse(result.blocked)

    def test_an_unhealthy_endpoint_fails_open(self):
        result = self._decide(status=503, content="")
        self.assertFalse(result.blocked)
        self.assertIn("fail open", result.message)

    def test_a_malformed_answer_fails_open(self):
        result = self._decide(content="I think this is probably fine!")
        self.assertFalse(result.blocked)
        self.assertIn("fail open", result.message)

    def test_a_timeout_fails_open(self):
        result = self._decide(raises=TimeoutError("timed out"))
        self.assertFalse(result.blocked)
        self.assertIn("fail open", result.message)


class AsyncRewakeTests(_StatePatchMixin, unittest.TestCase):
    """Claude-Code-parity addition (`async_rewake = true`): a hook runs
    detached from the triggering call; its findings are queued as a
    follow-up instruction once it finishes, even though the turn that
    triggered it has already completed."""

    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def test_a_rewake_hook_queues_a_follow_up_after_the_turn_already_completed(self):
        from tamfis_code import state as state_module

        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[post_tool_use]]\n'
                'matcher = "write_file"\n'
                'async_rewake = true\n'
                'command = "sleep 0.3; echo \\"security review complete, no issues\\""\n'
                'rewake_message = "Background review: $FINDINGS"\n'
            )
            target = Path(ws) / "out.txt"
            args = json.dumps({"path": str(target), "content": "hello"})
            client = _FakeClient([
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=args)]))],
                [_chunk(_delta(content="File written."))],
            ])
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            async def run_and_check():
                outcome = await run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None,
                    [{"role": "user", "content": "write hello to out.txt"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                )
                self.assertEqual(outcome.status, "completed")
                # The turn itself has already fully completed by this point
                # (real classification/orchestration overhead means the
                # 0.3s hook may or may not have already finished too --
                # that race isn't what this test is checking). What matters
                # is that the hook's finding eventually surfaces as a
                # queued follow-up, proving the detached background path
                # actually reaches state.py, not that it's strictly slower
                # than the turn that triggered it.
                await asyncio.sleep(0.6)
                queued = state_module.get_session_state(1).queued_user_instructions
                self.assertEqual(len(queued), 1)
                self.assertEqual(queued[0]["classification"], "follow_up")
                self.assertIn("security review complete, no issues", queued[0]["text"])
                from tamfis_code.hooks import drain_pending_rewake_tasks
                await drain_pending_rewake_tasks()

            asyncio.run(run_and_check())


if __name__ == "__main__":
    unittest.main()
