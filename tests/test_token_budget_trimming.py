"""Regression tests for runner_local.py's token-budget accounting and
compaction: _estimate_tokens and _trim_tool_outputs. Closes the Codex
token_budget.rs/token_usage_rollout.rs parity gap -- before this, neither
function had any direct test anywhere in the suite (test_round_budget_
extension.py's MAX_AGENT_ROUND_EXTENSIONS is a *round-count* safety valve,
a distinct concept from token-count accounting against a provider's
context window).

Confirmed live before writing these: _estimate_tokens counts both message
content and any tool_calls[].function.arguments strings, at
_CHARS_PER_TOKEN_ESTIMATE (4) chars per token; _trim_tool_outputs
iteratively shrinks the oldest/tool-heavy messages in escalating passes
until the working list fits target_tokens, while always preserving the
latest user message and the leading system message intact.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from tamfis_code.runner_local import (
    _estimate_tokens,
    _prepare_direct_provider_messages,
    _stream_one_completion,
    _trim_tool_outputs,
)
from tamfis_code.providers import ProviderManager, ProviderType


class EstimateTokensTests(unittest.TestCase):
    def test_counts_message_content(self):
        messages = [{"role": "user", "content": "x" * 40}]
        self.assertEqual(_estimate_tokens(messages), 10)

    def test_counts_tool_call_arguments_too(self):
        messages = [{
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"arguments": "z" * 40}}],
        }]
        self.assertEqual(_estimate_tokens(messages), 10)

    def test_sums_across_every_message(self):
        messages = [
            {"role": "system", "content": "x" * 40},
            {"role": "user", "content": "y" * 20},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": "z" * 40}}]},
        ]
        self.assertEqual(_estimate_tokens(messages), (40 + 20 + 40) // 4)

    def test_empty_messages_estimate_to_zero(self):
        self.assertEqual(_estimate_tokens([]), 0)


def _big_tool_heavy_history(rounds: int = 20, blob_size: int = 5000) -> list[dict]:
    messages = [{"role": "system", "content": "sys"}]
    for i in range(rounds):
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "function": {"name": "read_file", "arguments": "{}"}}],
        })
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "A" * blob_size})
    messages.append({"role": "user", "content": "final question"})
    return messages


class TrimToolOutputsTests(unittest.TestCase):
    def test_already_under_budget_is_a_no_op(self):
        messages = [{"role": "user", "content": "short"}]
        self.assertFalse(_trim_tool_outputs(messages, target_tokens=1000))
        self.assertEqual(messages, [{"role": "user", "content": "short"}])

    def test_shrinks_an_oversized_history_below_the_target(self):
        messages = _big_tool_heavy_history()
        before = _estimate_tokens(messages)
        trimmed = _trim_tool_outputs(messages, target_tokens=500)
        after = _estimate_tokens(messages)
        self.assertTrue(trimmed)
        self.assertGreater(before, 500)
        self.assertLessEqual(after, 500)

    def test_the_latest_user_message_is_never_touched(self):
        messages = _big_tool_heavy_history()
        _trim_tool_outputs(messages, target_tokens=500)
        self.assertEqual(messages[-1], {"role": "user", "content": "final question"})

    def test_the_leading_system_message_is_never_touched(self):
        messages = _big_tool_heavy_history()
        _trim_tool_outputs(messages, target_tokens=500)
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})

    def test_an_impossibly_low_target_still_terminates_without_raising(self):
        messages = _big_tool_heavy_history()
        # Not asserting it reaches the target here -- the leading system
        # message and latest user message are protected in normal passes,
        # so an unreasonably tiny target may not be fully reachable. What
        # matters is that trimming still runs to completion and returns
        # rather than looping forever or raising.
        trimmed = _trim_tool_outputs(messages, target_tokens=1)
        self.assertIsInstance(trimmed, bool)


class DirectProviderBoundaryTests(unittest.TestCase):
    def test_oversized_direct_request_is_bounded_before_provider_call(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "older evidence\n" + "x" * 700_000},
            {"role": "user", "content": "current request"},
        ]
        original = [dict(item) for item in messages]
        bounded, changed, before, after = _prepare_direct_provider_messages(
            messages, provider="tier_iv", model="", tools=[],
        )
        self.assertTrue(changed)
        self.assertGreater(before, after)
        self.assertLessEqual(after, 90_000)
        self.assertEqual(messages, original, "provider protection must not mutate the durable transcript")

    def test_direct_boundary_accounts_for_tool_schema_and_output_reserve(self):
        messages = [{"role": "user", "content": "y" * 420_000}]
        tools = [{"type": "function", "function": {"name": "x", "description": "z" * 40_000}}]
        bounded, changed, _before, after = _prepare_direct_provider_messages(
            messages, provider="tier_iv", model="", tools=tools,
        )
        self.assertTrue(changed)
        self.assertLessEqual(after, 80_000)
        self.assertLess(len(bounded[0]["content"]), len(messages[0]["content"]))

    def test_direct_boundary_does_not_mutate_nested_tool_calls(self):
        messages = [{
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "write_file", "arguments": "a" * 500_000}}],
        }, {"role": "user", "content": "continue"}]
        original_arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
        _prepare_direct_provider_messages(messages, provider="tier_iv", model="", tools=[])
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], original_arguments)

    def test_context_rejection_gets_one_emergency_compaction_retry(self):
        messages = [{"role": "user", "content": "x" * 500_000}]
        calls = []

        async def fake_stream(_client, **kwargs):
            calls.append(kwargs["messages"])
            if len(calls) == 1:
                raise RuntimeError(
                    "Error code: 400 maximum context length: requested 170066 tokens"
                )
            return "ok", [], "stop"

        class Renderer:
            def __init__(self):
                self.events = []

            def handle_event(self, event):
                self.events.append(event)

        renderer = Renderer()
        original = [dict(item) for item in messages]
        with patch("tamfis_code.runner_local._stream_one_completion_impl", fake_stream):
            result = __import__("asyncio").run(
                _stream_one_completion(
                    object(), model="tamfis-gpt-pro", messages=messages, tools=[],
                    renderer=renderer, provider=ProviderType.TAMFIS,
                )
            )

        self.assertEqual(result[0], "ok")
        self.assertEqual(len(calls), 2)
        self.assertLessEqual(_estimate_tokens(calls[1]), _estimate_tokens(calls[0]))
        self.assertEqual(messages, original, "request compaction must not mutate the transcript")
        self.assertTrue(any("retrying the same request safely" in str(event) for event in renderer.events))

    def test_provider_manager_boundary_compacts_and_retries_context_rejection(self):
        """The ProviderManager path is used by local chat and must not bypass
        runner_local's request-size guard."""
        config = ProviderManager.PROVIDERS[ProviderType.TAMFIS]
        calls = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError(
                    "Error code: 400 maximum context length: requested 174162 tokens"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=fake_create)))
        )
        manager = ProviderManager.__new__(ProviderManager)
        manager.runtime_mode = "standalone"
        manager.clients = {ProviderType.TAMFIS: client}
        manager._nim_client_pool = []
        manager._nim_key_index = 0
        manager.resolve_route = lambda *_args, **_kwargs: (ProviderType.TAMFIS, config)
        manager.select_model = lambda *_args, **_kwargs: config.default_model
        manager.normalize_model_for_endpoint = lambda _provider, selected: selected
        manager.record_route_attempt = lambda *_args, **_kwargs: None
        manager.record_route_success = lambda *_args, **_kwargs: None
        manager.record_route_failure = lambda *_args, **_kwargs: None

        messages = _big_tool_heavy_history(rounds=80, blob_size=8_000)
        original_size = _estimate_tokens(messages)
        result = __import__("asyncio").run(
            self._collect_provider(manager, messages)
        )

        self.assertEqual(result, ["ok"])
        self.assertEqual(len(calls), 2)
        self.assertLessEqual(_estimate_tokens(calls[0]["messages"]), 100_000)
        self.assertLess(_estimate_tokens(calls[1]["messages"]), _estimate_tokens(calls[0]["messages"]))
        self.assertEqual(_estimate_tokens(messages), original_size)

    @staticmethod
    async def _collect_provider(manager, messages):
        return [
            chunk async for chunk in manager.chat_completion(
                ProviderType.TAMFIS, messages, stream=False,
            )
        ]


if __name__ == "__main__":
    unittest.main()
