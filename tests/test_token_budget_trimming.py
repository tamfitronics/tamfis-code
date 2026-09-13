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

from tamfis_code.runner_local import _estimate_tokens, _trim_tool_outputs


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


if __name__ == "__main__":
    unittest.main()
