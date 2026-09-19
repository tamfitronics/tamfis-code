"""Multi-stage context compression (Pillar 1) -- context invincibility.

The headline test is the one the design spec demands: a 200k-token
conversation must still be able to answer a question about a variable defined
in its very first message, after everything around that message was compressed
away, purely because the structured State-of-the-Union layer carried it
forward.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tamfis_code.orchestrator.compression import (
    CACHE_BOUNDARY_MARKER,
    CacheBoundary,
    CompressionCascade,
    CompressionReport,
    build_state_of_union,
    extract_facts,
    extract_signatures,
    micro_compact,
    render_state_of_union,
    signature_view,
)


def _filler(tokens: int) -> str:
    return ("lorem ipsum dolor sit amet consectetur adipiscing elit " * ((tokens * 4) // 57))[: tokens * 4]


def _tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant_read(path: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
        }],
    }


# --------------------------------------------------------------------------
# Stage 1 -- micro truncation
# --------------------------------------------------------------------------


def test_micro_stage_bounds_a_large_tool_output_and_keeps_a_pointer():
    huge = "A" * 40_000
    compacted, changed = micro_compact(
        huge, pointer="read_file(src/big.py)", label="tool output",
    )
    assert changed is True
    assert len(compacted) < len(huge) / 5
    assert "read_file(src/big.py)" in compacted
    assert compacted.startswith("A" * 100)
    assert compacted.rstrip().endswith("A" * 100)


def test_micro_stage_leaves_small_payloads_untouched():
    text = "a short tool result"
    compacted, changed = micro_compact(text, label="tool output")
    assert changed is False
    assert compacted == text


def test_cascade_stage1_compacts_tool_outputs_but_not_the_current_request():
    messages = [
        {"role": "system", "content": "system rules"},
        _assistant_read("/ws/src/app.py", "call_1"),
        _tool_message("call_1", _filler(3_000)),
        {"role": "user", "content": "please fix the bug in app.py"},
    ]
    report = CompressionCascade().compact(messages, token_budget=100_000)
    assert report.stage1_compactions >= 1
    assert "read_file(/ws/src/app.py)" in messages[2]["content"]
    assert messages[-1]["content"] == "please fix the bug in app.py"
    assert messages[0]["content"] == "system rules"


# --------------------------------------------------------------------------
# Stage 3 -- elastic signature view
# --------------------------------------------------------------------------


def test_signature_view_keeps_declarations_and_docstrings_not_bodies(tmp_path: Path):
    source = tmp_path / "service.py"
    source.write_text(
        '"""Service module."""\n'
        "\n"
        "SECRET_PAYLOAD = b'" + "x" * 5000 + "'\n"
        "\n"
        "def compute_total(items):\n"
        '    """Add up item prices."""\n'
        "    return sum(item.price for item in items)\n"
        "\n"
        "class Reconciler:\n"
        '    """Reconciles two workbooks."""\n'
        "    def reconcile(self, left, right):\n"
        "        return left == right\n"
    )
    view = extract_signatures(source.read_text())
    assert "def compute_total(items):" in view
    assert "Add up item prices." in view
    assert "class Reconciler:" in view
    assert "SECRET_PAYLOAD" not in view
    assert "sum(item.price for item in items)" not in view


def test_stage3_prunes_superseded_file_reads_to_a_signature_view(tmp_path: Path):
    source = tmp_path / "big_module.py"
    source.write_text("def alpha(a, b):\n" + "    x = a + b\n" * 500 + "\n\ndef beta():\n    return 2\n")

    other = tmp_path / "other.py"
    other.write_text("def gamma():\n    return 1\n")
    messages = [
        {"role": "system", "content": "rules"},
        _assistant_read(str(source), "call_read"),
        _tool_message("call_read", "def alpha(a, b):\n" + "    x = a + b\n" * 500),
        # Two later reads make the first one superseded (keep_recent=2).
        _assistant_read(str(other), "call_read_2"),
        _tool_message("call_read_2", other.read_text()),
        _assistant_read(str(other), "call_read_3"),
        _tool_message("call_read_3", other.read_text()),
    ]

    report = CompressionReport()
    CompressionCascade(signature_keep_recent=2).stage3_elastic(messages, report)
    assert report.stage3_signature_prunes == 1
    pruned = messages[2]["content"]
    assert "<body elided>" in pruned
    assert "def alpha(a, b):" in pruned
    assert f"re-read {source}" in pruned
    # The most recent reads keep their full bodies: the layer prunes what the
    # model has already moved past, never what it just looked at.
    assert "<body elided>" not in messages[-1]["content"]
    assert "<body elided>" not in messages[-3]["content"]


def test_stage3_is_skipped_when_the_context_already_fits():
    """Every layer is budget-driven: a small conversation must not pay the
    cost (or lose the detail) of layers it does not need."""
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "quick question"},
        _tool_message("call_1", "small result"),
    ]
    report = CompressionCascade().compact(messages, token_budget=100_000)
    assert report.stage1_compactions == 0
    assert report.stage2_summary_injected is False
    assert report.stage3_signature_prunes == 0
    assert report.tokens_after == report.tokens_before


def test_signature_view_is_none_for_a_missing_file(tmp_path: Path):
    assert signature_view(tmp_path / "nope.py") is None


# --------------------------------------------------------------------------
# Stage 2 -- structured State of the Union
# --------------------------------------------------------------------------


def _conversation_of_about(tokens: int, *, first_message: str, turns: int = 100) -> list[dict]:
    per_turn = max(200, tokens * 4 // turns)
    messages: list[dict] = [
        {"role": "system", "content": "system rules for the agent"},
        {"role": "user", "content": first_message},
    ]
    for index in range(turns):
        messages.append({"role": "assistant", "content": f"step {index}: " + ("y" * per_turn)})
        messages.append(
            _tool_message(f"call_{index}", f"result {index}: " + ("z" * per_turn)),
        )
    return messages


def test_200k_token_conversation_still_remembers_a_variable_from_100_messages_ago():
    """The mandated test: simulate a 200k-token conversation and verify the
    agent can still 'remember' state defined in the very first message, via
    the structured summary layer rather than the (now-evicted) original text.
    """
    first_message = (
        "Investigate the flaky inference test. Note: retry_backoff_seconds = 45 is the "
        "documented backoff and must not be changed. TODO: confirm the retry budget."
    )
    messages = _conversation_of_about(200_000, first_message=first_message)
    cascade = CompressionCascade()
    from tamfis_code.orchestrator.compression import estimate_tokens

    budget = 200_000
    assert estimate_tokens("\n".join(str(m.get("content") or "") for m in messages)) >= 150_000

    report = cascade.compact(messages, token_budget=budget, target_tokens=int(budget * 0.85))

    assert report.stage2_summary_injected is True, report.skipped
    summary = next(
        m for m in messages if isinstance(m, dict) and m.get("_tamfis_compression") == "state_of_union"
    )
    # The variable defined in message #2 is carried forward by the summary...
    assert "retry_backoff_seconds" in summary["content"]
    # ...and the summary sits before the detail it replaces, right after the
    # leading system message.
    assert messages[0]["role"] == "system"
    assert messages[1] is summary
    # The layer is bounded, and compaction actually reduced the context.
    assert report.stage2_summary_tokens <= 20_000
    assert report.tokens_after < report.tokens_before
    # The durable layers that must never be evicted are still present.
    assert messages[0]["content"] == "system rules for the agent"
    assert any(m.get("role") == "user" for m in messages)


def test_state_of_union_records_files_failures_and_todos():
    messages = [
        {"role": "user", "content": "Fix the build. It touches src/app/main.py and tests/test_app.py."},
        {"role": "tool", "content": "Traceback (most recent call last): AssertionError: expected 3 got 4"},
        {"role": "assistant", "content": "TODO: re-run the suite after fixing src/app/main.py"},
    ]
    rendered = render_state_of_union(extract_facts(messages))
    assert "src/app/main.py" in rendered
    assert "Failed attempts" in rendered
    assert "TODO" in rendered
    assert "tests/test_app.py" in rendered


def test_state_of_union_is_empty_for_a_trivial_conversation():
    rendered, facts = build_state_of_union([{"role": "assistant", "content": "hello"}])
    assert rendered == ""
    assert facts.is_empty()


# --------------------------------------------------------------------------
# Cache boundary
# --------------------------------------------------------------------------


def test_cache_boundary_splits_static_instructions_from_volatile_state():
    boundary = CacheBoundary("STATIC TOOLS AND RULES", "volatile plan + fingerprint")
    messages = boundary.as_system_messages()
    assert messages[0] == {"role": "system", "content": "STATIC TOOLS AND RULES"}
    assert messages[1]["content"] == "volatile plan + fingerprint"
    # The cacheable prefix is byte-identical whether or not volatile state is
    # present -- that is the entire point (prefix caching is prefix-based).
    assert messages[0]["content"] == CacheBoundary("STATIC TOOLS AND RULES").as_system_messages()[0]["content"]
    assert CacheBoundary.split(boundary.joined()).static_prefix == "STATIC TOOLS AND RULES"
    assert CACHE_BOUNDARY_MARKER in boundary.joined()


def test_context_bundle_keeps_a_stable_static_prefix_and_puts_volatile_state_second(tmp_path: Path):
    from tamfis_code import state as local_state
    from tamfis_code.orchestrator.context import build_context_bundle
    from tamfis_code.routing import classify_task

    original_config, original_state = local_state.CONFIG_DIR, local_state.STATE_PATH
    local_state.CONFIG_DIR = tmp_path / ".config"
    local_state.STATE_PATH = local_state.CONFIG_DIR / "state.json"
    try:
        local_state.save_session_state(11, workspace_root=str(tmp_path))
        profile = classify_task("Fix the login bug in src/auth.py")
        first = build_context_bundle(
            session_id=11, workspace_root=str(tmp_path), objective="Fix the login bug in src/auth.py",
            profile=profile, conversation_messages=[{"role": "user", "content": "Fix the login bug"}],
        )
        # A second turn with different volatile state (a plan) must not change
        # the cacheable static prefix.
        second = build_context_bundle(
            session_id=11, workspace_root=str(tmp_path), objective="Fix the login bug in src/auth.py",
            profile=profile, conversation_messages=[{"role": "user", "content": "Fix the login bug"}],
            plan={"objective": "Fix the login bug", "steps": ["read src/auth.py"]},
        )
    finally:
        local_state.CONFIG_DIR, local_state.STATE_PATH = original_config, original_state

    assert first.messages[0]["role"] == "system"
    assert second.messages[0]["content"] == first.messages[0]["content"]
    assert first.layers["cache_boundary"]["volatile_chars"] > 0
    assert second.layers["cache_boundary"]["volatile_chars"] > first.layers["cache_boundary"]["volatile_chars"]
