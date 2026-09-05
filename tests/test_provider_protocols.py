import pytest

from tamfis_code.provider_protocols import (
    ProviderStreamError,
    normalize_stream_chunk,
    system_messages_first,
)


def test_system_messages_are_merged_to_one_without_breaking_tool_transcript():
    # FIX (2026-09-05, live-confirmed): hoisting-only (keeping N separate
    # system messages, just reordered to the front) was not enough --
    # a backend still rejected the request with the same "System message
    # must be at the beginning" error, meaning its real constraint is
    # "exactly one system message, and it's first". Collapsing to one
    # combined message satisfies both readings.
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "system", "content": "repair after evidence"},
        {"role": "assistant", "content": "partial answer"},
        {"role": "system", "content": "stream reconnect"},
    ]

    normalized = system_messages_first(messages)

    assert normalized[0] == {
        "role": "system", "content": "base\n\nrepair after evidence\n\nstream reconnect",
    }
    assert [item["role"] for item in normalized[1:]] == [
        "user", "assistant", "tool", "assistant",
    ]
    assert normalized[2]["tool_calls"][0]["id"] == "call_1"
    assert normalized[3]["tool_call_id"] == "call_1"


def test_system_message_with_list_content_is_flattened_to_text():
    messages = [
        {"role": "system", "content": [{"text": "part one"}, {"text": "part two"}]},
        {"role": "user", "content": "hi"},
    ]
    normalized = system_messages_first(messages)
    assert normalized[0] == {"role": "system", "content": "part one\npart two"}


def test_no_system_messages_returns_transcript_unchanged():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert system_messages_first(messages) == messages


def test_blank_system_messages_do_not_produce_an_empty_leading_message():
    messages = [{"role": "system", "content": "   "}, {"role": "user", "content": "hi"}]
    assert system_messages_first(messages) == [{"role": "user", "content": "hi"}]


def test_normalizes_ollama_native_text_and_done():
    events = normalize_stream_chunk({"message": {"content": "Hello"}, "done": True, "done_reason": "stop"})
    assert [event.event_type.value for event in events] == ["assistant_delta", "done"]


def test_normalizes_anthropic_text_delta():
    events = normalize_stream_chunk({"type": "content_block_delta", "delta": {"text": "Hi"}})
    assert events[0].payload["content"] == "Hi"


def test_json_looking_assistant_text_is_not_tool_call():
    events = normalize_stream_chunk({
        "choices": [{"delta": {"content": '{"name":"execute_command"}'}, "finish_reason": None}]
    })
    assert [event.event_type.value for event in events] == ["assistant_delta"]


def test_canonical_event_field_preserves_generated_file_payload():
    events = normalize_stream_chunk({
        "event": "file_generated",
        "filename": "updated-project.zip",
        "file_url": "/files/serve/abc",
        "size_bytes": 42,
    })
    assert [event.event_type.value for event in events] == ["file_generated"]
    assert events[0].payload["filename"] == "updated-project.zip"
    assert events[0].payload["file_url"] == "/files/serve/abc"


def test_openai_structured_tool_delta_is_normalized():
    events = normalize_stream_chunk({
        "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]}, "finish_reason": None}]
    })
    assert events[0].event_type.value == "tool_call_delta"
    assert events[0].payload["name"] == "read_file"


def test_reasoning_content_delta_is_normalized_separately_from_content():
    # Confirmed live against NVIDIA NIM (nemotron-3-super with
    # reasoning_effort set): reasoning_content streams ahead of content in
    # separate deltas, not mixed into the same one.
    events = normalize_stream_chunk({
        "choices": [{"delta": {"reasoning_content": "let me think"}, "finish_reason": None}]
    })
    assert [event.event_type.value for event in events] == ["reasoning_delta"]
    assert events[0].payload["content"] == "let me think"


def test_reasoning_alias_field_is_also_normalized():
    events = normalize_stream_chunk({
        "choices": [{"delta": {"reasoning": "thinking"}, "finish_reason": None}]
    })
    assert [event.event_type.value for event in events] == ["reasoning_delta"]


def test_reasoning_and_content_in_separate_deltas_stay_separate():
    reasoning_events = normalize_stream_chunk({"choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]})
    content_events = normalize_stream_chunk({"choices": [{"delta": {"content": "answer"}}]})
    assert [e.event_type.value for e in reasoning_events] == ["reasoning_delta"]
    assert [e.event_type.value for e in content_events] == ["assistant_delta"]


def test_embedded_resource_exhausted_stream_error_is_raised_as_retryable():
    chunk = {
        "error": {
            "message": "ResourceExhausted: Worker local total request limit reached (32/32)",
            "type": "internal_server_error",
            "code": 500,
        }
    }
    with pytest.raises(ProviderStreamError) as raised:
        normalize_stream_chunk(chunk, provider="nvidia", model="nvidia/nemotron")

    assert raised.value.retryable is True
    assert raised.value.status_code == 500
    assert raised.value.provider == "nvidia"
    assert "32/32" in str(raised.value)


def test_canonical_error_event_is_not_silently_ignored():
    with pytest.raises(ProviderStreamError) as raised:
        normalize_stream_chunk({
            "event_type": "error",
            "payload": {
                "message": "service unavailable",
                "status_code": 503,
            },
        })
    assert raised.value.retryable is True
