import pytest

from tamfis_code.provider_protocols import ProviderStreamError, normalize_stream_chunk


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
