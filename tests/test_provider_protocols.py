import json

import pytest

from tamfis_code.provider_protocols import (
    ProviderStreamError,
    normalize_stream_chunk,
    normalize_tool_call,
    provider_requires_single_tool_call,
    single_tool_call_messages,
    system_messages_first,
)


def test_system_messages_are_merged_to_one_without_breaking_tool_transcript():
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "system", "content": "repair after evidence"},
        {"role": "assistant", "content": "partial answer"},
        {"role": "system", "content": "stream reconnect"},
    ]
    normalized = system_messages_first(messages)
    assert normalized[0] == {"role": "system", "content": "base\n\nrepair after evidence\n\nstream reconnect"}
    assert [item["role"] for item in normalized[1:]] == ["user", "assistant", "tool", "assistant"]
    assert normalized[2]["tool_calls"][0]["id"] == "call_1"
    assert normalized[3]["tool_call_id"] == "call_1"


def test_channel_markup_is_removed_from_a_registered_tool_name():
    name, arguments = normalize_tool_call(
        "search_code<|Channel|>Commentary({\"query\":\"gemma4\"})",
        "",
        allowed_names={"search_code", "read_file"},
    )
    assert name == "search_code"
    assert json.loads(arguments) == {"query": "gemma4"}


def test_header_correction_prefers_the_final_registered_tool_name():
    name, arguments = normalize_tool_call(
        "to=functions.read_file <|constrain|>write_file? Actually read_file.<|end|>",
        "",
        allowed_names={"read_file", "write_file"},
    )
    assert name == "read_file"


def test_unknown_channel_markup_is_not_authorized_as_a_tool():
    name, arguments = normalize_tool_call(
        "unknown_tool<|Channel|>Commentary({\"x\":1})",
        "",
        allowed_names={"search_code"},
    )
    assert name.startswith("unknown_tool")
    assert arguments == ""


def test_malformed_provider_header_is_retryable():
    from tamfis_code.providers import ProviderManager

    assert ProviderManager.is_retryable_provider_error(
        RuntimeError("unexpected tokens remaining in message header: to=functions.read_file")
    )


def test_single_tool_call_provider_error_is_detected_narrowly():
    assert provider_requires_single_tool_call("This model only supports single tool-calls at once!")
    assert not provider_requires_single_tool_call("invalid API key")


def test_multi_tool_history_is_split_with_matching_results():
    calls = [
        {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
    ]
    result = single_tool_call_messages([
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
    ])
    assert [len(message["tool_calls"]) for message in result if message.get("role") == "assistant"] == [1, 1]
    assert [message["tool_call_id"] for message in result if message.get("role") == "tool"] == ["a", "b"]


def test_system_message_with_list_content_is_flattened_to_text():
    normalized = system_messages_first([
        {"role": "system", "content": [{"text": "part one"}, {"text": "part two"}]},
        {"role": "user", "content": "hi"},
    ])
    assert normalized[0] == {"role": "system", "content": "part one\npart two"}


def test_no_system_messages_returns_transcript_unchanged():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert system_messages_first(messages) == messages


def test_malformed_checkpointed_tool_arguments_are_repaired_for_provider():
    messages = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "call-1", "type": "function",
        "function": {"name": "write_file", "arguments": '{"path":"/tmp/example","content":"truncated'},
    }]}]
    normalized = system_messages_first(messages)
    arguments = normalized[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"_tamfis_code_recovered": "malformed historical tool arguments omitted"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"].endswith("truncated")


def test_dictionary_tool_arguments_are_serialized_as_json_object_string():
    messages = [{"role": "assistant", "tool_calls": [{
        "id": "call-1", "type": "function",
        "function": {"name": "read_file", "arguments": {"path": "a.py"}},
    }]}]
    normalized = system_messages_first(messages)
    assert json.loads(normalized[0]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}


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
    events = normalize_stream_chunk({"choices": [{"delta": {"content": '{"name":"execute_command"}'}, "finish_reason": None}]})
    assert [event.event_type.value for event in events] == ["assistant_delta"]


def test_canonical_event_field_preserves_generated_file_payload():
    events = normalize_stream_chunk({"event": "file_generated", "filename": "updated-project.zip", "file_url": "/files/serve/abc", "size_bytes": 42})
    assert [event.event_type.value for event in events] == ["file_generated"]
    assert events[0].payload["filename"] == "updated-project.zip"
    assert events[0].payload["file_url"] == "/files/serve/abc"


def test_openai_tool_delta_strips_channel_marker_before_runner_sees_it():
    events = normalize_stream_chunk({"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "id": "c1",
        "function": {"name": "search_code<|channel|>commentary", "arguments": ""},
    }]}, "finish_reason": None}]})
    assert events[0].payload["name"] == "search_code"


def test_openai_structured_tool_delta_is_normalized():
    events = normalize_stream_chunk({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]}, "finish_reason": None}]})
    assert events[0].event_type.value == "tool_call_delta"
    assert events[0].payload["name"] == "read_file"


def test_reasoning_content_delta_is_normalized_separately_from_content():
    events = normalize_stream_chunk({"choices": [{"delta": {"reasoning_content": "let me think"}, "finish_reason": None}]})
    assert [event.event_type.value for event in events] == ["reasoning_delta"]
    assert events[0].payload["content"] == "let me think"


def test_reasoning_alias_field_is_also_normalized():
    events = normalize_stream_chunk({"choices": [{"delta": {"reasoning": "thinking"}, "finish_reason": None}]})
    assert [event.event_type.value for event in events] == ["reasoning_delta"]


def test_reasoning_and_content_in_separate_deltas_stay_separate():
    reasoning_events = normalize_stream_chunk({"choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]})
    content_events = normalize_stream_chunk({"choices": [{"delta": {"content": "answer"}}]})
    assert [e.event_type.value for e in reasoning_events] == ["reasoning_delta"]
    assert [e.event_type.value for e in content_events] == ["assistant_delta"]


def test_embedded_resource_exhausted_stream_error_is_raised_as_retryable():
    chunk = {"error": {"message": "ResourceExhausted: Worker local total request limit reached (32/32)", "type": "internal_server_error", "code": 500}}
    with pytest.raises(ProviderStreamError) as raised:
        normalize_stream_chunk(chunk, provider="nvidia", model="nvidia/nemotron")
    assert raised.value.retryable is True
    assert raised.value.status_code == 500
    assert raised.value.provider == "nvidia"
    assert "32/32" in str(raised.value)


def test_canonical_error_event_is_not_silently_ignored():
    with pytest.raises(ProviderStreamError) as raised:
        normalize_stream_chunk({"event_type": "error", "payload": {"message": "service unavailable", "status_code": 503}})
    assert raised.value.retryable is True


def test_system_messages_first_removes_empty_assistant_without_tools():
    result = system_messages_first([
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": ""},
    ])
    assert result == [{"role": "user", "content": "continue"}]


def test_system_messages_first_repairs_tool_only_assistant_content():
    result = system_messages_first([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
    ])
    assert result[0]["content"] == "[tool call]"
    assert result[0]["tool_calls"] == [{"id": "call-1"}]


def test_system_messages_first_keeps_valid_tool_arguments_unchanged():
    call = {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"README.md"}'}}
    result = system_messages_first([{"role": "assistant", "content": "", "tool_calls": [call]}])
    assert result[0]["content"] == "[tool call]"
    assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"path":"README.md"}'


def test_system_messages_first_drops_orphaned_tool_results_for_failover_replay():
    result = system_messages_first([
        {"role": "user", "content": "continue"},
        {"role": "tool", "tool_call_id": "chatcmpl-tool-orphan", "content": "stale"},
        {"role": "assistant", "content": "fresh answer"},
    ])
    assert [message["role"] for message in result] == ["user", "assistant"]
    assert all(message.get("tool_call_id") != "chatcmpl-tool-orphan" for message in result)


def test_system_messages_first_preserves_tool_results_with_matching_call():
    result = system_messages_first([
        {"role": "assistant", "content": "[tool call]", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ])
    assert result[1]["tool_call_id"] == "call-1"
