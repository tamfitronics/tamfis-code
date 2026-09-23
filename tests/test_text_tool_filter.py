from tamfis_code.runner_local import _TextToolStreamFilter, _extract_text_json_tool_calls


def test_complete_xml_tool_markup_never_leaks_into_assistant_text():
    stream_filter = _TextToolStreamFilter({"write_file"})
    visible = stream_filter.feed(
        "before\n<tool_call><function=write_file>"
        "<parameter=path>example.py</parameter>"
        "<parameter=content>pass</parameter></tool_call>\nafter"
    )
    trailing, calls = stream_filter.finish()

    assert visible == "before\n\nafter"
    assert trailing == ""
    assert len(calls) == 1
    assert calls[0].name == "write_file"


def test_unoffered_xml_tool_is_quarantined_as_structured_call():
    stream_filter = _TextToolStreamFilter({"read_file"})
    visible = stream_filter.feed(
        "<tool_call><function=write_file>"
        "<parameter=path>x.py</parameter></tool_call>"
    )
    trailing, calls = stream_filter.finish()

    assert visible == ""
    assert trailing == ""
    assert len(calls) == 1
    assert calls[0].name == "write_file"


def test_channel_markup_is_quarantined_even_without_a_closing_marker():
    stream_filter = _TextToolStreamFilter({"read_file"})
    visible = stream_filter.feed(
        "I will inspect this first.\n"
        "<|tool_call>call:write_todos{todos:[{completed:false}]}"
    )
    trailing, calls = stream_filter.finish()

    assert visible == "I will inspect this first.\n"
    assert trailing == ""
    assert calls == []
    assert stream_filter.malformed_protocol is True


def test_channel_markup_split_across_stream_chunks_is_quarantined():
    stream_filter = _TextToolStreamFilter({"read_file"})
    assert stream_filter.feed("<|tool_") == ""
    visible = stream_filter.feed(
        "call>call:write_todos{todos:[]}" 
    )
    trailing, calls = stream_filter.finish()

    assert visible == ""
    assert trailing == ""
    assert calls == []
    assert stream_filter.malformed_protocol is True


def test_plain_json_tool_request_is_quarantined_and_becomes_structured_call():
    cleaned, calls = _extract_text_json_tool_calls(
        'I will inspect the tools now. {"name":"list_tools","parameters":{}}'
    )

    assert cleaned == "I will inspect the tools now. "
    assert len(calls) == 1
    assert calls[0].name == "list_tools"
    assert calls[0].arguments == "{}"


def test_fenced_json_tool_request_is_not_presented_as_assistant_evidence():
    cleaned, calls = _extract_text_json_tool_calls(
        '```json\n{"name":"deploy_model","arguments":{"model":"x"}}\n```'
    )

    assert "deploy_model" not in cleaned
    assert len(calls) == 1
    assert calls[0].name == "deploy_model"
