import json

from tamfis_code.runtime import telemetry


def test_provider_span_uses_shared_vocabulary_and_never_records_payload(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", path)
    with telemetry.trace("cli.task", mode="agent") as trace_id:
        with telemetry.span("provider.invoke", provider="nvidia_nim", model="safe-model",
                            operation="stream", prompt="must-not-appear"):
            telemetry.record_usage(input_tokens=12, output_tokens=4)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    provider = next(row for row in rows if row["event_type"] == "invocation")
    assert provider["trace_id"] == trace_id
    assert provider["source_system"] == "tamfis-code"
    assert provider["tier"] == "tamfis-code"
    assert provider["input_tokens"] == 12 and provider["output_tokens"] == 4
    assert provider["calculated_cost"] is None
    assert "must-not-appear" not in repr(provider)


def test_nested_operations_share_trace_and_have_unique_spans(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", tmp_path / "events.jsonl")
    with telemetry.trace("cli.task") as trace_id:
        with telemetry.span("provider.invoke", provider="openrouter", model="m"):
            with telemetry.span("tool.invoke", tool_name="read_file"):
                pass
    rows = telemetry.read_spans(10)
    assert {row["trace_id"] for row in rows} == {trace_id}
    assert len({row["span_id"] for row in rows}) == 3
    assert any(row["parent_span_id"] for row in rows)


def test_invalid_retry_metadata_does_not_break_the_observed_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", tmp_path / "events.jsonl")

    with telemetry.span("provider.invoke", attempt="retry"):
        observed_result = "completed"

    assert observed_result == "completed"
    invocation = next(
        row for row in telemetry.read_spans(10) if row["component"] == "main_completion"
    )
    assert invocation["is_retry"] is False


def test_redirected_telemetry_path_creates_its_own_parent(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "telemetry" / "events.jsonl"
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", path)

    with telemetry.span("tool.invoke", attempt=2):
        pass

    assert path.is_file()
    operation = next(
        row for row in telemetry.read_spans(10) if row["component"] == "tool.invoke"
    )
    assert operation["is_retry"] is True
