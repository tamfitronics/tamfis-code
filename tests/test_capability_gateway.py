import asyncio
import json

import pytest

from tamfis_code.capability_gateway import (
    CompositeCapabilityGateway,
    ExecutionRequest,
    FailoverCapabilityGateway,
    RemoteCapabilityGateway,
    TamfisCodeCapabilityGateway,
    connect_remote_mcp_gateway_with_local_fallback,
)
from tamfis_code.mcp import MCPServer
from tamfis_code.mcp_client import MCPTransportError
from tamfis_code.runner_local import _grant_mcp_external_roots, _scope_tool_arguments


@pytest.mark.asyncio
async def test_gateway_discovers_and_executes_existing_native_mcp_tool(tmp_path):
    gateway = TamfisCodeCapabilityGateway(MCPServer(workspace_root=str(tmp_path)))
    names = {item.name for item in gateway.discover(tags={"native"})}
    assert "read_file" in names
    result = await gateway.execute(ExecutionRequest("native.list_agent_types", {}))
    assert result.error is None
    assert result.output["built_in"]


@pytest.mark.asyncio
async def test_gateway_writes_the_scope_normalized_project_path(tmp_path):
    """The unified gateway must receive the corrected path, not /home/foo."""
    project = tmp_path / "finitron"
    (project / "configs").mkdir(parents=True)
    arguments, error = _scope_tool_arguments(
        "write_file",
        {"path": str(tmp_path / "configs" / "mixture_weights.yaml"), "content": "x"},
        workspace_root=str(tmp_path),
        scope_roots=[project],
    )
    assert error is None
    gateway = TamfisCodeCapabilityGateway(
        MCPServer(workspace_root=str(tmp_path), allowed_workspace_roots=[str(project)])
    )
    result = await gateway.call_tool("write_file", arguments)
    assert result["success"] is True
    assert (project / "configs" / "mixture_weights.yaml").read_text() == "x"


@pytest.mark.asyncio
async def test_approved_external_read_updates_gateway_and_wrapped_server(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "cron-entry"
    target.write_text("0 * * * * root true\n")
    gateway = TamfisCodeCapabilityGateway(
        MCPServer(workspace_root=str(tmp_path), allowed_workspace_roots=[str(tmp_path)])
    )
    _grant_mcp_external_roots(gateway, (outside,))
    result = await gateway.call_tool(
        "execute_command", {"command": f"cat {target}", "cwd": str(tmp_path)}
    )
    assert result["success"] is True
    assert "0 * * * * root true" in result["result"]["stdout"]


def test_gateway_filters_side_effects_without_exposing_everything():
    gateway = TamfisCodeCapabilityGateway(MCPServer())
    writes = gateway.discover(operation="WRITE", tags={"side_effect"})
    assert writes
    assert all(item.approval_required for item in writes)


def test_openai_schemas_are_generated_from_canonical_descriptors():
    class Server:
        def list_tools(self):
            return [
                {
                    "name": "read_status",
                    "description": "Read status",
                    "parameters": {"type": "object", "properties": {"scope": {"type": "string"}}},
                },
                {
                    "name": "write_status",
                    "description": "Write status",
                    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
                },
            ]

    gateway = TamfisCodeCapabilityGateway(Server())
    schemas = gateway.tool_schemas_openai(names=["read_status"])
    assert schemas == [{
        "type": "function",
        "function": {
            "name": "read_status",
            "description": "Read status",
            "parameters": {"type": "object", "properties": {"scope": {"type": "string"}}},
        },
    }]


@pytest.mark.asyncio
async def test_gateway_repairs_json_encoded_object_arguments(tmp_path):
    """Live failure 2026-09-24: execute_command arrived with ``environment``
    as a JSON-encoded string; the strict validator rejected it, the provider
    re-sent the identical call, and the stuck-detector killed the round.
    The normaliser must decode it before validation."""
    calls = []

    class Server:
        def list_tools(self):
            return [{
                "name": "execute_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {"type": "string"},
                        "environment": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                },
            }]

        async def call_tool(self, name, arguments, **_kwargs):
            calls.append((name, arguments))
            return {"success": True, "result": "ok"}

    gateway = TamfisCodeCapabilityGateway(Server())
    result = await gateway.execute(ExecutionRequest("native.execute_command", {
        "command": "echo hi",
        "environment": '{"FOO": "bar"}',
    }))
    assert result.error is None
    assert calls == [("execute_command", {"command": "echo hi", "environment": {"FOO": "bar"}})]


@pytest.mark.asyncio
async def test_gateway_accepts_mcp_namespaced_capability_for_a_native_tool(tmp_path):
    """Live failure 2026-09-24: a provider emitted
    ``mcp__huggingface_hub__hub_repo_search`` as the capability id and got
    "unknown native capability" on every retry until the round was killed.
    An MCP-namespaced id whose bare name IS a registered native tool must
    route to it."""
    gateway = TamfisCodeCapabilityGateway(MCPServer(workspace_root=str(tmp_path)))
    result = await gateway.execute(ExecutionRequest("mcp__local__list_agent_types", {}))
    assert result.error is None
    assert result.output["built_in"]


@pytest.mark.asyncio
async def test_gateway_reports_an_mcp_only_tool_clearly_instead_of_misrouting():
    class Server:
        def list_tools(self):
            return [{"name": "read_file", "description": "read", "parameters": {}}]

    gateway = TamfisCodeCapabilityGateway(Server())
    result = await gateway.execute(ExecutionRequest("mcp__huggingface_hub__hub_repo_search", {}))
    assert result.error is not None
    assert "external MCP tool" in result.error
    assert "huggingface_hub" in result.error


@pytest.mark.asyncio
async def test_native_gateway_rejects_invalid_contract_before_dispatch():
    calls = []

    class Server:
        def list_tools(self):
            return [{
                "name": "write_status",
                "description": "Write status",
                "parameters": {
                    "type": "object",
                    "required": ["value"],
                    "additionalProperties": False,
                    "properties": {"value": {"type": "string"}},
                },
            }]

        async def call_tool(self, name, arguments, **_kwargs):
            calls.append((name, arguments))
            return {"success": True, "result": "ok"}

    gateway = TamfisCodeCapabilityGateway(Server())
    result = await gateway.execute(ExecutionRequest("native.write_status", {}))
    assert result.error == "invalid arguments: missing required argument(s): value"
    assert calls == []


@pytest.mark.asyncio
async def test_native_gateway_repairs_json_encoded_ask_question_arrays():
    calls = []

    class Server:
        def list_tools(self):
            return [{
                "name": "ask_user_question",
                "description": "Ask the user a structured question",
                "parameters": {
                    "type": "object",
                    "required": ["questions"],
                    "additionalProperties": False,
                    "properties": {"questions": {"type": "array"}},
                },
            }]

        async def call_tool(self, name, arguments, **_kwargs):
            calls.append((name, arguments))
            return {"success": True, "result": "answered"}

    encoded_questions = json.dumps([{
        "question": "Modify the cron jobs?",
        "options": json.dumps([{"label": "Yes"}, {"label": "No"}]),
    }])
    gateway = TamfisCodeCapabilityGateway(Server())
    result = await gateway.execute(ExecutionRequest(
        "native.ask_user_question", {"questions": encoded_questions},
    ))

    assert result.error is None
    assert calls[0][0] == "ask_user_question"
    assert isinstance(calls[0][1]["questions"], list)
    assert isinstance(calls[0][1]["questions"][0]["options"], list)


@pytest.mark.asyncio
async def test_native_gateway_repairs_python_repr_ask_question_arrays():
    calls = []

    class Server:
        def list_tools(self):
            return [{
                "name": "ask_user_question",
                "parameters": {
                    "type": "object",
                    "required": ["questions"],
                    "properties": {"questions": {"type": "array"}},
                },
            }]

        async def call_tool(self, name, arguments, **_kwargs):
            calls.append(arguments)
            return {"success": True, "result": "answered"}

    encoded = "[{'question': 'Proceed?', 'header': 'Revised Plan', 'options': "
    encoded += "[{'label': 'Yes'}, {'label': 'No'}]}]"
    gateway = TamfisCodeCapabilityGateway(Server())
    result = await gateway.execute(ExecutionRequest(
        "native.ask_user_question", {"questions": encoded},
    ))

    assert result.error is None
    assert calls[0]["questions"][0]["header"] == "Revised Plan"
    assert calls[0]["questions"][0]["options"] == [{"label": "Yes"}, {"label": "No"}]


@pytest.mark.asyncio
async def test_native_gateway_repairs_single_ask_question_shapes():
    class Server:
        def list_tools(self):
            return [{
                "name": "ask_user_question",
                "parameters": {
                    "type": "object",
                    "required": ["questions"],
                    "properties": {
                        "questions": {"type": "array"},
                        "options": {"type": "array"},
                    },
                },
            }]

        async def call_tool(self, name, arguments, **_kwargs):
            return {"success": True, "result": arguments}

    gateway = TamfisCodeCapabilityGateway(Server())
    result = await gateway.execute(ExecutionRequest(
        "native.ask_user_question",
        {"questions": {"question": "Which route?", "options": "Continue"}},
    ))
    assert result.error is None
    assert result.output["questions"][0]["question"] == "Which route?"
    assert result.output["questions"][0]["options"] == ["Continue"]


@pytest.mark.asyncio
async def test_native_gateway_coerces_numeric_search_arguments():
    calls = []

    class Server:
        def list_tools(self):
            return [{
                "name": "search_code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "offset": {"type": "integer"},
                        "max_results": {"type": "integer"},
                    },
                },
            }]

        async def call_tool(self, name, arguments, **_kwargs):
            calls.append(arguments)
            return {"success": True, "result": []}

    gateway = TamfisCodeCapabilityGateway(Server())
    result = await gateway.execute(ExecutionRequest("native.search_code", {
        "query": "dashboard.json", "offset": "10", "max_results": "25",
    }))
    assert result.error is None
    assert calls == [{"query": "dashboard.json", "offset": 10, "max_results": 25}]


@pytest.mark.asyncio
async def test_remote_gateway_repairs_json_encoded_ask_question_arrays():
    calls = []

    class Connector:
        async def list_tools(self):
            return [{
                "name": "ask_user_question",
                "operation": "READ",
                "parameters": {
                    "type": "object",
                    "required": ["questions"],
                    "properties": {"questions": {"type": "array"}},
                },
            }]

        async def call_tool(self, name, arguments, **_kwargs):
            calls.append(arguments)
            return {"success": True, "result": "answered"}

    gateway = RemoteCapabilityGateway(Connector(), connector_name="remote")
    result = await gateway.execute(ExecutionRequest(
        "remote.remote.ask_user_question",
        {"questions": json.dumps([{"question": "Continue?", "options": "[]"}])},
    ))

    assert result.error is None
    assert isinstance(calls[0]["questions"], list)
    assert calls[0]["questions"][0]["options"] == []


class _RemoteConnector:
    async def list_tools(self):
        return [
            {"name": "workspace_status", "description": "Read remote workspace status", "operation": "READ"},
            {"name": "workspace_update", "description": "Update remote workspace", "operation": "WRITE"},
        ]

    async def call_tool(self, name, arguments):
        return {"success": True, "result": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio
async def test_remote_gateway_validates_descriptor_schema_before_dispatch():
    calls = []

    class Connector:
        async def list_tools(self):
            return [{
                "name": "workspace_update",
                "operation": "WRITE",
                "parameters": {
                    "type": "object",
                    "required": ["value"],
                    "additionalProperties": False,
                    "properties": {"value": {"type": "string"}},
                },
            }]

        async def call_tool(self, name, arguments, **kwargs):
            calls.append((name, arguments, kwargs))
            return {"success": True, "result": "unexpected"}

    gateway = RemoteCapabilityGateway(Connector(), connector_name="tamfisgpt")
    result = await gateway.execute(ExecutionRequest(
        "remote.tamfisgpt.workspace_update", {"value": 42},
    ))

    assert result.error == "invalid arguments: argument 'value' must have type string"
    assert calls == []


@pytest.mark.asyncio
async def test_remote_gateway_requires_approval_context_for_side_effects():
    calls = []

    class Connector:
        async def list_tools(self):
            return [{"name": "workspace_update", "operation": "WRITE"}]

        async def call_tool(self, name, arguments, **kwargs):
            calls.append((name, arguments, kwargs))
            return {"success": True, "result": "unexpected"}

    gateway = RemoteCapabilityGateway(Connector(), connector_name="tamfisgpt")
    result = await gateway.execute(ExecutionRequest(
        "remote.tamfisgpt.workspace_update", {"value": 1},
    ))

    assert result.error == "approval required for side-effecting capability"
    assert calls == []


@pytest.mark.asyncio
async def test_remote_capabilities_are_explicitly_namespaced_without_local_registry_merge(tmp_path):
    local = TamfisCodeCapabilityGateway(MCPServer(workspace_root=str(tmp_path)))
    remote = RemoteCapabilityGateway(_RemoteConnector(), connector_name="tamfisgpt")
    composite = CompositeCapabilityGateway(local, [remote])

    descriptors = await composite.discover(tags={"remote"})
    assert [item.id for item in descriptors] == [
        "remote.tamfisgpt.workspace_status",
        "remote.tamfisgpt.workspace_update",
    ]
    result = await composite.execute(ExecutionRequest(descriptors[0].id, {"workspace": "demo"}))
    assert result.error is None
    assert result.output["name"] == "workspace_status"

    local_descriptors = local.discover()
    assert all(not item.id.startswith("remote.") for item in local_descriptors)


@pytest.mark.asyncio
async def test_remote_discovery_operation_filter_is_not_overwritten_per_tool():
    remote = RemoteCapabilityGateway(_RemoteConnector(), connector_name="tamfisgpt")
    descriptors = await remote.discover(operation="READ")
    assert [item.name for item in descriptors] == ["workspace_status"]


class _UnavailableRemoteConnector:
    async def list_tools(self):
        return [{"name": "list_agent_types", "operation": "READ"}]

    async def call_tool(self, name, arguments):
        raise ConnectionError("remote host unavailable")


@pytest.mark.asyncio
async def test_remote_transport_failure_can_use_matching_local_tool_once(tmp_path):
    local = TamfisCodeCapabilityGateway(MCPServer(workspace_root=str(tmp_path)))
    remote = RemoteCapabilityGateway(_UnavailableRemoteConnector(), connector_name="tamfisgpt")
    gateway = FailoverCapabilityGateway(local, remote)
    result = await gateway.execute(ExecutionRequest("remote.tamfisgpt.list_agent_types", {}))
    assert result.error is None
    assert result.output["built_in"]


class _ContentRemoteConnector:
    async def list_tools(self):
        return [{"name": "read_status", "operation": "READ"}]

    async def call_tool(self, name, arguments):
        return {"success": True, "content": [{"type": "text", "text": "ok"}]}


@pytest.mark.asyncio
async def test_remote_mcp_content_is_not_discarded():
    gateway = RemoteCapabilityGateway(_ContentRemoteConnector(), connector_name="tamfisgpt")
    descriptor = (await gateway.discover())[0]
    result = await gateway.execute(ExecutionRequest(descriptor.id, {}))
    assert result.error is None
    assert result.output == [{"type": "text", "text": "ok"}]


@pytest.mark.asyncio
async def test_remote_execution_identity_is_forwarded_to_capable_connector():
    class Connector:
        def __init__(self):
            self.call = None

        async def list_tools(self):
            return [{"name": "write_status", "operation": "WRITE"}]

        async def call_tool(self, name, arguments, *, idempotency_key=None, request_id=None):
            self.call = (name, arguments, idempotency_key, request_id)
            return {"success": True, "result": "ok"}

    connector = Connector()
    gateway = RemoteCapabilityGateway(connector, connector_name="tamfisgpt")
    descriptor = (await gateway.discover())[0]
    request = ExecutionRequest(
        descriptor.id,
        {"value": 1},
        request_id="execution-1",
        idempotency_key="mission-1:write-status",
        approval_token="approval-1",
    )
    result = await gateway.execute(request)

    assert result.error is None
    assert connector.call == (
        "write_status", {"value": 1}, "mission-1:write-status", "execution-1",
    )


@pytest.mark.asyncio
async def test_remote_execution_preserves_mission_scope_without_auth_passthrough():
    class Connector:
        def __init__(self):
            self.kwargs = None

        async def list_tools(self):
            return [{"name": "read_status", "operation": "READ"}]

        async def call_tool(self, name, arguments, **kwargs):
            self.kwargs = kwargs
            return {"success": True, "result": "ok"}

    connector = Connector()
    gateway = RemoteCapabilityGateway(connector, connector_name="tamfisgpt")
    descriptor = (await gateway.discover())[0]
    result = await gateway.execute(ExecutionRequest(
        descriptor.id,
        request_id="execution-2",
        idempotency_key="mission-2:read-status",
        mission_id="mission-2",
        conversation_id="conversation-2",
        agent_id="agent-2",
        workspace_id="workspace-2",
        correlation_id="trace-2",
        approval_token="approval-2",
    ))

    assert result.error is None
    assert connector.kwargs == {
        "idempotency_key": "mission-2:read-status",
        "request_id": "execution-2",
        "mission_id": "mission-2",
        "conversation_id": "conversation-2",
        "agent_id": "agent-2",
        "workspace_id": "workspace-2",
        "correlation_id": "trace-2",
        "approval_token": "approval-2",
    }


class _TransportErrorRemoteConnector:
    async def list_tools(self):
        return [{"name": "list_agent_types", "operation": "READ"}]

    async def call_tool(self, name, arguments):
        raise MCPTransportError("TamfisGPT became unavailable")


@pytest.mark.asyncio
async def test_remote_mcp_transport_error_uses_local_fallback(tmp_path):
    local = TamfisCodeCapabilityGateway(MCPServer(workspace_root=str(tmp_path)))
    remote = RemoteCapabilityGateway(_TransportErrorRemoteConnector(), connector_name="tamfisgpt")
    gateway = FailoverCapabilityGateway(local, remote)
    result = await gateway.execute(ExecutionRequest("remote.tamfisgpt.list_agent_types", {}))
    assert result.error is None
    assert result.output["built_in"]


@pytest.mark.asyncio
async def test_transport_fallback_preserves_canonical_execution_context():
    class Local:
        def __init__(self):
            self.request = None

        def discover(self, **_kwargs):
            return [type("Descriptor", (), {"id": "native.list_agent_types"})()]

        async def execute(self, request, **_kwargs):
            self.request = request
            return type("Result", (), {"request_id": request.request_id, "output": "local", "error": None})()

    local = Local()
    remote = RemoteCapabilityGateway(_TransportErrorRemoteConnector(), connector_name="tamfisgpt")
    gateway = FailoverCapabilityGateway(local, remote)
    result = await gateway.execute(ExecutionRequest(
        "remote.tamfisgpt.list_agent_types",
        request_id="execution-fallback",
        idempotency_key="fallback-key",
        mission_id="mission-fallback",
        workspace_id="workspace-fallback",
        approval_token="approval-fallback",
    ))

    assert result.error is None
    assert local.request.mission_id == "mission-fallback"
    assert local.request.workspace_id == "workspace-fallback"
    assert local.request.approval_token == "approval-fallback"


class _ApplicationErrorRemoteConnector:
    async def list_tools(self):
        return [{"name": "list_agent_types", "operation": "READ"}]

    async def call_tool(self, name, arguments):
        raise RuntimeError("remote tool rejected the request")


@pytest.mark.asyncio
async def test_remote_application_error_is_not_replayed_locally(tmp_path):
    local = TamfisCodeCapabilityGateway(MCPServer(workspace_root=str(tmp_path)))
    remote = RemoteCapabilityGateway(_ApplicationErrorRemoteConnector(), connector_name="tamfisgpt")
    gateway = FailoverCapabilityGateway(local, remote)
    with pytest.raises(RuntimeError, match="rejected"):
        await gateway.execute(ExecutionRequest("remote.tamfisgpt.list_agent_types", {}))


@pytest.mark.asyncio
async def test_remote_initial_connection_failure_returns_local_only_gateway(tmp_path):
    local = TamfisCodeCapabilityGateway(MCPServer(workspace_root=str(tmp_path)))
    gateway = await connect_remote_mcp_gateway_with_local_fallback(
        local, "http://127.0.0.1:1/unavailable", connector_name="tamfisgpt"
    )
    result = await gateway.execute(ExecutionRequest("native.list_agent_types", {}))
    assert result.error is None
    assert result.output["built_in"]


@pytest.mark.asyncio
async def test_local_gateway_replays_concurrent_idempotent_call_once():
    class CountingServer:
        def __init__(self):
            self.calls = 0

        def list_tools(self):
            return [{"name": "side_effect", "description": "side effect", "parameters": {}}]

        async def call_tool(self, name, arguments, extra_kwargs=None):
            self.calls += 1
            await asyncio.sleep(0.01)
            return {"success": True, "result": {"name": name, "arguments": arguments}}

    server = CountingServer()
    gateway = TamfisCodeCapabilityGateway(server)
    request = ExecutionRequest("native.side_effect", {"value": 1}, idempotency_key="local-once")
    first, second = await asyncio.gather(
        gateway.execute(request), gateway.execute(request),
    )
    assert first.output == second.output
    assert server.calls == 1


@pytest.mark.asyncio
async def test_local_gateway_rejects_idempotency_key_argument_change():
    class Server:
        def list_tools(self):
            return [{"name": "read_status", "parameters": {}}]

        async def call_tool(self, name, arguments, extra_kwargs=None):
            return {"success": True, "result": arguments}

    gateway = TamfisCodeCapabilityGateway(Server())
    await gateway.execute(ExecutionRequest("native.read_status", {"value": 1}, idempotency_key="same"))
    result = await gateway.execute(ExecutionRequest("native.read_status", {"value": 2}, idempotency_key="same"))
    assert result.error == "idempotency key reused with different arguments"
