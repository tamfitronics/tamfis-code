"""Tamfis-Code compatibility facade for the Universal Capability Gateway.

Tamfis-Code remains usable as a standalone package and therefore does not
import the backend repository.  This facade uses the same wire-level contract
names as ``tamgpt6.shared.capabilities.gateway`` and translates native
MCPServer tools without changing the existing runner or tool policy.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

from .mcp_client import MCPTransportError


_SIDE_EFFECT_NAMES = {
    "write_file", "edit_file", "delete_file", "execute_command", "code_exec",
    "send_email", "publish", "remote_exec", "generate_file", "generate_document",
}


def _normalise_ask_user_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Repair JSON-encoded arrays emitted by weaker provider tool adapters.

    The canonical schema is intentionally strict, but several providers send
    an array as a JSON string (and sometimes encode the nested ``options`` a
    second time).  Decode only this well-known interactive capability; all
    other tools retain their exact arguments and malformed values still fail
    normal schema validation.
    """
    result = dict(arguments)
    if name != "ask_user_question":
        return result

    def decode(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text.startswith(("[", "{")):
            return value
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
        return decoded

    for key in ("questions", "options"):
        if key in result:
            result[key] = decode(result[key])
    questions = result.get("questions")
    if isinstance(questions, list):
        repaired: list[Any] = []
        for item in questions:
            if isinstance(item, Mapping):
                item = dict(item)
                if "options" in item:
                    item["options"] = decode(item["options"])
            repaired.append(item)
        result["questions"] = repaired
    return result


def _normalise_schema_arguments(
    name: str, schema: Mapping[str, Any], arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Coerce conservative scalar JSON-wire variants before validation.

    Some providers serialize numeric tool fields as strings even when the
    declared schema says integer (for example search ``offset`` and
    ``max_results``). Only schema-declared scalar fields are converted; bad
    values remain unchanged and are rejected by the normal validator.
    """
    result = _normalise_ask_user_arguments(name, arguments)
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping):
        return result
    for key, rule in properties.items():
        value = result.get(key)
        if not isinstance(value, str) or not isinstance(rule, Mapping):
            continue
        value_type = rule.get("type")
        text = value.strip()
        try:
            if value_type == "integer" and re.fullmatch(r"[+-]?\d+", text):
                result[key] = int(text)
            elif value_type == "number" and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
                result[key] = float(text) if "." in text else int(text)
        except (TypeError, ValueError, OverflowError):
            continue
    return result


def _validate_input_schema(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    """Validate the common JSON-schema contract before native dispatch."""
    if not schema:
        return
    if not isinstance(arguments, Mapping):
        raise ValueError("capability arguments must be an object")
    matchers = {
        "object": lambda value: isinstance(value, Mapping),
        "array": lambda value: isinstance(value, (list, tuple)),
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "null": lambda value: value is None,
    }
    expected = schema.get("type")
    if isinstance(expected, str) and expected in matchers and not matchers[expected](arguments):
        raise ValueError(f"arguments must have type {expected}")
    required = schema.get("required") or []
    missing = [str(key) for key in required if key not in arguments]
    if missing:
        raise ValueError("missing required argument(s): " + ", ".join(missing))
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(str(key) for key in arguments if key not in properties)
        if unknown:
            raise ValueError("unknown argument(s): " + ", ".join(unknown))
    for key, value in arguments.items():
        rule = properties.get(key)
        if not isinstance(rule, Mapping):
            continue
        allowed = rule.get("enum")
        if allowed is not None and value not in allowed:
            raise ValueError(f"argument {key!r} is not an allowed value")
        value_type = rule.get("type")
        if isinstance(value_type, str) and value_type in matchers and not matchers[value_type](value):
            raise ValueError(f"argument {key!r} must have type {value_type}")


@dataclass(frozen=True)
class CapabilityDescriptor:
    id: str
    name: str
    description: str
    protocol: str = "mcp"
    operation: str = "READ"
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    approval_required: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionRequest:
    capability_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: str | None = None
    mission_id: str | None = None
    conversation_id: str | None = None
    agent_id: str | None = None
    workspace_id: str | None = None
    correlation_id: str | None = None
    approval_token: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    request_id: str
    output: Any = None
    error: str | None = None


class RemoteCapabilityGateway:
    """Optional remote MCP capability view for standalone installations.

    ``connector`` is any authenticated MCP client exposing asynchronous
    ``list_tools`` and ``call_tool`` methods (Tamfis-Code's existing
    ``StandaloneMCPBridge`` satisfies this contract). The remote registry is
    never merged into the local registry: every descriptor is namespaced as
    ``remote.<connector>.<tool>`` and credentials remain owned by the client.
    """

    def __init__(self, connector: Any, *, connector_name: str) -> None:
        self.connector = connector
        self.connector_name = _safe_namespace(connector_name)

    async def discover(self, *, operation: str | None = None, tags: set[str] | None = None) -> list[CapabilityDescriptor]:
        descriptors: list[CapabilityDescriptor] = []
        tools = await self.connector.list_tools()
        for tool in tools or []:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            tool_operation = str(tool.get("operation", "EXECUTE")).upper()
            if tool_operation not in {"READ", "WRITE", "DELETE", "SEND", "PUBLISH", "EXECUTE"}:
                tool_operation = "EXECUTE"
            descriptor = CapabilityDescriptor(
                id=f"remote.{self.connector_name}.{_safe_namespace(name)}",
                name=name,
                description=str(tool.get("description", "")),
                protocol="mcp",
                operation=tool_operation,
                input_schema=tool.get("parameters", tool.get("inputSchema", {})),
                approval_required=tool_operation != "READ",
                tags=("remote", "mcp", self.connector_name),
            )
            if operation and operation.upper() != descriptor.operation:
                continue
            if tags and not tags.issubset(set(descriptor.tags)):
                continue
            descriptors.append(descriptor)
        return sorted(descriptors, key=lambda item: item.id)

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        prefix = f"remote.{self.connector_name}."
        if not request.capability_id.startswith(prefix):
            return ExecutionResult(request.request_id, error="unknown remote capability namespace")
        name = request.capability_id[len(prefix):]
        # Resolve the wire name from discovery so names containing punctuation
        # are not reconstructed incorrectly from the safe namespace.
        tools = await self.connector.list_tools()
        match = next((tool for tool in tools or [] if _safe_namespace(str(tool.get("name", ""))) == name), None)
        if match is None:
            return ExecutionResult(request.request_id, error=f"remote capability not found: {name}")
        arguments = _normalise_schema_arguments(
            name, match.get("parameters", match.get("inputSchema", {})) or {}, request.arguments,
        )
        try:
            _validate_input_schema(
                match.get("parameters", match.get("inputSchema", {})) or {},
                arguments,
            )
        except ValueError as exc:
            # Reject malformed input locally so a remote connector cannot
            # observe or partially execute a request that violates the
            # canonical descriptor contract.
            return ExecutionResult(request.request_id, error=f"invalid arguments: {exc}")
        if str(match.get("operation", "EXECUTE")).upper() != "READ" and not request.approval_token:
            # The authenticated backend performs the cryptographic approval
            # check. This local guard is still fail-closed for custom or
            # embedded connectors, so a side-effecting descriptor cannot be
            # forwarded accidentally without an approval context.
            return ExecutionResult(request.request_id, error="approval required for side-effecting capability")
        # Preserve the canonical execution identity across the optional
        # remote MCP hop.  StandaloneMCPBridge maps these values to per-call
        # HTTP headers; older/custom connectors may expose only the legacy
        # two-argument surface, so retain compatibility without weakening
        # the local idempotency boundary.
        call_tool = self.connector.call_tool
        context_kwargs = {
            "idempotency_key": request.idempotency_key,
            "request_id": request.request_id,
            "mission_id": request.mission_id,
            "conversation_id": request.conversation_id,
            "agent_id": request.agent_id,
            "workspace_id": request.workspace_id,
            "correlation_id": request.correlation_id,
            "approval_token": request.approval_token,
        }
        # Inspect before invoking so a legacy connector is called once with
        # its supported surface. Catching TypeError after invocation would be
        # unsafe: a connector could perform the side effect and then raise a
        # TypeError internally, making a compatibility retry a duplicate.
        try:
            parameters = inspect.signature(call_tool).parameters.values()
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            supported_kwargs = {
                key: value for key, value in context_kwargs.items()
                if value and (
                    accepts_kwargs
                    or key in inspect.signature(call_tool).parameters
                )
            }
        except (TypeError, ValueError):
            # Opaque extension callables retain the old two-argument contract
            # rather than receiving speculative keywords.
            supported_kwargs = {}
        result = await call_tool(
            str(match["name"]), arguments, **supported_kwargs,
        )
        if not result.get("success", True):
            return ExecutionResult(request.request_id, error=str(result.get("error", "remote tool failed")))
        output = result.get("result")
        if output is None:
            # StandaloneMCPBridge preserves the MCP wire ``content`` array;
            # do not discard it merely because a connector uses MCP rather
            # than the backend's convenience ``result`` field.
            output = result.get("content", result)
        return ExecutionResult(request.request_id, output=output)


class CompositeCapabilityGateway:
    """Namespaced local/remote capability view for optional remote use."""

    def __init__(self, local: "TamfisCodeCapabilityGateway", remotes: Iterable[RemoteCapabilityGateway] = ()) -> None:
        self.local = local
        self.remotes = tuple(remotes)

    async def discover(self, *, operation: str | None = None, tags: set[str] | None = None) -> list[CapabilityDescriptor]:
        descriptors = self.local.discover(operation=operation, tags=tags)
        for remote in self.remotes:
            descriptors.extend(await remote.discover(operation=operation, tags=tags))
        return sorted(descriptors, key=lambda item: item.id)

    async def execute(self, request: ExecutionRequest, *, extra_kwargs: dict[str, Any] | None = None) -> ExecutionResult:
        if request.capability_id.startswith("native."):
            return await self.local.execute(request, extra_kwargs=extra_kwargs)
        for remote in self.remotes:
            if request.capability_id.startswith(f"remote.{remote.connector_name}."):
                return await remote.execute(request)
        return ExecutionResult(request.request_id, error="unknown capability namespace")


class FailoverCapabilityGateway:
    """Remote-first view with a conservative local fallback.

    A local retry is never attempted after an ambiguous remote execution.
    Only a caller that explicitly selects a local/native capability executes
    locally; this prevents duplicate writes after a network interruption.
    """

    def __init__(self, local: "TamfisCodeCapabilityGateway", remote: RemoteCapabilityGateway | None = None) -> None:
        self.local = local
        self.remote = remote

    async def discover(self, *, operation: str | None = None, tags: set[str] | None = None) -> list[CapabilityDescriptor]:
        local = self.local.discover(operation=operation, tags=tags)
        if self.remote is None:
            return local
        try:
            return sorted(local + await self.remote.discover(operation=operation, tags=tags), key=lambda item: item.id)
        except Exception:
            return local

    async def execute(self, request: ExecutionRequest, *, extra_kwargs: dict[str, Any] | None = None) -> ExecutionResult:
        if self.remote is None or not request.capability_id.startswith("remote."):
            return await self.local.execute(request, extra_kwargs=extra_kwargs)
        try:
            return await self.remote.execute(request)
        except (MCPTransportError, ConnectionError, TimeoutError, OSError) as exc:
            # A transport failure occurred before a result was received. A
            # matching local tool may therefore be used once; ambiguous
            # application results are never replayed here.
            wire_name = request.capability_id.split(".", 2)[-1]
            local_id = f"native.{wire_name}"
            local_names = {item.id for item in self.local.discover()}
            if local_id in local_names:
                return await self.local.execute(
                    ExecutionRequest(
                        local_id,
                        request.arguments,
                        request.request_id,
                        request.idempotency_key,
                        request.mission_id,
                        request.conversation_id,
                        request.agent_id,
                        request.workspace_id,
                        request.correlation_id,
                        request.approval_token,
                    ),
                    extra_kwargs=extra_kwargs,
                )
            return ExecutionResult(request.request_id, error=f"remote transport unavailable and no local fallback: {exc}")


async def connect_remote_mcp_gateway(
    url: str,
    *,
    connector_name: str,
    workspace_root: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> RemoteCapabilityGateway:
    """Create an authenticated, optional Streamable-HTTP MCP connector.

    Credentials are supplied by the caller/configuration layer and are held
    by the MCP client; this function never copies them into capability
    descriptors or forwards an inbound Tamfis-Code session token.
    """
    from .mcp_client import MCPServerConfig, StandaloneMCPBridge

    config = MCPServerConfig(
        name=_safe_namespace(connector_name), url=url,
        headers=dict(headers or {}),
    )
    bridge = StandaloneMCPBridge(
        workspace_root=workspace_root,
        servers={config.name: config},
    )
    if not await bridge.initialize(background=False):
        raise RuntimeError(f"could not initialize remote MCP connector: {connector_name}")
    return RemoteCapabilityGateway(bridge, connector_name=connector_name)


async def connect_remote_mcp_gateway_with_local_fallback(
    local: "TamfisCodeCapabilityGateway",
    url: str,
    *,
    connector_name: str,
    workspace_root: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> FailoverCapabilityGateway:
    """Connect remote MCP when available while preserving local operation.

    Initial connection failure is an expected availability state and returns
    a local-only gateway. Once connected, only a transport failure before a
    remote result can invoke a matching local capability; ambiguous results
    are returned as failures and are not replayed.
    """
    try:
        remote = await connect_remote_mcp_gateway(
            url, connector_name=connector_name, workspace_root=workspace_root, headers=headers,
        )
    except (ConnectionError, TimeoutError, OSError, RuntimeError):
        return FailoverCapabilityGateway(local)
    return FailoverCapabilityGateway(local, remote)


def _safe_namespace(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value).strip())
    return normalized.strip("._") or "unnamed"


class TamfisCodeCapabilityGateway:
    """Expose the existing MCPServer through canonical discovery/execution.

    The local runner wraps its existing MCPServer with this facade. Direct
    MCPServer callers remain compatible, while remote registries stay
    namespaced and are never merged into the local registry.
    """

    def __init__(self, server: Any) -> None:
        self.server = server
        # The backend gateway owns cross-worker durability.  This small
        # process-local ledger protects the standalone fallback from
        # concurrent duplicate calls while a caller is reconnecting; it does
        # not pretend to replace PostgreSQL when the remote gateway is used.
        self._idempotency_lock = asyncio.Lock()
        self._completed: dict[str, tuple[str, ExecutionResult]] = {}
        self._inflight: dict[str, tuple[str, asyncio.Future[ExecutionResult]]] = {}

    def discover(self, *, operation: str | None = None, tags: set[str] | None = None) -> list[CapabilityDescriptor]:
        descriptors = []
        for tool in self.server.list_tools():
            name = str(tool["name"])
            tool_operation = "EXECUTE" if name == "execute_command" else ("WRITE" if name in _SIDE_EFFECT_NAMES else "READ")
            tool_tags = {"native", "mcp"}
            if name in _SIDE_EFFECT_NAMES:
                tool_tags.add("side_effect")
            if operation and operation.upper() != tool_operation:
                continue
            if tags and not tags.issubset(tool_tags):
                continue
            descriptors.append(CapabilityDescriptor(
                id=f"native.{name}", name=name,
                description=str(tool.get("description", "")),
                protocol="mcp", operation=tool_operation,
                input_schema=tool.get("parameters", {}),
                approval_required=name in _SIDE_EFFECT_NAMES,
                tags=tuple(sorted(tool_tags)),
            ))
        return sorted(descriptors, key=lambda item: item.id)

    async def execute(self, request: ExecutionRequest, *, extra_kwargs: dict[str, Any] | None = None) -> ExecutionResult:
        prefix, _, name = request.capability_id.partition(".")
        if prefix != "native" or not name:
            return ExecutionResult(request.request_id, error="unknown capability namespace")
        descriptor = next((item for item in self.discover() if item.id == request.capability_id), None)
        if descriptor is None:
            return ExecutionResult(request.request_id, error=f"unknown native capability: {name}")
        arguments = _normalise_schema_arguments(name, descriptor.input_schema, request.arguments)
        try:
            _validate_input_schema(descriptor.input_schema, arguments)
        except ValueError as exc:
            return ExecutionResult(request.request_id, error=f"invalid arguments: {exc}")
        key = request.idempotency_key
        fingerprint = hashlib.sha256(json.dumps({
            "capability_id": request.capability_id,
            "arguments": arguments,
            "extra_kwargs": extra_kwargs or {},
        }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        owner = False
        wait_for: asyncio.Future[ExecutionResult] | None = None
        if key:
            async with self._idempotency_lock:
                cached = self._completed.get(key)
                if cached is not None:
                    old_fingerprint, old_result = cached
                    if old_fingerprint != fingerprint:
                        return ExecutionResult(request.request_id, error="idempotency key reused with different arguments")
                    return ExecutionResult(request.request_id, output=old_result.output, error=old_result.error)
                running = self._inflight.get(key)
                if running is not None:
                    old_fingerprint, wait_for = running
                    if old_fingerprint != fingerprint:
                        return ExecutionResult(request.request_id, error="idempotency key is already in progress with different arguments")
                else:
                    wait_for = asyncio.get_running_loop().create_future()
                    self._inflight[key] = (fingerprint, wait_for)
                    owner = True
        if not owner and wait_for is not None:
            return await wait_for
        try:
            result = await self.server.call_tool(
                name, arguments, extra_kwargs=extra_kwargs,
            )
            if not result.get("success"):
                outcome = ExecutionResult(request.request_id, error=str(result.get("error", "tool failed")))
            else:
                outcome = ExecutionResult(request.request_id, output=result.get("result"))
        except Exception as exc:
            outcome = ExecutionResult(request.request_id, error=str(exc))
        if key and owner:
            async with self._idempotency_lock:
                self._inflight.pop(key, None)
                self._completed[key] = (fingerprint, outcome)
                assert wait_for is not None
                if not wait_for.done():
                    wait_for.set_result(outcome)
        return outcome

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, extra_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Legacy MCPServer-compatible surface backed by the gateway."""
        result = await self.execute(
            ExecutionRequest(f"native.{name}", arguments), extra_kwargs=extra_kwargs,
        )
        if result.error:
            return {"success": False, "error": result.error}
        return {"success": True, "result": result.output, "tool": name}

    def list_tools(self):
        """Preserve the existing synchronous MCP discovery contract."""
        return self.server.list_tools()

    def tool_schemas_openai(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Build model schemas from canonical descriptors.

        The raw MCPServer method remains available through the wrapped server
        for legacy callers, but agent requests must use the same descriptor
        view that the gateway executes. This prevents a stale local registry
        entry from being advertised and then rejected at execution time.
        """
        selected = set(names) if names is not None else None
        schemas: list[dict[str, Any]] = []
        for descriptor in self.discover():
            if selected is not None and descriptor.name not in selected:
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": descriptor.name,
                    "description": descriptor.description,
                    "parameters": dict(descriptor.input_schema),
                },
            })
        return schemas

    def __getattr__(self, name: str) -> Any:
        # Existing runner helpers access MCPServer policy/transport attributes
        # directly; delegation preserves those without creating a second
        # registry.
        return getattr(self.server, name)
