from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tamfis_code.acp import ACPAgent, ACPError, ACPSession, _ACPRenderer
from tamfis_code.config import Config
from tamfis_code import state as state_module


@pytest.mark.asyncio
async def test_initialize_advertises_acp_v1_and_durable_sessions(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    result = await agent.handle("initialize", {"protocolVersion": 1})
    assert result["protocolVersion"] == 1
    assert result["agentCapabilities"]["loadSession"] is True
    assert result["agentInfo"]["name"] == "Tamfis Code"


@pytest.mark.asyncio
async def test_initialize_negotiates_down_instead_of_rejecting_a_mismatched_version(tmp_path: Path):
    # Real ACP spec requirement (schema/v1 InitializeResponse.protocolVersion
    # doc): "the protocol version the client specified if supported by the
    # agent, or the latest protocol version supported by the agent." The
    # agent must always return its own supported version and let the CLIENT
    # decide whether to disconnect -- confirmed live this used to raise an
    # ACPError instead, breaking the handshake for any well-behaved client
    # that offers a version this agent doesn't happen to match.
    agent = ACPAgent(tmp_path, Config())
    result = await agent.handle("initialize", {"protocolVersion": 99})
    assert result["protocolVersion"] == 1


@pytest.mark.asyncio
async def test_authenticate_returns_an_empty_result(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    assert await agent.handle("authenticate", {}) == {}


@pytest.mark.asyncio
async def test_prompt_streams_agent_message_and_returns_stop_reason(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config(approval_policy="read-only"))
    agent.sessions["7"] = ACPSession("7", 7, tmp_path)
    notifications = []

    async def notify(method, params):
        notifications.append((method, params))

    async def fake_run(session, text, renderer: _ACPRenderer):
        assert text == "Review this change"
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Looks good"}})
        return SimpleNamespace(status="completed", summary="Looks good", error=None)

    agent.notify = notify
    agent._run_prompt = fake_run
    result = await agent.handle(
        "session/prompt",
        {"sessionId": "7", "prompt": [{"type": "text", "text": "Review this change"}]},
    )
    assert result == {"stopReason": "end_turn"}
    assert notifications[0][0] == "session/update"
    assert notifications[0][1]["update"]["sessionUpdate"] == "agent_message_chunk"


@pytest.mark.asyncio
async def test_prompt_emits_the_summary_only_when_no_text_was_already_streamed(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    agent.sessions["7"] = ACPSession("7", 7, tmp_path)
    notifications = []

    async def notify(method, params):
        notifications.append((method, params))

    async def fake_run(session, text, renderer):
        # Never calls renderer.handle_event -- nothing streamed.
        return SimpleNamespace(status="completed", summary="Final answer text", error=None)

    agent.notify = notify
    agent._run_prompt = fake_run
    await agent.handle("session/prompt", {"sessionId": "7", "prompt": "hi"})
    assert len(notifications) == 1
    assert notifications[0][1]["update"]["content"]["text"] == "Final answer text"


@pytest.mark.asyncio
async def test_prompt_on_an_unknown_session_raises_invalid_params(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    with pytest.raises(ACPError) as excinfo:
        await agent.handle("session/prompt", {"sessionId": "999", "prompt": "hi"})
    assert excinfo.value.code == -32602


@pytest.mark.asyncio
async def test_prompt_raises_when_the_underlying_turn_fails(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    agent.sessions["7"] = ACPSession("7", 7, tmp_path)

    async def fake_run(session, text, renderer):
        return SimpleNamespace(status="failed", summary="", error="tool exploded")

    agent._run_prompt = fake_run
    with pytest.raises(ACPError) as excinfo:
        await agent.handle("session/prompt", {"sessionId": "7", "prompt": "hi"})
    assert excinfo.value.code == -32603
    assert "tool exploded" in str(excinfo.value)


@pytest.mark.asyncio
async def test_prompt_text_accepts_resource_link_blocks(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    text = agent._prompt_text([
        {"type": "text", "text": "Look at"},
        {"type": "resource_link", "uri": "file:///a.py"},
    ])
    assert "Look at" in text
    assert "file:///a.py" in text


@pytest.mark.asyncio
async def test_prompt_text_rejects_content_with_no_supported_text(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    with pytest.raises(ACPError):
        agent._prompt_text([{"type": "image", "data": "base64..."}])


@pytest.mark.asyncio
async def test_prompt_text_extracts_embedded_resource_content(tmp_path: Path):
    # Real ACP schema requirement: an EmbeddedResource ("resource" type)
    # block nests its actual payload one level down under "resource"
    # (TextResourceContents' text/uri), unlike ResourceLink which has uri
    # directly at the top level. Confirmed live this used to always read
    # block.get("text")/block.get("uri") at the wrong nesting level,
    # silently dropping every embedded-context resource a client sent even
    # though `embeddedContext: true` is advertised in this agent's own
    # initialize capabilities.
    agent = ACPAgent(tmp_path, Config())
    text = agent._prompt_text([
        {"type": "text", "text": "Review this:"},
        {"type": "resource", "resource": {"uri": "file:///app.py", "text": "def foo(): pass"}},
    ])
    assert "Review this:" in text
    assert "def foo(): pass" in text


@pytest.mark.asyncio
async def test_prompt_text_falls_back_to_uri_for_a_binary_embedded_resource(tmp_path: Path):
    # A BlobResourceContents-backed EmbeddedResource has no "text" field at
    # all (only "blob"/"mimeType"/"uri") -- falls back to the uri as a
    # reference rather than silently producing nothing.
    agent = ACPAgent(tmp_path, Config())
    text = agent._prompt_text([
        {"type": "resource", "resource": {"uri": "file:///image.png", "mimeType": "image/png", "blob": "base64data"}},
    ])
    assert text == "file:///image.png"


@pytest.mark.asyncio
async def test_acp_rejects_workspace_escape(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    agent = ACPAgent(root, Config())
    with pytest.raises(Exception, match="outside configured workspace roots"):
        agent._allowed_cwd(str(outside))


@pytest.mark.asyncio
async def test_acp_allows_an_additionally_configured_workspace_root(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    agent = ACPAgent(root, Config(workspace_roots=[str(extra)]))
    assert agent._allowed_cwd(str(extra)) == extra.resolve()


class TestSessionLoadReplay:
    """Real ACP spec requirement (session-setup docs): 'The Agent MUST
    replay the entire conversation to the Client in the form of
    session/update notifications' on session/load. Confirmed live this was
    previously skipped entirely -- a resumed session's client-side UI
    showed no history at all."""

    def setup_method(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self.workspace = Path(self._tmp.name) / "project"
        self.workspace.mkdir()

    def teardown_method(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self._tmp.cleanup()

    @pytest.mark.asyncio
    async def test_load_session_replays_history_in_order(self):
        state_module.save_session_state(
            42, workspace_root=str(self.workspace),
            conversation_history=[
                {"role": "user", "content": "what is 2+2"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "and 3+3"},
                {"role": "assistant", "content": "6"},
            ],
        )
        agent = ACPAgent(self.workspace, Config())
        notifications = []

        async def notify(method, params):
            notifications.append((method, params))

        agent.notify = notify
        result = await agent.handle("session/load", {"sessionId": "42", "cwd": str(self.workspace)})
        assert result == {"sessionId": "42"}
        updates = [n[1]["update"] for n in notifications]
        assert [u["sessionUpdate"] for u in updates] == [
            "user_message_chunk", "agent_message_chunk", "user_message_chunk", "agent_message_chunk",
        ]
        assert [u["content"]["text"] for u in updates] == ["what is 2+2", "4", "and 3+3", "6"]

    @pytest.mark.asyncio
    async def test_load_session_with_no_history_sends_no_replay_notifications(self):
        state_module.save_session_state(43, workspace_root=str(self.workspace))
        agent = ACPAgent(self.workspace, Config())
        notifications = []

        async def notify(method, params):
            notifications.append((method, params))

        agent.notify = notify
        await agent.handle("session/load", {"sessionId": "43", "cwd": str(self.workspace)})
        assert notifications == []

    @pytest.mark.asyncio
    async def test_load_session_rejects_a_non_numeric_session_id(self):
        agent = ACPAgent(self.workspace, Config())
        with pytest.raises(ACPError) as excinfo:
            await agent.handle("session/load", {"sessionId": "not-a-number"})
        assert excinfo.value.code == -32602

    @pytest.mark.asyncio
    async def test_load_session_requires_a_session_id(self):
        agent = ACPAgent(self.workspace, Config())
        with pytest.raises(ACPError):
            await agent.handle("session/load", {})


class TestNewSession:
    def setup_method(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self.workspace = Path(self._tmp.name) / "project"
        self.workspace.mkdir()

    def teardown_method(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self._tmp.cleanup()

    @pytest.mark.asyncio
    async def test_new_session_returns_a_usable_session_id(self):
        agent = ACPAgent(self.workspace, Config())
        result = await agent.handle("session/new", {"cwd": str(self.workspace)})
        assert result["sessionId"] in agent.sessions
        assert agent.sessions[result["sessionId"]].cwd == self.workspace.resolve()

    @pytest.mark.asyncio
    async def test_new_session_registers_session_scoped_mcp_servers(self):
        agent = ACPAgent(self.workspace, Config())
        result = await agent.handle("session/new", {
            "cwd": str(self.workspace),
            "mcpServers": [{
                "name": "review-tools",
                "command": "python3",
                "args": ["server.py"],
                "env": [{"name": "MODE", "value": "review"}],
            }],
        })
        session = agent.sessions[result["sessionId"]]
        config = session.mcp_servers["review-tools"]
        assert config.command == "python3"
        assert config.args == ("server.py",)
        assert config.env == {"MODE": "review"}

    @pytest.mark.asyncio
    async def test_new_session_rejects_malformed_mcp_server(self):
        agent = ACPAgent(self.workspace, Config())
        with pytest.raises(ACPError, match="needs command or url"):
            await agent.handle("session/new", {
                "cwd": str(self.workspace),
                "mcpServers": [{"name": "broken"}],
            })


class TestDispatchProtocol:
    """Wire-level JSON-RPC framing tests: unknown method, malformed request,
    and generic exceptions never leak a traceback to stdout."""

    @pytest.mark.asyncio
    async def test_unknown_method_returns_method_not_found(self, tmp_path):
        agent = ACPAgent(tmp_path, Config())
        sent = []
        agent.send = lambda payload: sent.append(payload) or asyncio.sleep(0)
        await agent._dispatch({"jsonrpc": "2.0", "id": 1, "method": "totally/unknown"})
        assert sent[0]["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_request_with_no_method_returns_invalid_request(self, tmp_path):
        agent = ACPAgent(tmp_path, Config())
        sent = []
        agent.send = lambda payload: sent.append(payload) or asyncio.sleep(0)
        await agent._dispatch({"jsonrpc": "2.0", "id": 1})
        assert sent[0]["error"]["code"] == -32600

    @pytest.mark.asyncio
    async def test_a_notification_with_no_id_never_sends_a_response(self, tmp_path):
        agent = ACPAgent(tmp_path, Config())
        sent = []
        agent.send = lambda payload: sent.append(payload) or asyncio.sleep(0)
        await agent._dispatch({"jsonrpc": "2.0", "method": "totally/unknown"})
        assert sent == []

    @pytest.mark.asyncio
    async def test_a_generic_exception_becomes_an_internal_error_not_a_crash(self, tmp_path):
        agent = ACPAgent(tmp_path, Config())
        sent = []
        agent.send = lambda payload: sent.append(payload) or asyncio.sleep(0)

        async def boom(method, params=None):
            raise RuntimeError("boom")

        agent.handle = boom
        await agent._dispatch({"jsonrpc": "2.0", "id": 1, "method": "session/new"})
        assert sent[0]["error"]["code"] == -32603
        assert "boom" in sent[0]["error"]["message"]


class TestSessionCancel:
    """session/cancel is a JSON-RPC NOTIFICATION per the real ACP spec (no
    id, no response expected) -- but the spec also requires that the
    session/prompt call it interrupts MUST still resolve with
    stopReason: 'cancelled', even though cancellation propagates as an
    exception internally."""

    @pytest.mark.asyncio
    async def test_cancel_interrupts_an_in_flight_prompt_with_cancelled_stop_reason(self, tmp_path):
        agent = ACPAgent(tmp_path, Config())
        agent.sessions["7"] = ACPSession("7", 7, tmp_path)
        sent = []
        agent.send = lambda payload: sent.append(payload) or asyncio.sleep(0)

        async def slow_run(session, text, renderer):
            await asyncio.sleep(10)
            return SimpleNamespace(status="completed", summary="", error=None)

        agent._run_prompt = slow_run

        prompt_task = asyncio.create_task(agent._dispatch({
            "jsonrpc": "2.0", "id": 1, "method": "session/prompt",
            "params": {"sessionId": "7", "prompt": "do something slow"},
        }))
        await asyncio.sleep(0.1)
        # Sent as a notification -- no "id" key, matching the real protocol.
        await agent._dispatch({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "7"}})
        await asyncio.wait_for(prompt_task, timeout=2)

        assert sent == [{"jsonrpc": "2.0", "id": 1, "result": {"stopReason": "cancelled"}}]

    @pytest.mark.asyncio
    async def test_cancel_for_a_session_with_no_active_prompt_is_a_safe_noop(self, tmp_path):
        agent = ACPAgent(tmp_path, Config())
        result = await agent.handle("session/cancel", {"sessionId": "unknown-session"})
        assert result == {}


class TestRenderer:
    @pytest.mark.asyncio
    async def test_tool_call_requested_becomes_a_pending_tool_call_update(self):
        events = []

        def notify(method, params):
            events.append((method, params))
            return asyncio.sleep(0)

        renderer = _ACPRenderer("7", notify)
        renderer.handle_event({
            "event_type": "tool_call_requested",
            "payload": {"call_id": "call_1", "name": "write_file", "arguments": {"path": "a.py"}},
        })
        assert len(renderer._pending) == 1
        await renderer.drain()

    @pytest.mark.asyncio
    async def test_unrecognized_event_types_are_a_no_op(self):
        events = []

        def notify(method, params):
            events.append((method, params))
            return asyncio.sleep(0)

        renderer = _ACPRenderer("7", notify)
        renderer.handle_event({"event_type": "something_unrelated", "payload": {}})
        assert renderer._pending == set()


if __name__ == "__main__":
    pytest.main([__file__])
