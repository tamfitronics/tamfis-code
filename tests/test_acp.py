from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tamfis_code.acp import ACPAgent, ACPSession, _ACPRenderer
from tamfis_code.config import Config


@pytest.mark.asyncio
async def test_initialize_advertises_acp_v1_and_durable_sessions(tmp_path: Path):
    agent = ACPAgent(tmp_path, Config())
    result = await agent.handle("initialize", {"protocolVersion": 1})
    assert result["protocolVersion"] == 1
    assert result["agentCapabilities"]["loadSession"] is True
    assert result["agentInfo"]["name"] == "Tamfis Code"


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
async def test_acp_rejects_workspace_escape(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    agent = ACPAgent(root, Config())
    with pytest.raises(Exception, match="outside configured workspace roots"):
        agent._allowed_cwd(str(outside))
