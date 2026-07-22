import asyncio
import tempfile
from pathlib import Path

import pytest

from tamfis_code.openhands.automation import Automation, AutomationStore
from tamfis_code.openhands.conversation import ConversationManager, ConversationState
from tamfis_code.openhands.events import EventKind, EventStore
from tamfis_code.openhands.leases import LeaseManager
from tamfis_code.openhands.security import SecretVault, SecurityAnalyzer
from tamfis_code.openhands.skills import SkillRegistry
from tamfis_code.openhands.tools import default_registry
from tamfis_code.openhands.workspace import LocalWorkspace, WorkspaceError


def test_event_store_is_append_only_and_replayable(tmp_path: Path):
    store = EventStore(tmp_path / "events")
    first = store.emit("conv", EventKind.USER_MESSAGE, {"content": "hello"})
    second = store.emit("conv", EventKind.AGENT_MESSAGE, {"content": "world"})
    assert first.sequence == 1
    assert second.sequence == 2
    assert [e.kind for e in store.read("conv")] == [EventKind.USER_MESSAGE, EventKind.AGENT_MESSAGE]
    assert [e.id for e in store.read("conv", after=1)] == [second.id]


def test_local_workspace_confines_paths_and_restores_snapshots(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("before")
    ws = LocalWorkspace(root, state_dir=tmp_path / "state")
    with pytest.raises(WorkspaceError):
        ws.resolve("../escape")
    snapshot = ws.snapshot(label="before change")
    ws.write_text("a.txt", "after")
    assert ws.read_text("a.txt") == "after"
    ws.restore(snapshot["id"])
    assert ws.read_text("a.txt") == "before"


def test_default_tools_execute_real_actions(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    ws = LocalWorkspace(root, state_dir=tmp_path / "state")
    registry = default_registry(ws)

    async def run():
        write = await registry.invoke("write_file", {"path": "hello.txt", "content": "hello"})
        assert write.ok
        read = await registry.invoke("read_file", {"path": "hello.txt"})
        assert read.ok and read.output == "hello"
        command = await registry.invoke("terminal", {"command": "printf terminal-ok"})
        assert command.ok and command.output.stdout == "terminal-ok"

    asyncio.run(run())


def test_skills_load_from_project_tree(tmp_path: Path):
    root = tmp_path / ".tamfis" / "skills" / "debug-ci"
    root.mkdir(parents=True)
    (root / "skill.md").write_text(
        "---\nname: debug-ci\ndescription: Diagnose CI failures\nrequired_tools: terminal, read_file\ntags: ci, tests\n---\nInspect logs, repair the defect and re-run tests."
    )
    registry = SkillRegistry([tmp_path / ".tamfis" / "skills"])
    skills = registry.load()
    assert skills["debug-ci"].required_tools == ("terminal", "read_file")
    assert registry.match("fix the CI tests")[0].name == "debug-ci"


def test_conversation_lifecycle_and_events(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    manager = ConversationManager(tmp_path / "server")
    conversation = manager.create(root)
    conversation.send_message("inspect the repository")
    conversation.pause()
    assert conversation.state == ConversationState.PAUSED
    conversation.resume()
    assert conversation.state == ConversationState.RUNNING
    conversation.cancel()
    assert conversation.state == ConversationState.CANCELLED
    assert len(conversation.replay()) >= 5


def test_leases_are_exclusive_and_releasable(tmp_path: Path):
    manager = LeaseManager(tmp_path / "leases.json")
    lease = manager.acquire("workspace", "agent-a", ttl=60)
    with pytest.raises(RuntimeError):
        manager.acquire("workspace", "agent-b", ttl=60)
    renewed = manager.renew(lease, ttl=120)
    assert renewed.expires_at > lease.expires_at
    manager.release(renewed)
    assert manager.acquire("workspace", "agent-b", ttl=60).owner == "agent-b"


def test_security_and_secret_vault(tmp_path: Path):
    analyzer = SecurityAnalyzer()
    destructive = analyzer.analyze("terminal", {"command": "rm -rf build"}, approval_policy="ask")
    assert destructive.risk == "critical" and destructive.requires_approval
    vault = SecretVault(tmp_path / "secrets.json")
    vault.set("TOKEN", "secret")
    assert vault.names() == ["TOKEN"]
    assert vault.environment(["TOKEN"]) == {"TOKEN": "secret"}
    assert oct((tmp_path / "secrets.json").stat().st_mode & 0o777) == "0o600"


def test_automation_store_round_trip(tmp_path: Path):
    store = AutomationStore(tmp_path / "automations.json")
    item = Automation("nightly", "run tests", str(tmp_path), 3600)
    store.save([item])
    loaded = store.load()
    assert loaded[0].name == "nightly"
    assert loaded[0].objective == "run tests"
