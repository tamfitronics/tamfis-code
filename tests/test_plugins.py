from pathlib import Path
from unittest.mock import patch

import pytest

from tamfis_code.mcp import MCPServer
from tamfis_code.plugins import load_plugins


async def _hello(name: str):
    return {"message": f"hello {name}"}


class _Dist:
    version = "2.0"


class _Entry:
    name = "sample"
    value = "sample:factory"
    dist = _Dist()

    def load(self):
        return lambda: {
            "name": "sample-plugin",
            "tools": [{
                "name": "plugin_hello", "description": "Say hello",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                "handler": _hello,
            }],
            "skill_roots": ["/tmp/sample-skills"],
        }


def test_loads_entrypoint_plugin_metadata():
    with patch("tamfis_code.plugins._entry_points", return_value=[_Entry()]):
        plugins = load_plugins()
    assert plugins[0].name == "sample-plugin"
    assert plugins[0].version == "2.0"


@pytest.mark.asyncio
async def test_plugin_tool_is_registered_and_callable(tmp_path: Path):
    with patch("tamfis_code.plugins._entry_points", return_value=[_Entry()]):
        server = MCPServer(workspace_root=str(tmp_path))
    result = await server.call_tool("plugin_hello", {"name": "Tamfis"})
    assert result["success"] is True
    assert result["result"]["message"] == "hello Tamfis"


class _SkillPluginEntry(_Entry):
    """Same shape as _Entry above but only ever used with a skill_roots
    value pointed at a real temp directory -- kept separate so that test's
    intent (skill discovery, not tool registration) reads clearly."""

    def __init__(self, skill_root: str):
        self._skill_root = skill_root

    def load(self):
        return lambda: {"name": "release-notes-plugin", "skill_roots": [self._skill_root]}


def test_plugin_skill_roots_are_actually_discovered_by_the_skill_registry(tmp_path: Path):
    """plugin_skill_roots() (this module) is consumed by
    openhands.skills.workspace_skill_registry alongside the Kimi/Claude/
    Codex/shared/project skill roots -- but that wiring had no test proving
    a plugin-contributed SKILL.md is actually discovered and injected, only
    that Plugin.skill_roots gets parsed off the plugin's own manifest
    (test_loads_entrypoint_plugin_metadata above). Closes that gap
    end-to-end: a real SKILL.md under a plugin-declared root, loaded
    through the real registry, ends up in a real skill_prompt() output.
    """
    from tamfis_code.openhands.skills import skill_prompt, workspace_skill_registry

    plugin_skills_root = tmp_path / "plugin-skills"
    (plugin_skills_root / "release-notes").mkdir(parents=True)
    (plugin_skills_root / "release-notes" / "SKILL.md").write_text(
        "---\nname: release-notes\ndescription: Draft release notes\ntags: release, changelog\n---\n"
        "Plugin-contributed release notes instructions."
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    with patch("tamfis_code.plugins._entry_points", return_value=[_SkillPluginEntry(str(plugin_skills_root))]):
        registry = workspace_skill_registry(workspace_root)
        registry.load()
        assert registry.get("release-notes").instructions == "Plugin-contributed release notes instructions."
        prompt = skill_prompt(workspace_root, "please draft the release notes")

    assert "Plugin-contributed release notes instructions." in prompt
