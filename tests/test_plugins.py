import asyncio
import json
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


# --------------------------------------------------------------------------
# Manifest plugins and degraded mode (Pillar 4)
# --------------------------------------------------------------------------


def _write_manifest(directory: Path, payload, filename: str = "kimi.plugin.json") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload))
    return path


def test_manifest_plugin_loads_and_registers_its_tools(tmp_path: Path, monkeypatch):
    from tamfis_code.plugins import load_manifest_plugins, register_plugin_tools

    # A real importable module, so the test exercises the manifest's
    # entrypoint resolution rather than a patched callable.
    (tmp_path / "manifest_plugin_mod.py").write_text(
        "async def hello(name):\n    return {'message': f'hello {name}'}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    _write_manifest(tmp_path / "plugins", {
        "name": "manifest-hello",
        "version": "3.1",
        "tools": [{
            "name": "manifest_hello",
            "description": "Say hello from a manifest",
            "entrypoint": "manifest_plugin_mod:hello",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
        }],
        "skill_roots": [str(tmp_path / "skills")],
    })

    plugins = load_manifest_plugins([tmp_path / "plugins"])
    assert [plugin.name for plugin in plugins] == ["manifest-hello"]
    assert plugins[0].version == "3.1"
    assert plugins[0].error is None
    assert plugins[0].skill_roots == [str(tmp_path / "skills")]

    server = MCPServer(workspace_root=str(tmp_path))
    loaded = register_plugin_tools(server, include_manifests=False)
    assert loaded == []
    assert "manifest_hello" not in server.tools

    loaded = load_manifest_plugins([tmp_path / "plugins"])
    for plugin in loaded:
        for tool in plugin.tools:
            server.register_tool(
                name=tool["name"], description=tool["description"],
                parameters=tool["parameters"], handler=tool["handler"],
            )
    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        server.call_tool("manifest_hello", {"name": "Tamfis"})
    )
    assert result["success"] is True
    assert result["result"]["message"] == "hello Tamfis"


def test_a_broken_manifest_degrades_instead_of_crashing(tmp_path: Path):
    """A third-party manifest this project does not control must never take the
    agent down: it is recorded as degraded, and built-in tools keep working."""
    from tamfis_code.plugins import load_manifest_plugins, plugin_diagnostics

    workspace = tmp_path / "workspace"
    plugins_dir = workspace / ".tamfis" / "plugins"
    _write_manifest(plugins_dir, "{ this is not json", filename="plugin.json")
    _write_manifest(
        plugins_dir,
        {"name": "shape-ok-but-tool-broken", "tools": [
            {"name": "no_handler"},
            {"description": "nameless"},
            {"name": "bad_entrypoint", "entrypoint": "not_a_module:nope"},
            {"name": "shell_tool", "command": ["echo", "hi"]},
        ]},
    )

    plugins = {
        plugin.name: plugin
        for plugin in load_manifest_plugins([plugins_dir])
    }
    assert any(plugin.error for plugin in plugins.values())
    broken = plugins["shape-ok-but-tool-broken"]
    assert broken.error is None  # the plugin loaded...
    assert broken.tools == []  # ...with none of its unusable tools

    report = plugin_diagnostics(workspace)
    # The report carries every degradation reason for debugging, and the
    # built-in tool surface is untouched by any of it.
    assert report.degraded_mode is True
    assert any("unreadable manifest" in item["reason"] for item in report.degraded)
    assert "plugin(s) loaded" in report.summary()

    server = MCPServer(workspace_root=str(tmp_path))
    assert "read_file" in server.tools  # built-ins survive a bad plugin


def test_unreadable_manifest_directory_is_skipped_quietly(tmp_path: Path):
    from tamfis_code.plugins import discover_manifest_paths, load_manifest_plugins

    assert discover_manifest_paths([tmp_path / "does-not-exist"]) == []
    assert load_manifest_plugins([tmp_path / "does-not-exist"]) == []


def test_kimi_code_home_manifests_are_discovered(tmp_path: Path, monkeypatch):
    from tamfis_code.plugins import default_manifest_dirs, discover_manifest_paths

    kimi_home = tmp_path / "kimi-home"
    _write_manifest(kimi_home / "plugins", {"name": "kimi-plugin"}, filename="kimi.plugin.json")
    monkeypatch.setenv("KIMI_CODE_HOME", str(kimi_home))

    directories = default_manifest_dirs(tmp_path / "workspace")
    assert kimi_home / "plugins" in directories
    assert tmp_path / "workspace" / ".tamfis" / "plugins" in directories
    assert discover_manifest_paths(directories)
