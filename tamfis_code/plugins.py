"""Python entry-point plugin discovery for Tamfis Code."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any


ENTRYPOINT_GROUP = "tamfis_code.plugins"


@dataclass
class Plugin:
    name: str
    version: str = "unknown"
    tools: list[dict[str, Any]] = field(default_factory=list)
    skill_roots: list[str] = field(default_factory=list)
    source: str = ""
    error: str | None = None


def _entry_points() -> list[Any]:
    discovered = metadata.entry_points()
    return list(discovered.select(group=ENTRYPOINT_GROUP)) if hasattr(discovered, "select") else list(discovered.get(ENTRYPOINT_GROUP, []))


def load_plugins() -> list[Plugin]:
    plugins: list[Plugin] = []
    for entry in _entry_points():
        try:
            loaded = entry.load()
            value = loaded() if callable(loaded) else loaded
            if not isinstance(value, dict):
                raise TypeError("plugin factory must return a mapping")
            plugins.append(Plugin(
                name=str(value.get("name") or entry.name),
                version=str(value.get("version") or getattr(entry.dist, "version", "unknown")),
                tools=[item for item in (value.get("tools") or []) if isinstance(item, dict)],
                skill_roots=[str(Path(item).expanduser()) for item in (value.get("skill_roots") or [])],
                source=str(getattr(entry, "value", "")),
            ))
        except Exception as exc:
            plugins.append(Plugin(name=str(entry.name), source=str(getattr(entry, "value", "")), error=str(exc)))
    return plugins


def register_plugin_tools(server: Any) -> list[Plugin]:
    plugins = load_plugins()
    for plugin in plugins:
        if plugin.error:
            continue
        for tool in plugin.tools:
            name = str(tool.get("name") or "").strip()
            handler = tool.get("handler")
            if not name or not callable(handler) or name in server.tools:
                continue
            server.register_tool(
                name=name,
                description=str(tool.get("description") or f"Tool from plugin {plugin.name}"),
                parameters=tool.get("parameters") or {"type": "object", "properties": {}},
                handler=handler,
            )
    return plugins


def plugin_skill_roots() -> list[str]:
    return [root for plugin in load_plugins() if not plugin.error for root in plugin.skill_roots]
