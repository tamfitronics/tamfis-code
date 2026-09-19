"""Plugin discovery for Tamfis Code: Python entry points and JSON manifests.

Manifests are third-party files (``kimi.plugin.json``, ``tamfis.plugin.json``,
``plugin.json``) written by people and tools this project does not control, so
manifest handling is explicitly *degraded-mode by design*: an unreadable file, a
badly-shaped document, an unknown schema version, or one broken tool entry
never raises, never aborts discovery, and never stops the agent. The failure is
recorded on the Plugin (``error``) or in the diagnostics report, and the agent
continues with its built-in tools.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


ENTRYPOINT_GROUP = "tamfis_code.plugins"

# Checked in this order inside each plugin directory.
MANIFEST_FILENAMES = ("tamfis.plugin.json", "kimi.plugin.json", "plugin.json")
# The manifest schema versions this build understands. An unknown version is a
# degraded plugin, not a crash and not a silent skip.
SUPPORTED_MANIFEST_SCHEMAS = (1, "1")


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


def register_plugin_tools(server: Any, *, include_manifests: bool = True) -> list[Plugin]:
    plugins = load_all_plugins() if include_manifests else load_plugins()
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
    return [root for plugin in load_all_plugins() if not plugin.error for root in plugin.skill_roots]


# --------------------------------------------------------------------------
# JSON manifest plugins (degraded mode)
# --------------------------------------------------------------------------


@dataclass
class PluginDiagnostics:
    """What discovery found and what it had to give up on -- inspectable for
    debugging (``tamfis-code doctor``-style) without any of it being fatal."""

    loaded: list[str] = field(default_factory=list)
    degraded: list[dict[str, str]] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)

    @property
    def degraded_mode(self) -> bool:
        return bool(self.degraded)

    def summary(self) -> str:
        if not self.degraded:
            return f"{len(self.loaded)} plugin(s) loaded"
        return (
            f"{len(self.loaded)} plugin(s) loaded; {len(self.degraded)} degraded "
            f"(built-in tools still available)"
        )


def default_manifest_dirs(workspace_root: str | Path | None = None) -> list[Path]:
    """Where manifests are looked for: the user config dir, the Kimi-compatible
    ``KIMI_CODE_HOME`` (that ecosystem already ships JSON plugins), and the
    project's own ``.tamfis/plugins``."""
    directories: list[Path] = []
    try:
        from . import state as local_state

        directories.append(Path(getattr(local_state, "CONFIG_DIR")) / "plugins")
    except Exception:
        pass
    kimi_home = os.environ.get("KIMI_CODE_HOME")
    if kimi_home:
        directories.append(Path(kimi_home).expanduser() / "plugins")
    if workspace_root is not None:
        directories.append(Path(workspace_root) / ".tamfis" / "plugins")
    return directories

def discover_manifest_paths(
    directories: Iterable[str | Path] | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> list[Path]:
    """Every manifest file under the given directories (or the defaults),
    deduplicated by resolved path. Unreadable directories are skipped."""
    roots = list(directories) if directories is not None else default_manifest_dirs(workspace_root)
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            base = Path(root).expanduser()
            if not base.is_dir():
                continue
        except (OSError, ValueError):
            continue
        for filename in MANIFEST_FILENAMES:
            candidate = base / filename
            try:
                if not candidate.is_file():
                    continue
                key = str(candidate.resolve())
            except (OSError, ValueError, RuntimeError):
                continue
            if key in seen:
                continue
            seen.add(key)
            found.append(candidate)
    return found


def _manifest_handler(raw: dict[str, Any]) -> tuple[Any, str]:
    """Resolve a manifest tool's handler. Returns (callable, failure_reason)."""
    handler = raw.get("handler")
    if callable(handler):
        return handler, ""
    entrypoint = str(raw.get("entrypoint") or "").strip()
    if entrypoint:
        module_name, _, attribute = entrypoint.partition(":")
        if not module_name or not attribute:
            return None, f"entrypoint {entrypoint!r} must be 'module:attribute'"
        try:
            module = importlib.import_module(module_name)
            resolved = getattr(module, attribute)
        except Exception as exc:
            return None, f"entrypoint {entrypoint!r} could not be imported ({type(exc).__name__}: {exc})"
        if not callable(resolved):
            return None, f"entrypoint {entrypoint!r} is not callable"
        return resolved, ""
    if raw.get("command"):
        return None, "command-style tools need an MCP bridge (not supported in-process)"
    return None, "no handler, entrypoint, or command"


def load_manifest_plugin(path: str | Path) -> tuple[Plugin, list[str]]:
    """Load one manifest. Always returns a Plugin (never raises) plus the list
    of degradation reasons for whatever part of it could not be used."""
    source = str(path)
    try:
        raw_text = Path(path).read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw_text)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return Plugin(name=Path(path).stem, source=source, error=f"unreadable manifest: {type(exc).__name__}: {exc}"), []
    if not isinstance(data, dict):
        return Plugin(name=Path(path).stem, source=source, error="manifest root must be a JSON object"), []

    notes: list[str] = []
    schema = data.get("schema", data.get("manifest_version", 1))
    if schema not in SUPPORTED_MANIFEST_SCHEMAS:
        notes.append(f"unknown manifest schema {schema!r}; attempting to load anyway")

    name = str(data.get("name") or Path(path).stem)
    tools: list[dict[str, Any]] = []
    for raw_tool in data.get("tools") or []:
        if not isinstance(raw_tool, dict):
            notes.append("ignored a tool entry that is not an object")
            continue
        tool_name = str(raw_tool.get("name") or "").strip()
        if not tool_name:
            notes.append("ignored a tool entry with no name")
            continue
        resolved, reason = _manifest_handler(raw_tool)
        if resolved is None:
            notes.append(f"tool {tool_name!r} skipped: {reason}")
            continue
        tools.append({
            "name": tool_name,
            "description": str(raw_tool.get("description") or f"Tool from manifest plugin {name}"),
            "parameters": raw_tool.get("parameters") or {"type": "object", "properties": {}},
            "handler": resolved,
        })

    skill_roots: list[str] = []
    for raw_root in data.get("skill_roots") or []:
        try:
            skill_roots.append(str(Path(str(raw_root)).expanduser()))
        except (OSError, ValueError, TypeError):
            notes.append(f"ignored unusable skill root {raw_root!r}")

    version = str(data.get("version") or "unknown")
    plugin = Plugin(name=name, version=version, tools=tools, skill_roots=skill_roots, source=source)
    return plugin, notes


def load_manifest_plugins(
    directories: Iterable[str | Path] | None = None,
    *,
    workspace_root: str | Path | None = None,
    diagnostics: PluginDiagnostics | None = None,
) -> list[Plugin]:
    """Load every discoverable manifest. Never raises; failures land in the
    Plugin's ``error`` and/or the diagnostics report."""
    report = diagnostics if diagnostics is not None else PluginDiagnostics()
    plugins: list[Plugin] = []
    for path in discover_manifest_paths(directories, workspace_root=workspace_root):
        report.searched.append(str(path))
        try:
            plugin, notes = load_manifest_plugin(path)
        except Exception as exc:  # pragma: no cover - defensive: discovery must not stop
            plugins.append(Plugin(name=path.stem, source=str(path), error=f"{type(exc).__name__}: {exc}"))
            report.degraded.append({"source": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        plugins.append(plugin)
        if plugin.error:
            report.degraded.append({"source": str(path), "reason": plugin.error})
        else:
            report.loaded.append(plugin.name)
        for note in notes:
            report.degraded.append({"source": str(path), "reason": note})
    return plugins


def load_all_plugins(
    *, workspace_root: str | Path | None = None, manifest_dirs: Iterable[str | Path] | None = None,
    diagnostics: PluginDiagnostics | None = None,
) -> list[Plugin]:
    """Entry-point plugins plus manifest plugins, manifests degrading instead
    of failing. The built-in tool set is never affected by either."""
    report = diagnostics if diagnostics is not None else PluginDiagnostics()
    plugins = list(load_plugins())
    for plugin in plugins:
        if plugin.error:
            report.degraded.append({"source": plugin.source or plugin.name, "reason": plugin.error})
        else:
            report.loaded.append(plugin.name)
    plugins.extend(load_manifest_plugins(manifest_dirs, workspace_root=workspace_root, diagnostics=report))
    return plugins


def plugin_diagnostics(workspace_root: str | Path | None = None) -> PluginDiagnostics:
    """Discovery report for debugging: what loaded, what degraded, and why."""
    report = PluginDiagnostics()
    load_all_plugins(workspace_root=workspace_root, diagnostics=report)
    return report
