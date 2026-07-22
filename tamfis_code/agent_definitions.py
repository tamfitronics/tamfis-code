"""Declarative subagent types -- Claude Code's `.claude/agents/*.md`
equivalent. Delegation itself already existed (agents.py's
DelegatedCodingAgent, swarm.py's fan-out) but only as ad-hoc task strings:
every delegated sub-task shared the same model/provider and got no
specialised instructions of its own. This module lets a user define a
NAMED subagent -- its own system-prompt prefix and, optionally, its own
model/provider -- once, in a file, then reference it by name from
`/delegate --agent`, `/swarm --agent`, or the model-callable
`delegate_parallel_tasks` tool's per-task `agent_type` field.

Discovery mirrors hooks.py/custom_commands.py: `<config_dir>/agents/*.md`
(every session) and `<project_root>/.tamfis/agents/*.md` (one project); a
project definition with the same name REPLACES the user one (same
override-not-merge precedent as custom_commands.py, for the same reason --
a project-specific subagent should win outright over a general personal
one sharing its name).

File shape: filename (minus `.md`) is the agent type name. Frontmatter
sets `description` and optionally `model`/`provider` (raw strings --
resolving "provider" to a real ProviderType, and validating the model
against a real catalog, is the caller's job, not this module's, to avoid
a tools-config module depending on providers.py). The rest of the file is
a system-prompt prefix prepended to every sub-task delegated to this
agent type, ahead of the sub-task's own objective.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CONFIG_DIR

USER_AGENTS_DIR = CONFIG_DIR / "agents"
PROJECT_AGENTS_RELATIVE = Path(".tamfis") / "agents"
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    system_prompt: str
    model: Optional[str]
    provider: Optional[str]
    source: str


def _parse_agent_file(path: Path, source: str) -> Optional[AgentDefinition]:
    name = path.stem
    if not _NAME_RE.match(name):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    description = ""
    model: Optional[str] = None
    provider: Optional[str] = None
    body = text
    match = _FRONTMATTER_RE.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "description":
                description = value
            elif key == "model" and value:
                model = value
            elif key == "provider" and value:
                provider = value
        body = text[match.end():]
    body = body.strip()
    if not body:
        return None
    return AgentDefinition(
        name=name, description=description or f"custom subagent ({path.name})",
        system_prompt=body, model=model, provider=provider, source=source,
    )


def _load_agents_dir(directory: Path, source: str) -> dict[str, AgentDefinition]:
    if not directory.is_dir():
        return {}
    definitions: dict[str, AgentDefinition] = {}
    for path in sorted(directory.glob("*.md")):
        definition = _parse_agent_file(path, source)
        if definition is not None:
            definitions[definition.name] = definition
    return definitions


def load_agent_definitions(project_root: Optional[str] = None) -> dict[str, AgentDefinition]:
    """Read fresh each call (not cached) -- matches hooks.py/
    custom_commands.py's freshness contract, so a new/edited agent
    definition is usable on the very next delegation without a restart."""
    definitions = _load_agents_dir(USER_AGENTS_DIR, "user config")
    if project_root is not None:
        definitions.update(_load_agents_dir(Path(project_root) / PROJECT_AGENTS_RELATIVE, "project config"))
    return definitions
