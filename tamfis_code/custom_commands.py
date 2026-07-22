"""User-defined custom slash commands -- Claude Code/Codex-style: drop a
markdown file into a commands directory and get a new `/<name>` command in
the interactive REPL, with no code changes required.

Discovery: `<config_dir>/commands/*.md` (every session, same platform-native
location config.py already resolves for hooks.toml/config.toml) and
`<project_root>/.tamfis/commands/*.md` (one project). Unlike hooks.py
(where user and project hooks both fire), a project command with the same
name as a user command REPLACES it -- a project-specific command should
win outright over a general-purpose personal one sharing its name, the
same way a project .tamfis/config.toml layer already overrides user config
in config.py's own precedence.

File shape: the filename (minus `.md`) is the command name (`foo.md` ->
`/foo`). An optional frontmatter block sets `description`:

    ---
    description: Review a diff for security issues
    ---
    Review the following diff for security issues, focusing on injection,
    auth, and secrets handling: $ARGUMENTS

The rest of the file is the prompt template sent as the AI objective when
the command runs. `$ARGUMENTS` is replaced with whatever the user typed
after the command name; a template with no `$ARGUMENTS` placeholder still
gets the typed text appended on a new line, so extra context the user
provides is never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CONFIG_DIR

USER_COMMANDS_DIR = CONFIG_DIR / "commands"
PROJECT_COMMANDS_RELATIVE = Path(".tamfis") / "commands"
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class CustomCommand:
    name: str
    description: str
    template: str
    source: str


def _parse_command_file(path: Path, source: str) -> Optional[CustomCommand]:
    name = path.stem
    if not _NAME_RE.match(name):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    description = ""
    body = text
    match = _FRONTMATTER_RE.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == "description":
                description = value.strip()
        body = text[match.end():]
    body = body.strip()
    if not body:
        return None
    return CustomCommand(
        name=name, description=description or f"custom command ({path.name})",
        template=body, source=source,
    )


def _load_commands_dir(directory: Path, source: str) -> dict[str, CustomCommand]:
    if not directory.is_dir():
        return {}
    commands: dict[str, CustomCommand] = {}
    for path in sorted(directory.glob("*.md")):
        command = _parse_command_file(path, source)
        if command is not None:
            commands[command.name] = command
    return commands


def load_custom_commands(project_root: Optional[str] = None) -> dict[str, CustomCommand]:
    """Read fresh each call (not cached) so adding/editing a command file
    takes effect on the next REPL turn without a restart -- matches
    hooks.load_hooks's same freshness contract."""
    commands = _load_commands_dir(USER_COMMANDS_DIR, "user config")
    if project_root is not None:
        commands.update(_load_commands_dir(Path(project_root) / PROJECT_COMMANDS_RELATIVE, "project config"))
    return commands


def expand_custom_command(command: CustomCommand, arguments: str) -> str:
    if "$ARGUMENTS" in command.template:
        return command.template.replace("$ARGUMENTS", arguments)
    if arguments:
        return f"{command.template}\n\n{arguments}"
    return command.template
