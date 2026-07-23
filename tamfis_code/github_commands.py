"""GitHub CLI parity commands.

Tamfis-Code exposes the familiar ``gh`` command surface without reimplementing
GitHub authentication or API semantics. Each command delegates directly to the
installed GitHub CLI, preserves the current working directory, streams PTY/TTY
I/O unchanged, and returns the exact child exit code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable

import click

GITHUB_COMMANDS: tuple[str, ...] = (
    "alias", "api", "auth", "browse", "cache", "co", "codespace",
    "completion", "config", "extension", "gist", "gpg-key", "issue",
    "label", "org", "pr", "project", "release", "repo", "ruleset",
    "run", "search", "secret", "ssh-key", "status", "variable", "workflow",
)


def _gh_binary() -> str:
    binary = shutil.which("gh")
    if not binary:
        raise click.ClickException(
            "GitHub CLI (`gh`) is not installed or is not available in PATH."
        )
    return binary


def run_gh(subcommand: str, args: Iterable[str]) -> None:
    """Execute one gh subcommand with transparent terminal behaviour."""
    command = [_gh_binary(), subcommand, *args]
    completed = subprocess.run(command, cwd=os.getcwd(), check=False)
    if completed.returncode:
        raise click.exceptions.Exit(completed.returncode)


def register_github_commands(root: click.Group) -> None:
    """Register all requested gh-compatible top-level commands."""
    existing = set(root.commands)
    for command_name in GITHUB_COMMANDS:
        # Tamfis-Code owns a richer native command with the same name.
        # Keep that native implementation rather than shadowing it.
        if command_name in existing:
            continue

        def callback(args: tuple[str, ...], _name: str = command_name) -> None:
            run_gh(_name, args)

        command = click.Command(
            name=command_name,
            callback=callback,
            params=[click.Argument(["args"], nargs=-1, type=click.UNPROCESSED)],
            context_settings={
                "ignore_unknown_options": True,
                "allow_extra_args": True,
                "help_option_names": [],
            },
            help=f"Delegate to `gh {command_name}` with full PTY/TTY passthrough.",
        )
        root.add_command(command)
