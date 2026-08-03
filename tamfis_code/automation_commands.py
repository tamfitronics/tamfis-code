"""Top-level CLI for creating and running Tamfis Code automations."""
from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import APPROVAL_MODES, CONFIG_DIR
from .openhands.automation import Automation, AutomationScheduler, AutomationStore


AUTOMATIONS_PATH = CONFIG_DIR / "automations.json"
_INTERVAL = re.compile(r"^(?P<number>\d+(?:\.\d+)?)(?P<unit>[smhd]?)$")


def parse_interval(value: str) -> float:
    match = _INTERVAL.fullmatch(value.strip().lower())
    if not match:
        raise click.BadParameter("use seconds or a suffix such as 15m, 2h, or 1d")
    multiplier = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group("unit")]
    seconds = float(match.group("number")) * multiplier
    if seconds < 60:
        raise click.BadParameter("the minimum interval is 60 seconds")
    return seconds


def _store() -> AutomationStore:
    return AutomationStore(AUTOMATIONS_PATH)


def automation_command(item: Automation) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tamfis_code",
        "--cwd",
        item.workspace,
        "--approval",
        item.approval_policy,
        "local",
        "--agent",
        item.objective,
    ]


async def run_automation(item: Automation) -> None:
    process = await asyncio.create_subprocess_exec(*automation_command(item))
    return_code = await process.wait()
    if return_code:
        raise RuntimeError(f"automation {item.name!r} exited with status {return_code}")


def _date(value: float | None) -> str:
    return "-" if value is None else datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


@click.group("automations")
def automations_group() -> None:
    """Create, inspect, run, and serve durable scheduled agent tasks."""


@automations_group.command("list")
def list_automations() -> None:
    table = Table(show_header=True, header_style="bold")
    for name in ("Name", "State", "Last result", "Every", "Last run", "Next run", "Workspace"):
        table.add_column(name)
    for item in _store().load():
        table.add_row(
            item.name,
            "enabled" if item.enabled else "disabled",
            item.last_status or "-",
            f"{item.interval_seconds:g}s",
            _date(item.last_run),
            _date(item.next_run),
            item.workspace,
        )
    Console().print(table)


@automations_group.command("add")
@click.argument("name")
@click.argument("objective")
@click.option("--every", required=True, help="Interval in seconds or with m/h/d suffix (for example 30m).")
@click.option("--workspace", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
@click.option("--approval", type=click.Choice(APPROVAL_MODES), default="accept-edits", show_default=True)
@click.option("--replace", is_flag=True, help="Replace a same-named automation while preserving its ID.")
@click.pass_context
def add_automation(
    ctx: click.Context,
    name: str,
    objective: str,
    every: str,
    workspace: Path | None,
    approval: str,
    replace: bool,
) -> None:
    root = workspace or ctx.find_root().obj["workspace_root"]
    item = Automation(name, objective, str(root), parse_interval(every), approval_policy=approval)
    try:
        _store().upsert(item, replace=replace)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved automation {item.name} ({item.id})")


def _set_enabled(name: str, enabled: bool) -> None:
    try:
        item = _store().update_enabled(name, enabled)
    except KeyError as exc:
        raise click.ClickException(f"unknown automation: {name}") from exc
    click.echo(f"{item.name}: {'enabled' if enabled else 'disabled'}")


@automations_group.command("enable")
@click.argument("name")
def enable_automation(name: str) -> None:
    _set_enabled(name, True)


@automations_group.command("disable")
@click.argument("name")
def disable_automation(name: str) -> None:
    _set_enabled(name, False)


@automations_group.command("remove")
@click.argument("name")
def remove_automation(name: str) -> None:
    try:
        item = _store().remove(name)
    except KeyError as exc:
        raise click.ClickException(f"unknown automation: {name}") from exc
    click.echo(f"Removed automation {item.name}")


@automations_group.command("run")
@click.argument("name")
def run_automation_now(name: str) -> None:
    try:
        item = _store().get(name)
    except KeyError as exc:
        raise click.ClickException(f"unknown automation: {name}") from exc
    try:
        asyncio.run(run_automation(item))
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@automations_group.command("serve")
@click.option("--poll", type=click.FloatRange(min=0.1), default=1.0, show_default=True)
def serve_automations(poll: float) -> None:
    """Run the scheduler in the foreground (suitable for systemd/launchd)."""
    scheduler = AutomationScheduler(_store(), run_automation)
    try:
        asyncio.run(scheduler.run_forever(poll=poll))
    except KeyboardInterrupt:
        scheduler.stop()


def register_automation_commands(root: click.Group) -> None:
    root.add_command(automations_group)
