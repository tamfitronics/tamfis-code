from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from tamfis_code.automation_commands import automation_command, parse_interval
from tamfis_code.cli import cli
from tamfis_code.openhands.automation import Automation, AutomationScheduler, AutomationStore


def test_parse_human_intervals_and_build_safe_subprocess_command(tmp_path: Path):
    assert parse_interval("15m") == 900
    assert parse_interval("2h") == 7200
    item = Automation("review", "review this repository", str(tmp_path), 3600)
    command = automation_command(item)
    assert command[-3:] == ["local", "--agent", "review this repository"]
    assert "accept-edits" in command


def test_automation_cli_add_list_disable_remove(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "state" / "automations.json"
    monkeypatch.setattr("tamfis_code.automation_commands.AUTOMATIONS_PATH", store_path)
    runner = CliRunner()
    added = runner.invoke(
        cli,
        ["--cwd", str(tmp_path), "automations", "add", "nightly", "run tests", "--every", "1h"],
    )
    assert added.exit_code == 0, added.output
    listed = runner.invoke(cli, ["--cwd", str(tmp_path), "automations", "list"])
    assert listed.exit_code == 0
    assert "nightly" in listed.output and "3600s" in listed.output
    disabled = runner.invoke(cli, ["automations", "disable", "nightly"])
    assert disabled.exit_code == 0 and "disabled" in disabled.output
    removed = runner.invoke(cli, ["automations", "remove", "nightly"])
    assert removed.exit_code == 0
    assert AutomationStore(store_path).load() == []


async def _noop(_item):
    return None


def test_scheduler_runs_due_once_and_reserves_next_slot(tmp_path: Path):
    import asyncio

    store = AutomationStore(tmp_path / "automations.json")
    store.save([Automation("hourly", "run tests", str(tmp_path), 3600, next_run=100)])
    scheduler = AutomationScheduler(store, _noop)
    assert asyncio.run(scheduler.run_due(now=101)) == ["hourly"]
    assert asyncio.run(scheduler.run_due(now=102)) == []
    loaded = store.load()[0]
    assert loaded.last_run == 101
    assert loaded.next_run == 3701
    assert loaded.last_status == "completed"


def test_scheduler_records_failure_without_crashing_service(tmp_path: Path):
    import asyncio

    async def fail(_item):
        raise RuntimeError("provider unavailable")

    store = AutomationStore(tmp_path / "automations.json")
    store.save([Automation("hourly", "run tests", str(tmp_path), 3600)])
    scheduler = AutomationScheduler(store, fail)
    assert asyncio.run(scheduler.run_due(now=10)) == ["hourly"]
    loaded = store.load()[0]
    assert loaded.last_status == "failed"
    assert loaded.last_error == "provider unavailable"
