from pathlib import Path

from tamfis_code.orchestrator.approvals import ApprovalAction, ApprovalBatch, describe_batch


def test_highest_risk_prefers_dangerous_over_medium_and_read_only():
    batch = ApprovalBatch()
    batch.add(ApprovalAction("read_file", {}, "inspect", "read_only"))
    batch.add(ApprovalAction("write_file", {}, "edit", "medium"))
    batch.add(ApprovalAction("execute_command", {"command": "rm -rf /tmp/x"}, "cleanup", "dangerous"))
    assert batch.highest_risk == "dangerous"


def test_requires_prompt_false_when_every_action_is_read_only():
    batch = ApprovalBatch()
    batch.add(ApprovalAction("read_file", {}, "inspect", "read_only"))
    batch.add(ApprovalAction("list_files", {}, "inspect", "read_only"))
    assert batch.requires_prompt is False
    assert batch.risky_actions == []


def test_risky_actions_excludes_read_only():
    batch = ApprovalBatch()
    batch.add(ApprovalAction("read_file", {}, "inspect", "read_only"))
    batch.add(ApprovalAction("write_file", {"path": "a.py"}, "edit", "medium"))
    batch.add(ApprovalAction("execute_command", {"command": "git push"}, "publish", "medium"))
    assert [a.tool_name for a in batch.risky_actions] == ["write_file", "execute_command"]


def test_describe_batch_numbers_only_risky_actions():
    batch = ApprovalBatch()
    batch.add(ApprovalAction("read_file", {"path": "a.py"}, "inspect", "read_only"))
    batch.add(ApprovalAction("write_file", {"path": "b.py"}, "edit", "medium"))
    batch.add(ApprovalAction("execute_command", {"command": "rm -rf /tmp/x"}, "cleanup", "dangerous"))
    text = describe_batch(batch)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1. write_file(")
    assert lines[0].endswith("[medium]")
    assert lines[1].startswith("2. execute_command(")
    assert lines[1].endswith("[dangerous]")


def test_runner_local_batches_same_turn_approvals():
    text = Path("tamfis_code/runner_local.py").read_text(encoding="utf-8")
    assert "_turn_batch = ApprovalBatch()" in text
    assert "describe_batch(_turn_batch)" in text
    assert "_batch_denied_ids" in text
    assert "_batch_approved_once_ids" in text
