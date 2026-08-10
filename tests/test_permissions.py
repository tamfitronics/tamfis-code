import tempfile
from pathlib import Path

from tamfis_code.permissions import decide_permission


def test_deny_precedes_ask_and_allow():
    decision = decide_permission(
        "execute_command", {"command": "git push origin main"}, workspace_root="/repo",
        allow=["execute_command(git *)"], ask=["execute_command(git push *)"],
        deny=["execute_command(*push*)"],
    )
    assert decision is not None
    assert decision.action == "deny"


def test_scoped_allow_rule_matches_tool_target():
    decision = decide_permission(
        "execute_command", {"command": "pytest -q"}, workspace_root="/repo",
        allow=["execute_command(pytest *)"],
    )
    assert decision is not None
    assert decision.action == "allow"
    assert decide_permission(
        "execute_command", {"command": "npm publish"}, workspace_root="/repo",
        allow=["execute_command(pytest *)"],
    ) is None


def test_scoped_file_rule_matches_absolute_workspace_path():
    decision = decide_permission(
        "edit_file", {"path": "/repo/src/app.py"}, workspace_root="/repo",
        allow=["edit_file(src/*)"],
    )
    assert decision is not None
    assert decision.action == "allow"


def test_protected_path_requires_approval_even_with_allow_rule():
    with tempfile.TemporaryDirectory() as root:
        protected = Path(root) / ".git" / "config"
        decision = decide_permission(
            "write_file", {"path": str(protected)}, workspace_root=root,
            allow=["write_file(*)"],
        )
    assert decision is not None
    assert decision.action == "ask"
    assert decision.protected


def test_explicit_deny_still_wins_for_protected_path():
    decision = decide_permission(
        "edit_file", {"path": ".env"}, workspace_root="/repo",
        deny=["edit_file(*)"], allow=["edit_file(*)"],
    )
    assert decision is not None
    assert decision.action == "deny"


def test_shell_command_mentioning_protected_path_requires_approval():
    decision = decide_permission(
        "execute_command", {"command": "printf secret > .env.production"},
        workspace_root="/repo", allow=["execute_command(*)"],
    )
    assert decision is not None
    assert decision.action == "ask"
    assert decision.protected
