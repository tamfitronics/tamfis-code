from pathlib import Path
from unittest.mock import patch

import pytest

from tamfis_code.sandbox import SandboxPolicy, build_sandbox_command
from tamfis_code.safety import RISK_DANGEROUS, classify_tool_call_risk


def test_workspace_write_uses_bubblewrap_and_blocks_network(tmp_path):
    with patch.dict("tamfis_code.sandbox.os.environ", {}, clear=True), patch("tamfis_code.sandbox.shutil.which", return_value="/usr/bin/bwrap"):
        result = build_sandbox_command(
            command="touch result.txt", shell="bash", cwd=tmp_path,
            workspace_root=tmp_path,
            policy=SandboxPolicy(mode="workspace-write", network_access=False),
        )
    assert result.active is True
    assert result.backend == "bubblewrap"
    assert "--unshare-net" in result.argv
    assert ("--bind", str(tmp_path), str(tmp_path)) == tuple(
        result.argv[result.argv.index("--bind"):result.argv.index("--bind") + 3]
    )


def test_read_only_does_not_bind_workspace_writable(tmp_path):
    with patch("tamfis_code.sandbox.shutil.which", return_value="/usr/bin/bwrap"):
        result = build_sandbox_command(
            command="true", shell="sh", cwd=tmp_path, workspace_root=tmp_path,
            policy=SandboxPolicy(mode="read-only", network_access=True),
        )
    assert "--bind" not in result.argv
    assert "--unshare-net" not in result.argv


def test_escalated_command_bypasses_sandbox_only_when_requested(tmp_path):
    result = build_sandbox_command(
        command="true", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
        policy=SandboxPolicy(), require_escalated=True,
    )
    assert result.active is False
    assert result.argv == ("bash", "-lc", "true")
    assert classify_tool_call_risk(
        "execute_command",
        {"command": "echo harmless", "sandbox_permissions": "require_escalated"},
        workspace_root=str(tmp_path),
    ) == RISK_DANGEROUS


def test_missing_required_backend_fails_closed(tmp_path):
    with patch("tamfis_code.sandbox.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="sandbox unavailable"):
            build_sandbox_command(
                command="true", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
                policy=SandboxPolicy(fail_if_unavailable=True),
            )
