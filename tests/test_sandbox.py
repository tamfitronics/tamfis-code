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


def test_default_policy_fails_closed_on_linux_without_bwrap(tmp_path):
    """FIX (2026-08-21): SandboxPolicy.fail_if_unavailable now defaults to
    True. Before this, a Linux host without bwrap installed silently ran
    every "sandboxed" command with zero kernel isolation by default."""
    with patch("tamfis_code.sandbox.sys.platform", "linux"), \
         patch("tamfis_code.sandbox.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="sandbox unavailable"):
            build_sandbox_command(
                command="true", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
                policy=SandboxPolicy(),  # no explicit fail_if_unavailable -- exercises the default
            )


def test_explicit_opt_out_still_runs_unsandboxed_with_warning(tmp_path):
    with patch("tamfis_code.sandbox.sys.platform", "linux"), \
         patch("tamfis_code.sandbox.shutil.which", return_value=None):
        result = build_sandbox_command(
            command="true", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
            policy=SandboxPolicy(fail_if_unavailable=False),
        )
    assert result.active is False
    assert result.backend == "unavailable"
    assert result.warning is not None


def test_non_linux_never_fails_closed_even_with_fail_if_unavailable_true(tmp_path):
    """No sandbox-exec/AppContainer backend exists for macOS/Windows yet --
    failing closed there would just break the tool outright, not protect
    anything, so the (now-default-True) flag must have no effect off
    Linux until a real backend exists for those platforms."""
    with patch("tamfis_code.sandbox.sys.platform", "darwin"):
        result = build_sandbox_command(
            command="true", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
            policy=SandboxPolicy(fail_if_unavailable=True),
        )
    assert result.active is False
    assert result.backend == "unavailable"
    assert result.warning is not None
