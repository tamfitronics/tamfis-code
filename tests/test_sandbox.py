import os
import pwd
from pathlib import Path
from unittest.mock import patch

import pytest

from tamfis_code.sandbox import SandboxPolicy, build_sandbox_command, resolve_workspace_owner
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


# ── Drop-to-workspace-owner privileges ──────────────────────────────────
#
# Regression coverage for a live-confirmed incident (2026-09): this process
# runs as root with no privilege-dropping anywhere in execute_command, so
# an ordinary `npm run build` inside a workspace owned by a different,
# dedicated service account (e.g. /home/tamfisseo, owned by user
# tamfisseo) left new build output/.git state root-owned -- silently
# re-breaking that project's ownership every time an agent touched it, and
# forcing the NEXT ordinary command in that workspace to fail on the
# permission mismatch and request require_escalated (root) just to push
# through, which wrote MORE root-owned files, repeating the cycle.

class _FakeStat:
    def __init__(self, uid):
        self.st_uid = uid


def _fake_pwent(name="tamfisseo", uid=1500, gid=1500, home="/home/tamfisseo"):
    return pwd.struct_passwd((name, "x", uid, gid, "", home, "/bin/bash"))


def test_resolve_workspace_owner_returns_none_for_root_owned_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: _FakeStat(0))
    assert resolve_workspace_owner(tmp_path) is None


def test_resolve_workspace_owner_returns_none_for_unresolvable_uid(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: _FakeStat(999999))
    with patch("tamfis_code.sandbox.pwd.getpwuid", side_effect=KeyError):
        assert resolve_workspace_owner(tmp_path) is None


def test_resolve_workspace_owner_returns_the_real_account(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: _FakeStat(1500))
    with patch("tamfis_code.sandbox.pwd.getpwuid", return_value=_fake_pwent()):
        owner = resolve_workspace_owner(tmp_path)
    assert owner is not None
    assert owner.uid == 1500 and owner.gid == 1500
    assert owner.name == "tamfisseo"
    assert owner.home == "/home/tamfisseo"


def test_bwrap_path_drops_to_the_workspace_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: _FakeStat(1500))
    with patch("tamfis_code.sandbox.pwd.getpwuid", return_value=_fake_pwent()), \
         patch("tamfis_code.sandbox.os.geteuid", return_value=0), \
         patch("tamfis_code.sandbox.shutil.which", return_value="/usr/bin/bwrap"):
        result = build_sandbox_command(
            command="npm run build", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
            policy=SandboxPolicy(mode="workspace-write"),
        )
    # Live-confirmed (2026-09-10): bwrap's own --uid/--gid, run as real
    # root, only fakes the identity *inside* bwrap's new user namespace --
    # files written through a --bind of a real host path still land
    # owned by root (uid_map showed the inner uid mapping straight back
    # to outer uid 0). The only way to get genuinely-owned files is to
    # run bwrap itself as the target user, hence the whole bwrap
    # invocation is wrapped in runuser rather than passed --uid/--gid.
    assert "--uid" not in result.argv
    assert "--gid" not in result.argv
    assert "--unshare-user" not in result.argv
    # shutil.which is mocked to return "/usr/bin/bwrap" for any name here,
    # so both the runuser wrapper and the wrapped bwrap resolve to it.
    assert result.argv[:4] == ("/usr/bin/bwrap", "-u", "tamfisseo", "--")
    assert result.argv[4] == "/usr/bin/bwrap"
    assert result.env_overrides == {"HOME": "/home/tamfisseo", "USER": "tamfisseo", "LOGNAME": "tamfisseo"}
    assert "drop-to-tamfisseo" in result.backend
    assert result.warning is None


def test_runuser_fallback_drops_privileges_when_bwrap_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: _FakeStat(1500))
    with patch("tamfis_code.sandbox.pwd.getpwuid", return_value=_fake_pwent()), \
         patch("tamfis_code.sandbox.os.geteuid", return_value=0), \
         patch("tamfis_code.sandbox.shutil.which", side_effect=lambda name: "/usr/sbin/runuser" if name == "runuser" else None):
        result = build_sandbox_command(
            command="npm run build", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
            policy=SandboxPolicy(fail_if_unavailable=False),
        )
    assert result.argv[:3] == ("runuser", "-u", "tamfisseo")
    assert result.argv[3] == "--"
    assert result.env_overrides == {"HOME": "/home/tamfisseo", "USER": "tamfisseo", "LOGNAME": "tamfisseo"}


def test_require_escalated_never_drops_privileges_even_with_a_known_owner(tmp_path, monkeypatch):
    """The explicit "this needs root" opt-out must be unaffected by
    ownership-based privilege dropping -- a model/user that deliberately
    asked for escalation still gets it."""
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: _FakeStat(1500))
    with patch("tamfis_code.sandbox.pwd.getpwuid", return_value=_fake_pwent()), \
         patch("tamfis_code.sandbox.os.geteuid", return_value=0):
        result = build_sandbox_command(
            command="chown -R tamfisseo:tamfisseo .", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
            policy=SandboxPolicy(), require_escalated=True,
        )
    assert result.argv == ("bash", "-lc", "chown -R tamfisseo:tamfisseo .")
    assert result.env_overrides == {}


def test_no_privilege_drop_when_this_process_is_not_root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: _FakeStat(1500))
    with patch("tamfis_code.sandbox.pwd.getpwuid", return_value=_fake_pwent()), \
         patch("tamfis_code.sandbox.os.geteuid", return_value=1500), \
         patch("tamfis_code.sandbox.shutil.which", return_value="/usr/bin/bwrap"):
        result = build_sandbox_command(
            command="npm run build", shell="bash", cwd=tmp_path, workspace_root=tmp_path,
            policy=SandboxPolicy(mode="workspace-write"),
        )
    assert "--uid" not in result.argv
    assert result.env_overrides == {}
    assert result.backend == "bubblewrap"
