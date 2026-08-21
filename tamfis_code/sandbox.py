"""OS-enforced command sandboxing for the local agent runtime."""

from __future__ import annotations

import shutil
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path


SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")


@dataclass(frozen=True)
class SandboxPolicy:
    mode: str = "workspace-write"
    network_access: bool = False
    writable_roots: tuple[str, ...] = field(default_factory=tuple)
    # Defaults to fail-closed on Linux (see build_sandbox_command's
    # docstring for why this has no effect on macOS/Windows, which have no
    # sandbox backend to fail closed into yet). Previously defaulted to
    # False, meaning a Linux host without bwrap installed silently ran
    # every "sandboxed" command with zero kernel isolation.
    fail_if_unavailable: bool = True


@dataclass(frozen=True)
class SandboxCommand:
    argv: tuple[str, ...]
    active: bool
    backend: str
    warning: str | None = None


def build_sandbox_command(
    *, command: str, shell: str, cwd: Path, workspace_root: Path,
    policy: SandboxPolicy, require_escalated: bool = False,
) -> SandboxCommand:
    """Return the executable argv and an auditable description of isolation.

    Linux uses bubblewrap when available. `policy.fail_if_unavailable`
    (default True as of 2026-08-21 -- previously defaulted to False,
    meaning every command on a Linux host without bwrap installed silently
    ran with zero kernel isolation) only applies on Linux: there is no
    sandbox-exec/AppContainer backend for macOS/Windows yet, so failing
    closed there would just break the tool outright rather than protect
    anything -- those platforms always retain the permissive fallback
    (with its warning still surfaced in the tool result) until a real
    backend exists for them.
    """
    direct = (shell, "-lc", command)
    if require_escalated or policy.mode == "danger-full-access":
        return SandboxCommand(direct, False, "none")

    is_linux = sys.platform.startswith("linux")
    bwrap = shutil.which("bwrap") if is_linux else None
    if not bwrap:
        warning = "OS sandbox unavailable; command ran without kernel isolation"
        if is_linux and policy.fail_if_unavailable:
            raise RuntimeError(
                f"{warning}. Install bubblewrap (`apt install bubblewrap` / "
                f"`dnf install bubblewrap`) for real kernel-level command "
                f"isolation, or set sandbox_fail_if_unavailable: false in "
                f"config (or TAMFIS_CODE_SANDBOX_FAIL_IF_UNAVAILABLE=false) "
                f"to run commands unsandboxed instead of failing here."
            )
        return SandboxCommand(direct, False, "unavailable", warning)

    argv = [
        bwrap, "--die-with-parent", "--new-session", "--ro-bind", "/", "/",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    ]
    # Nested managed runtimes (including Codex) can already enforce a host
    # network deny while prohibiting creation of another network namespace.
    # Preserve that stronger outer boundary instead of making every command
    # fail at bubblewrap startup.
    outer_network_denied = str(os.environ.get("CODEX_SANDBOX_NETWORK_DISABLED") or "").lower() in {"1", "true", "yes"}
    if not policy.network_access and not outer_network_denied:
        argv.append("--unshare-net")

    writable: list[Path] = []
    if policy.mode == "workspace-write":
        writable.append(workspace_root.resolve())
    writable.extend(Path(item).expanduser().resolve() for item in policy.writable_roots)
    seen: set[Path] = set()
    for root in writable:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        # A tmpfs hides host-side descendants. Recreate the mount target
        # before overlaying a workspace that itself lives under /tmp.
        if root == Path("/tmp") or Path("/tmp") in root.parents:
            current = Path("/tmp")
            for component in root.relative_to("/tmp").parts:
                current /= component
                argv.extend(("--dir", str(current)))
        argv.extend(("--bind", str(root), str(root)))

    argv.extend(("--chdir", str(cwd), shell, "-lc", command))
    backend = "bubblewrap+outer-network-policy" if outer_network_denied and not policy.network_access else "bubblewrap"
    return SandboxCommand(tuple(argv), True, backend)
