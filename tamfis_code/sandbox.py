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
    fail_if_unavailable: bool = False


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

    Linux uses bubblewrap when available. Other platforms fail closed when
    configured to do so, otherwise they retain compatibility but return a
    warning that is surfaced in the tool result.
    """
    direct = (shell, "-lc", command)
    if require_escalated or policy.mode == "danger-full-access":
        return SandboxCommand(direct, False, "none")

    bwrap = shutil.which("bwrap") if sys.platform.startswith("linux") else None
    if not bwrap:
        warning = "OS sandbox unavailable; command ran without kernel isolation"
        if policy.fail_if_unavailable:
            raise RuntimeError(warning)
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
