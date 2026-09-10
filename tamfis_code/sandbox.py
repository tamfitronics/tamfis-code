"""OS-enforced command sandboxing for the local agent runtime."""

from __future__ import annotations

import pwd
import shutil
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional


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
    # Merged into the child's environment by the caller (mcp.py's
    # _execute_command) alongside its own env= dict. Only ever set when
    # this command is running as a dropped-to workspace owner (see
    # resolve_workspace_owner below) -- corrects HOME/USER/LOGNAME to that
    # account instead of leaking root's, which would otherwise point tools
    # like npm/git at the wrong home directory for config/cache lookups
    # even though the process uid itself is now correct.
    env_overrides: dict[str, str] = field(default_factory=dict)


class WorkspaceOwner(NamedTuple):
    uid: int
    gid: int
    name: str
    home: str


# This process (tamfis-code-server) runs as root -- same privilege tier as
# any other root-run agent working on this host. Confirmed live (2026-09):
# with no privilege-dropping anywhere in the execute_command path, running
# ordinary project commands (a plain `npm run build`) inside a workspace
# OWNED BY A DIFFERENT, dedicated service account (e.g. /home/tamfisseo,
# owned by user tamfisseo, running as User=tamfisseo in its own systemd
# unit) left new build output/config/.git state root-owned -- silently
# re-breaking that project's ownership every time an agent touched it,
# forcing the NEXT ordinary command to fail on a permission mismatch and
# request require_escalated (root) just to push through it, which then
# wrote MORE root-owned files, repeating the cycle. A careful human
# operator (or another agent with the same root access) would run project
# commands as the project's own owning account rather than as root by
# default, escalating only when a command genuinely needs it -- that is
# what resolve_workspace_owner + the --uid/--gid (or runuser) wiring below
# does automatically, with require_escalated remaining the explicit,
# unchanged opt-out for commands that really do need root (installing
# system packages, chown, systemctl, etc).
def resolve_workspace_owner(workspace_root: Path) -> Optional[WorkspaceOwner]:
    """The real, existing, non-root account that owns `workspace_root`, or
    None when there's nothing safe to drop privileges to (root-owned
    workspace, or an owning uid with no resolvable passwd entry -- e.g. a
    container base image's leftover numeric-only ownership)."""
    try:
        uid = workspace_root.stat().st_uid
    except OSError:
        return None
    if uid == 0:
        return None
    try:
        pw = pwd.getpwuid(uid)
    except KeyError:
        return None
    return WorkspaceOwner(uid=pw.pw_uid, gid=pw.pw_gid, name=pw.pw_name, home=pw.pw_dir)


def _can_drop_privileges() -> bool:
    """Only root can become another uid without already being that uid."""
    try:
        return os.geteuid() == 0
    except AttributeError:
        return False  # Windows has no geteuid; nothing to drop there.


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
        # Explicit "this needs root" escape hatch (a model-set
        # sandbox_permissions="require_escalated", or a policy deliberately
        # configured for unrestricted access) -- never drop privileges
        # here regardless of who owns workspace_root.
        return SandboxCommand(direct, False, "none")

    owner = resolve_workspace_owner(workspace_root) if _can_drop_privileges() else None
    env_overrides = (
        {"HOME": owner.home, "USER": owner.name, "LOGNAME": owner.name} if owner else {}
    )

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
        if owner and shutil.which("runuser"):
            # No kernel sandbox available, but still don't run an ordinary
            # command as root inside someone else's project just because
            # bubblewrap happens to be missing -- see resolve_workspace_owner's
            # comment for the ownership-corruption cycle this closes.
            direct = ("runuser", "-u", owner.name, "--", *direct)
            return SandboxCommand(direct, False, "unavailable", warning, env_overrides)
        return SandboxCommand(direct, False, "unavailable", warning)

    argv = [
        bwrap, "--die-with-parent", "--new-session", "--ro-bind", "/", "/",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    ]
    # NOTE: privilege dropping is done by wrapping the *entire* bwrap
    # invocation in `runuser -u <owner> --` below, not with bwrap's own
    # --uid/--gid flags. Live-tested (2026-09-10): when bwrap itself runs
    # as real root and is told --uid 1005 --gid 1005 (which requires
    # --unshare-user), the child's `id` genuinely reports uid=1005 inside
    # the sandbox, but that's only a fake identity in bwrap's own new user
    # namespace -- `cat /proc/self/uid_map` inside showed "1005 0 1",
    # i.e. inner uid 1005 maps back to real/outer uid 0. Any file the
    # child then wrote through a --bind of a real host path (not a
    # namespace-private tmpfs) was still recorded on disk as owned by
    # root, because the kernel resolves file ownership against the real,
    # initial-namespace credential -- which was never actually changed.
    # `id`-based verification alone would have missed this; only checking
    # the resulting file's on-disk owner caught it. Running bwrap itself
    # as the target user via runuser gives it a genuine, non-fake uid --
    # bwrap then creates its *own* unprivileged user namespace the normal
    # way (the same mechanism Flatpak relies on to sandbox as a regular
    # user), and files land with correct real ownership.
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
    warning = None
    if owner:
        runuser = shutil.which("runuser")
        if runuser:
            # Real credential change BEFORE bwrap runs -- see the long
            # comment above for why bwrap's own --uid/--gid can't do this.
            argv = [runuser, "-u", owner.name, "--", *argv]
            backend += f"+drop-to-{owner.name}"
        else:
            # Can't actually drop privileges here (see the comment above:
            # bwrap's --uid/--gid alone is a no-op for real file
            # ownership) -- surface that loudly rather than silently
            # running as root against someone else's workspace.
            warning = (
                f"runuser not found; command ran as root inside "
                f"{workspace_root} (owned by {owner.name}) -- install "
                f"util-linux's runuser to avoid re-corrupting file "
                f"ownership there."
            )
    return SandboxCommand(tuple(argv), True, backend, warning, env_overrides=env_overrides)
