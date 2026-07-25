"""Git worktree isolation for background/parallel agent execution.

Mirrors the conservative isolation semantics this project itself relies on:
a worktree is created for the caller to use, and never auto-removed or
auto-merged -- cleanup only happens on an explicit call, and only when the
worktree is clean unless the caller explicitly forces it.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class WorktreeError(RuntimeError):
    """Raised for any worktree operation Tamfis-Code refuses to perform."""


@dataclass(frozen=True)
class WorktreeHandle:
    path: Path
    branch: str
    repo_root: Path


def _slug(branch: str) -> str:
    slug = _SLUG_RE.sub("-", branch.strip().lower()).strip("-")
    return slug or "worktree"


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True,
    )


def _require_git_repo(repo_root: Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    result = _run_git(root, "rev-parse", "--is-inside-work-tree")
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise WorktreeError(f"{root} is not a git repository; cannot create a worktree.")
    return root


def create_worktree(
    repo_root: str | Path, *, branch: str, base: str = "HEAD", path: str | Path | None = None,
) -> WorktreeHandle:
    root = _require_git_repo(Path(repo_root))
    target = Path(path).expanduser().resolve() if path is not None else root / ".tamfis" / "worktrees" / _slug(branch)
    if target.exists():
        raise WorktreeError(f"Worktree path already exists: {target}. Choose a different path or branch.")

    branch_exists = _run_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0
    target.parent.mkdir(parents=True, exist_ok=True)
    args = ["worktree", "add"]
    if branch_exists:
        args += [str(target), branch]
    else:
        args += ["-b", branch, str(target), base]
    result = _run_git(root, *args)
    if result.returncode != 0:
        raise WorktreeError(f"git worktree add failed: {result.stderr.strip() or result.stdout.strip()}")
    return WorktreeHandle(path=target, branch=branch, repo_root=root)


def is_worktree_clean(handle: WorktreeHandle) -> bool:
    result = _run_git(handle.path, "status", "--porcelain")
    if result.returncode != 0:
        raise WorktreeError(f"git status failed in worktree {handle.path}: {result.stderr.strip()}")
    return result.stdout.strip() == ""


def remove_worktree(handle: WorktreeHandle, *, force: bool = False) -> None:
    if not force and not is_worktree_clean(handle):
        raise WorktreeError(
            f"Worktree {handle.path} has uncommitted changes; refusing to remove without force=True."
        )
    args = ["worktree", "remove", str(handle.path)]
    if force:
        args.append("--force")
    result = _run_git(handle.repo_root, *args)
    if result.returncode != 0:
        raise WorktreeError(f"git worktree remove failed: {result.stderr.strip() or result.stdout.strip()}")
    _run_git(handle.repo_root, "worktree", "prune")


def list_worktrees(repo_root: str | Path) -> list[WorktreeHandle]:
    root = _require_git_repo(Path(repo_root))
    result = _run_git(root, "worktree", "list", "--porcelain")
    if result.returncode != 0:
        raise WorktreeError(f"git worktree list failed: {result.stderr.strip()}")
    handles: list[WorktreeHandle] = []
    current_path: Path | None = None
    current_branch = ""
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            if current_path is not None:
                handles.append(WorktreeHandle(path=current_path, branch=current_branch, repo_root=root))
            current_path = Path(line[len("worktree "):])
            current_branch = ""
        elif line.startswith("branch "):
            current_branch = line[len("branch "):].removeprefix("refs/heads/")
    if current_path is not None:
        handles.append(WorktreeHandle(path=current_path, branch=current_branch, repo_root=root))
    return [h for h in handles if h.path != root]
