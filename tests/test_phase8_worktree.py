import subprocess
from pathlib import Path

import pytest

from tamfis_code.runtime.worktree import (
    WorktreeError,
    create_worktree,
    is_worktree_clean,
    list_worktrees,
    remove_worktree,
)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=path, check=True)
    return path


def test_create_worktree_requires_git_repo(tmp_path: Path):
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()
    with pytest.raises(WorktreeError):
        create_worktree(plain_dir, branch="feature-x")


def test_create_list_remove_round_trip(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    handle = create_worktree(repo, branch="feature-x")
    assert handle.path.is_dir()
    assert (handle.path / "README.md").is_file()

    listed = {h.branch: h for h in list_worktrees(repo)}
    assert "feature-x" in listed

    assert is_worktree_clean(handle) is True
    remove_worktree(handle)
    assert not handle.path.exists()
    assert "feature-x" not in {h.branch for h in list_worktrees(repo)}


def test_remove_refuses_dirty_worktree_without_force(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    handle = create_worktree(repo, branch="feature-y")
    (handle.path / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")

    assert is_worktree_clean(handle) is False
    with pytest.raises(WorktreeError):
        remove_worktree(handle)
    assert handle.path.exists()

    remove_worktree(handle, force=True)
    assert not handle.path.exists()


def test_create_worktree_refuses_existing_path(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    handle = create_worktree(repo, branch="feature-z")
    with pytest.raises(WorktreeError):
        create_worktree(repo, branch="feature-z-2", path=handle.path)
