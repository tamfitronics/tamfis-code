"""Local self-update: detect a newer version already sitting in the source
checkout tamfis-code was installed from, apply it, and re-exec in place.

tamfis-code is deployed from a local git checkout (see
install_commit_upgrade_tamfis_code_*.sh), not a package index, so "is an
update available" is a cheap local version-string comparison against that
checkout's pyproject.toml -- no network round trip needed, unlike a
PyPI/npm-registry check. This intentionally only ever offers to update
while the REPL is idle at the prompt, never mid-task: re-exec replaces the
running process image, which would abandon an in-flight tool call or a
raw-mode terminal state if triggered during one. Session/task state is
already durable on disk (see state.py) and the CLI already resumes/
reattaches to the last active session+task on startup, so re-exec is what
actually delivers "update, then land back where I was" -- no separate
resume logic is needed here.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from . import __version__

DEFAULT_REPO_PATH = Path(
    os.environ.get("TAMFIS_CODE_REPO") or Path(__file__).resolve().parents[1]
).expanduser()
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def _repo_version(repo_path: Path) -> Optional[str]:
    try:
        text = (repo_path / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def _parse_version(value: str) -> Tuple[int, ...]:
    parts = []
    for piece in value.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_update_available(repo_path: Path = DEFAULT_REPO_PATH) -> Optional[str]:
    """Return the repo checkout's version string if it's newer than the
    running package, else None. Silent no-op (returns None) if the repo
    checkout isn't present -- e.g. a machine that installed tamfis-code
    without keeping the source tree around."""
    repo_version = _repo_version(repo_path)
    if not repo_version:
        return None
    if _parse_version(repo_version) > _parse_version(__version__):
        return repo_version
    return None


def apply_update(repo_path: Path = DEFAULT_REPO_PATH) -> Tuple[bool, str]:
    """Reinstall the package from the source checkout. Does not re-exec --
    callers decide whether/when to restart the process (see reexec())."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo_path)],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Update failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        return False, f"Update failed: {detail or 'pip exited non-zero'}"
    return True, f"Updated to {_repo_version(repo_path) or 'latest'}."


def reexec() -> None:
    """Replace this process image in place with a fresh invocation of the
    same command line. Never returns on success."""
    os.execv(sys.executable, [sys.executable, "-m", "tamfis_code", *sys.argv[1:]])
