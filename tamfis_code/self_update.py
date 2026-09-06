"""Discover published or checkout releases, verify downloads and update.

Explicit checkout paths work offline. Portable installations discover the
HTTPS release manifest and verify its wheel checksum before invoking pip.
This intentionally only ever offers to update
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
import hashlib
import json
import tempfile
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple

from . import __version__

DEFAULT_REPO_PATH = Path(
    os.environ.get("TAMFIS_CODE_REPO") or Path(__file__).resolve().parents[1]
).expanduser()
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
RELEASE_BASE = "https://gpt.tamfitronics.com/releases/tamfis-code"
_release = None


def _remote_release():
    global _release
    if _release is not None:
        return _release
    try:
        with urlopen(Request(RELEASE_BASE + "/latest.json", headers={"User-Agent": "Tamfis-Code/" + __version__, "Accept": "application/json"}), timeout=3) as response:
            info = json.loads(response.read(16384))
        if not re.fullmatch(r"\d+\.\d+\.\d+", info.get("version", "")):
            return None
        url = info.get("url", "")
        if not url.startswith(RELEASE_BASE + "/") or not url.endswith(".whl"):
            return None
        if not re.fullmatch(r"[a-f0-9]{64}", info.get("sha256", "")):
            return None
        _release = info
        return info
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def update_instructions() -> str:
    return (
        "Run tamfis-code update to update, or:\n"
        "sh -c 'curl -fsSL " + RELEASE_BASE + "/install.sh | TAMFIS_CODE_NON_INTERACTIVE=1 sh'\n\n"
        "Release notes: " + RELEASE_BASE + "/release-notes.md"
    )


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


def check_update_available(repo_path: Optional[Path] = None) -> Optional[str]:
    """Return the newest available release; explicit paths only check locally."""
    repo_version = _repo_version(repo_path or DEFAULT_REPO_PATH)
    versions = [repo_version] if repo_version else []
    # Explicit paths retain the offline checkout API used by local tooling.
    if repo_path is None:
        remote = _remote_release()
        if remote:
            versions.append(remote["version"])
    newest = max(versions, key=_parse_version) if versions else None
    return newest if newest and _parse_version(newest) > _parse_version(__version__) else None


def apply_update(repo_path: Optional[Path] = None) -> Tuple[bool, str]:
    """Reinstall the package from the source checkout. Does not re-exec --
    callers decide whether/when to restart the process (see reexec())."""
    remote = _remote_release() if repo_path is None else None
    checkout = repo_path or DEFAULT_REPO_PATH
    local_version = _repo_version(checkout)
    if remote and (not local_version or _parse_version(remote["version"]) > _parse_version(local_version)):
        try:
            with tempfile.TemporaryDirectory(prefix="tamfis-code-update-") as directory:
                wheel = Path(directory) / Path(urlparse(remote["url"]).path).name
                with urlopen(Request(remote["url"], headers={"User-Agent": "Tamfis-Code/" + __version__}), timeout=60) as response:
                    data = response.read(64 * 1024 * 1024 + 1)
                if len(data) > 64 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != remote["sha256"]:
                    return False, "Update failed: release checksum mismatch or oversized download"
                wheel.write_bytes(data)
                result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", str(wheel)], capture_output=True, text=True, timeout=180)
                if result.returncode:
                    return False, "Update failed: " + (result.stderr or result.stdout)[-2000:]
                return True, f"Updated to {remote['version']}."
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"Update failed: {exc}"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(checkout)],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Update failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        return False, f"Update failed: {detail or 'pip exited non-zero'}"
    return True, f"Updated to {_repo_version(checkout) or 'latest'}."


def reexec() -> None:
    """Replace this process image in place with a fresh invocation of the
    same command line. Never returns on success."""
    os.execv(sys.executable, [sys.executable, "-m", "tamfis_code", *sys.argv[1:]])
