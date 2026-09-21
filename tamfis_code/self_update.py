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
import sysconfig
import hashlib
import json
import tempfile
import time
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple

from . import __version__
from . import config as _config

DEFAULT_REPO_PATH = Path(
    os.environ.get("TAMFIS_CODE_REPO") or Path(__file__).resolve().parents[1]
).expanduser()
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
RELEASE_BASE = "https://gpt.tamfitronics.com/releases/tamfis-code"
_SEMVER_RE = re.compile(r"\d+\.\d+\.\d+")
_SHA_RE = re.compile(r"[a-f0-9]{64}")

# How long a remembered answer is trusted before the network is asked again. The in-memory copy is
# short so a long-lived session notices a release published while it runs; the on-disk copy lets every
# new process alert INSTANTLY (no network, no startup delay) and be refreshed in the background.
MEMORY_TTL_SECONDS = 10 * 60
DISK_TTL_SECONDS = 6 * 60 * 60
ONESHOT_NOTICE_INTERVAL_SECONDS = 24 * 60 * 60

_release = None
_release_at = 0.0


def _valid_release(info) -> bool:
    """Same checks for a network answer and a cached one: a tampered or truncated cache file must
    never be able to point the updater at an arbitrary URL."""
    if not isinstance(info, dict):
        return False
    url = str(info.get("url", ""))
    return bool(
        _SEMVER_RE.fullmatch(str(info.get("version", "")))
        and url.startswith(RELEASE_BASE + "/") and url.endswith(".whl")
        and _SHA_RE.fullmatch(str(info.get("sha256", "")))
    )


def _fetch_manifest(timeout: float = 3.0):
    try:
        with urlopen(Request(RELEASE_BASE + "/latest.json", headers={"User-Agent": "Tamfis-Code/" + __version__, "Accept": "application/json"}), timeout=timeout) as response:
            info = json.loads(response.read(16384))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return info if _valid_release(info) else None


def _cache_path() -> Path:
    return _config.CONFIG_DIR / "update_check.json"


def _read_cache() -> dict:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(**changes) -> None:
    try:
        data = _read_cache()
        data.update(changes)
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # an update hint must never be able to break the CLI


def _remote_release(*, force: bool = False):
    """The published release manifest. Cached in memory only briefly (see MEMORY_TTL_SECONDS): it
    used to be remembered for the life of the process, so the REPL's "check every 30 minutes" poll
    could never see a release published after startup. A transient network failure keeps the last
    known answer rather than reporting "no update"."""
    global _release, _release_at
    now = time.monotonic()
    if not force and _release is not None and now - _release_at < MEMORY_TTL_SECONDS:
        return _release
    info = _fetch_manifest()
    if info is not None:
        _release, _release_at = info, now
        _write_cache(release=info, checked_at=time.time())
    return info if info is not None else _release


def cached_release():
    """The last manifest this machine saw, from disk. No network -- safe at startup and in one-shot
    commands. None when nothing valid is cached."""
    info = _read_cache().get("release")
    return info if _valid_release(info) else None


def cache_age_seconds() -> float:
    """Seconds since the manifest was last fetched from the network (inf when never)."""
    try:
        checked = float(_read_cache().get("checked_at", 0))
    except (TypeError, ValueError):
        return float("inf")
    return max(0.0, time.time() - checked) if checked > 0 else float("inf")


def cache_is_stale() -> bool:
    try:
        return time.time() - float(_read_cache().get("checked_at", 0)) > DISK_TTL_SECONDS
    except (TypeError, ValueError):
        return True


def _newest(versions) -> Optional[str]:
    versions = [v for v in versions if v]
    newest = max(versions, key=_parse_version) if versions else None
    return newest if newest and _parse_version(newest) > _parse_version(__version__) else None


def cached_update_available(repo_path: Optional[Path] = None) -> Optional[str]:
    """Like check_update_available but answered from the on-disk cache only (instant, offline)."""
    checkout = _repo_version(repo_path or DEFAULT_REPO_PATH)
    release = cached_release()
    return _newest([checkout, release["version"] if release else None])


def refresh_update_cache() -> Optional[str]:
    """Ask the network now (bypassing the memory cache) and remember the answer on disk. Blocking:
    run it in a thread. Returns the newest available version, if any."""
    _remote_release(force=True)
    return cached_update_available()


def should_notify_oneshot() -> Optional[str]:
    """The newer version to mention on a one-shot command (ask/agent/...), at most once a day, from
    the cache only. None when up to date, nothing is cached, or it was already mentioned recently."""
    available = cached_update_available()
    if not available:
        return None
    try:
        last = float(_read_cache().get("oneshot_notified_at", 0))
    except (TypeError, ValueError):
        last = 0.0
    return available if time.time() - last > ONESHOT_NOTICE_INTERVAL_SECONDS else None


def mark_oneshot_notified() -> None:
    _write_cache(oneshot_notified_at=time.time())


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


def _pip_install_command(*arguments: str) -> list[str]:
    """Build the updater's pip command for this interpreter.

    Debian/Ubuntu mark their system interpreter as externally managed. An
    already-installed system-wide Tamfis Code still has to be able to update
    itself when the user explicitly clicks Install; pip otherwise rejects the
    operation before looking at the wheel. Virtual environments never receive
    the override.
    """
    command = [sys.executable, "-m", "pip", "install"]
    marker = Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"
    if sys.prefix == sys.base_prefix and marker.exists():
        command.append("--break-system-packages")
    return [*command, *arguments]


def check_update_available(repo_path: Optional[Path] = None) -> Optional[str]:
    """Return the newest available release; explicit paths only check locally."""
    repo_version = _repo_version(repo_path or DEFAULT_REPO_PATH)
    versions = [repo_version] if repo_version else []
    # Explicit paths retain the offline checkout API used by local tooling.
    if repo_path is None:
        remote = _remote_release()
        if remote:
            versions.append(remote["version"])
    return _newest(versions)


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
                result = subprocess.run(_pip_install_command("--upgrade", str(wheel)), capture_output=True, text=True, timeout=180)
                if result.returncode:
                    return False, "Update failed: " + (result.stderr or result.stdout)[-2000:]
                return True, f"Updated to {remote['version']}."
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"Update failed: {exc}"
    try:
        result = subprocess.run(
            _pip_install_command("--no-deps", "-e", str(checkout)),
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
