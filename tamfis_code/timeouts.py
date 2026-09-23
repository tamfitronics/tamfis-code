"""Workload-aware command timeouts.

Validation must not inherit a small interactive-command timeout.  A package
with hundreds of Python modules, or a real test suite, can legitimately run
for many minutes.  The timeout is still bounded, but its floor is derived
from the package being checked.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", "checkpoints"}
_COMPILEALL = re.compile(r"(?:^|\s)(?:python(?:3)?)(?:\s+[^;&|]*?)?\s+-m\s+compileall\b", re.I)
_IMPORT_CHECK = re.compile(r"(?:python(?:3)?).*\s-c\s+.*\b(?:import|from)\b", re.I)
_TESTING = re.compile(r"(?:pytest|tox|nox|npm\s+(?:run\s+)?(?:install|test|check|typecheck|build)|cargo\s+test|go\s+test)", re.I)
_TRAINING = re.compile(r"(?:train(?:ing)?|pretrain|fine[-_ ]?tune|frontier)", re.I)


def _package_workload(root: Path) -> tuple[int, int, int]:
    """Return (source files, test files, source bytes) for an active package."""
    source_files = test_files = source_bytes = 0
    try:
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for name in files:
                if not name.endswith((".py", ".pyi", ".js", ".ts", ".tsx", ".rs", ".go")):
                    continue
                source_files += 1
                if name.startswith("test_") or name.endswith(("_test.py", ".spec.ts", ".test.ts", ".spec.js", ".test.js")):
                    test_files += 1
                try:
                    source_bytes += (Path(current) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return source_files, test_files, source_bytes


def adaptive_command_timeout(command: str, cwd: str | Path, requested: int | float | None) -> int:
    """Return a safe timeout floor for validation/test workloads.

    Explicit values remain valid for ordinary commands.  For package checks,
    a model-supplied ``120`` seconds is treated as an interactive default and
    raised to the workload-aware floor so large repositories are not killed
    mid-audit.
    """
    try:
        requested_seconds = max(int(requested or 0), 0)
    except (TypeError, ValueError):
        requested_seconds = 0

    text = str(command or "")
    if not (_COMPILEALL.search(text) or _IMPORT_CHECK.search(text) or _TESTING.search(text) or _TRAINING.search(text)):
        return requested_seconds or 60

    files, test_files, source_bytes = _package_workload(Path(cwd).expanduser().resolve())
    # A workload unit combines breadth and source volume.  This is deliberately
    # continuous: adding one module or a large generated source file changes
    # the estimate gradually instead of crossing an arbitrary package-size
    # bucket.  Test files carry extra weight because they commonly fan out into
    # subprocesses, fixtures, browsers, or network-backed integration checks.
    workload = files + (source_bytes / (256 * 1024)) + (test_files * 3)
    if _TRAINING.search(text):
        estimate = 300 + (workload * 15)
    elif _TESTING.search(text):
        # Dependency installation and builds scale with package metadata and
        # source breadth too; they share the same workload floor as tests.
        estimate = 180 + (workload * 8)
    elif _IMPORT_CHECK.search(text):
        estimate = 60 + (workload * 3)
    else:  # compileall is I/O-bound and scales with the amount of source.
        estimate = 60 + (workload * 2)
    return max(requested_seconds, int(estimate))
