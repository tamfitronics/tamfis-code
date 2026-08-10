"""Persistent fine-grained tool permissions and protected-path policy."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


_RULE_RE = re.compile(r"^([A-Za-z0-9_*.-]+)(?:\((.*)\))?$")
_PROTECTED_DIRS = {".git", ".tamfis", ".vscode", ".idea", ".husky"}
_PROTECTED_FILES = {
    ".env", ".gitconfig", ".gitmodules", ".bashrc", ".bash_profile",
    ".zshrc", ".zprofile", ".profile", ".ripgreprc",
}
_PATH_KEYS = ("path", "destination", "output_path", "source_dir", "archive_path")
_PROTECTED_COMMAND_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.git(?:/[^\s;&|<>]*)?|\.env(?:\.[A-Za-z0-9_.-]+)?|"
    r"\.tamfis(?:/[^\s;&|<>]*)?|\.vscode(?:/[^\s;&|<>]*)?|\.idea(?:/[^\s;&|<>]*)?|"
    r"\.husky(?:/[^\s;&|<>]*)?|\.(?:bashrc|bash_profile|zshrc|zprofile|profile|ripgreprc))"
)


@dataclass(frozen=True)
class PermissionDecision:
    action: str  # allow | ask | deny
    rule: str
    reason: str
    protected: bool = False


def _target(tool_name: str, arguments: dict) -> str:
    if tool_name == "execute_command":
        return str(arguments.get("command") or "")
    for key in _PATH_KEYS:
        value = arguments.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _matches(rule: str, tool_name: str, arguments: dict, workspace_root: str) -> bool:
    match = _RULE_RE.fullmatch(rule.strip())
    if match is None:
        return False
    tool_pattern, target_pattern = match.groups()
    if not fnmatch.fnmatchcase(tool_name.casefold(), tool_pattern.casefold()):
        return False
    if target_pattern is None:
        return True
    target = _target(tool_name, arguments)
    candidates = [target]
    if tool_name != "execute_command" and target:
        candidate = Path(target).expanduser()
        if not candidate.is_absolute():
            candidate = Path(workspace_root) / candidate
        try:
            candidates.append(str(candidate.resolve().relative_to(Path(workspace_root).resolve())))
        except (OSError, RuntimeError, ValueError):
            pass
    pattern = target_pattern.casefold()
    return any(fnmatch.fnmatchcase(candidate.casefold(), pattern) for candidate in candidates)


def _matching_rule(
    rules: Iterable[str], tool_name: str, arguments: dict, workspace_root: str,
) -> str | None:
    return next(
        (rule for rule in rules if _matches(str(rule), tool_name, arguments, workspace_root)),
        None,
    )


def _protected_path(tool_name: str, arguments: dict, workspace_root: str) -> Path | None:
    if tool_name == "execute_command":
        matched = _PROTECTED_COMMAND_RE.search(_target(tool_name, arguments))
        return Path(matched.group(0)) if matched else None
    if tool_name not in {
        "write_file", "edit_file", "create_file", "patch_file",
        "extract_archive", "repackage_archive", "create_artifact",
    }:
        return None
    raw = _target(tool_name, arguments)
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path(workspace_root) / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return candidate
    if (
        resolved.name in _PROTECTED_FILES
        or resolved.name.startswith(".env.")
        or any(part in _PROTECTED_DIRS for part in resolved.parts)
    ):
        return resolved
    return None


def decide_permission(
    tool_name: str,
    arguments: dict,
    *,
    workspace_root: str,
    allow: Iterable[str] = (),
    ask: Iterable[str] = (),
    deny: Iterable[str] = (),
) -> PermissionDecision | None:
    """Apply deny -> protected-path ask -> explicit ask -> allow precedence."""
    matched = _matching_rule(deny, tool_name, arguments, workspace_root)
    if matched:
        return PermissionDecision("deny", matched, f"Denied by permission rule: {matched}")
    protected = _protected_path(tool_name, arguments, workspace_root)
    if protected is not None:
        return PermissionDecision(
            "ask", "protected-path",
            f"Protected path requires explicit approval: {protected}", protected=True,
        )
    matched = _matching_rule(ask, tool_name, arguments, workspace_root)
    if matched:
        return PermissionDecision("ask", matched, f"Permission rule requires approval: {matched}")
    matched = _matching_rule(allow, tool_name, arguments, workspace_root)
    if matched:
        return PermissionDecision("allow", matched, f"Allowed by permission rule: {matched}")
    return None
