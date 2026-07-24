"""Stable action/observation fingerprints and evidence classification."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_EMPTY_MARKERS = {"", "(empty)", "[]", "{}", "null", "none", "no matches", "no results"}
_SPACE_RE = re.compile(r"\s+")


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(value)


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8", "replace")).hexdigest()[:24]


def action_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}:{digest(arguments)}"


def normalise_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip().casefold()


def observation_fingerprint(tool_name: str, result: dict[str, Any]) -> str:
    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    compact = {
        "success": bool(result.get("success")),
        "stdout": payload.get("stdout") if isinstance(payload, dict) else "",
        "stderr": payload.get("stderr") if isinstance(payload, dict) else "",
        "error": result.get("error") or (payload.get("error") if isinstance(payload, dict) else ""),
        "path": payload.get("path") if isinstance(payload, dict) else "",
        "items": payload.get("items") if isinstance(payload, dict) else None,
        "matches": payload.get("matches") if isinstance(payload, dict) else None,
        "files_changed": payload.get("files_changed") if isinstance(payload, dict) else None,
    }
    return f"{tool_name}:{digest(compact)}"


def is_empty_result(tool_name: str, result: dict[str, Any]) -> bool:
    if not bool(result.get("success")):
        return True
    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    if not isinstance(payload, dict):
        return normalise_text(payload) in _EMPTY_MARKERS
    if any(payload.get(key) for key in ("files_changed", "path", "destination")):
        return False
    for key in ("items", "matches", "results", "entries"):
        if key in payload:
            value = payload.get(key)
            return not bool(value)
    text = normalise_text(payload.get("stdout") or payload.get("content") or payload.get("output") or "")
    error = normalise_text(payload.get("stderr") or payload.get("error") or result.get("error") or "")
    if error and not text:
        return True
    return text in _EMPTY_MARKERS


def evidence_labels(tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> list[str]:
    if is_empty_result(tool_name, result):
        return []
    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    labels = [f"tool:{tool_name}"]
    path = ""
    if isinstance(payload, dict):
        path = str(payload.get("path") or payload.get("destination") or "")
    path = path or str(arguments.get("path") or arguments.get("cwd") or "")
    if path:
        labels.append(f"path:{path}")
    return labels
