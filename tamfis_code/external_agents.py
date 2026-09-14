"""Read-only discovery and import of session history recorded by *other*
AI coding agents installed on this machine -- Claude Code, Codex CLI,
GitHub Copilot CLI, OpenCode, Kimi Code, and (via TAMFIS_CODE_EXTERNAL_AGENT_DIRS)
anything else that keeps a JSON/JSONL conversation history -- so a user can
ask tamfis-code to pick up work left in another tool instead of
re-explaining it from scratch ("continue what Codex was doing on this").

Every adapter here is read-only and best-effort by design: a missing,
malformed, or future-format-changed store for one tool degrades to an
empty result for that tool alone, never an exception that breaks discovery
of the others (see discover_external_sessions). Nothing in this module
writes to another tool's directory, and credential/config files
(auth.json, .credentials.json, config.toml/json, *.key, session
databases) are never opened -- only the plain transcript/session-index
files each tool already treats as its own conversation history. The
generic adapters additionally skip any filename that merely *looks*
credential-shaped (see _looks_sensitive) as defense in depth for tool
stores this module has no verified format for.

Ground truth for the Claude Code, Codex, and Copilot adapters was read
directly off real on-disk stores at ~/.claude, ~/.codex, and ~/.copilot;
the OpenCode and Kimi Code adapters are generic best-effort scanners since
no verified sample store was available -- they simply return nothing if
the real format doesn't match.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

_MAX_TRANSCRIPT_CHARS = 20_000
_MAX_TURNS = 200

_SECRET_NAME_HINTS = (
    "credential", "auth", "token", "secret", "password",
    "apikey", "api_key", ".env", "config",
)


@dataclass
class ExternalTurn:
    role: str
    text: str
    timestamp: str = ""


@dataclass
class ExternalSession:
    tool: str
    session_id: str
    title: str
    cwd: str
    updated_at: str
    path: str
    turn_count: int = 0

    def label(self) -> str:
        return f"[{self.tool}] {self.title.strip() or '(untitled session)'}"


def _home() -> Path:
    return Path.home()


def _safe_iter(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    try:
        return sorted(p for p in root.glob(pattern) if p.is_file())
    except OSError:
        return []


def _looks_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return any(hint in name for hint in _SECRET_NAME_HINTS)


def _cwd_matches(session_cwd: str, workspace_root: Optional[str]) -> bool:
    """Return whether a session is relevant to ``workspace_root``.

    Coding agents are often launched one directory above the repository
    they eventually work in (for example Claude Code records ``/home`` even
    after its shell commands ``cd /home/project``). Treat that recorded cwd
    as a containing scope as well as accepting an exact match. The reverse
    is intentionally not true: a session rooted in one child repository is
    not relevant when tamfis-code is opened from a broader parent directory.
    A session with no cwd remains visible because there is no reliable basis
    on which to exclude it.
    """
    if not workspace_root or not session_cwd:
        return True
    try:
        session_path = Path(session_cwd).expanduser().resolve()
        workspace_path = Path(workspace_root).expanduser().resolve()
        return session_path == workspace_path or session_path in workspace_path.parents
    except OSError:
        return session_cwd.rstrip("/") == workspace_root.rstrip("/")


def _condense_title(text: str, limit: int = 70) -> str:
    text = re.sub(r"<pasted_content[^>]*/?>", "[pasted content]", text or "")
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    return text[:limit] + ("…" if len(text) > limit else "")


def _content_text(content: Any) -> str:
    """Extract plain text from either a bare string or an Anthropic/OpenAI
    -style content-block list (`{"type": "text"/"input_text"/"output_text",
    "text": ...}`), shared by the Claude Code and Codex adapters."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return ""


def _bound_turns(turns: list[ExternalTurn], max_chars: int) -> list[ExternalTurn]:
    """Keep the most recent turns/chars -- what matters for "continue this"
    is the tail of the conversation, not its opening, and an unbounded
    transcript would blow past the model's context on read."""
    turns = turns[-_MAX_TURNS:]
    total = 0
    kept: list[ExternalTurn] = []
    for turn in reversed(turns):
        total += len(turn.text)
        kept.append(turn)
        if total >= max_chars:
            break
    kept.reverse()
    return kept


def _mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Claude Code: ~/.claude/projects/<cwd-slug>/<session-id>.jsonl
# --------------------------------------------------------------------------

def _claude_code_discover(home: Path) -> list[ExternalSession]:
    projects_dir = home / ".claude" / "projects"
    sessions: list[ExternalSession] = []
    for jsonl_path in _safe_iter(projects_dir, "*/*.jsonl"):
        try:
            session = _claude_code_parse_session(jsonl_path)
        except Exception:
            continue
        if session is not None:
            sessions.append(session)
    return sessions


def _claude_code_parse_session(path: Path) -> Optional[ExternalSession]:
    session_id = path.stem
    title = ""
    cwd = ""
    turn_count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "ai-title" and not title:
                    title = str(entry.get("aiTitle") or "").strip()
                if not cwd and isinstance(entry.get("cwd"), str):
                    cwd = entry["cwd"]
                if entry.get("type") in ("user", "assistant"):
                    turn_count += 1
                    if not title and entry.get("type") == "user":
                        text = _content_text((entry.get("message") or {}).get("content"))
                        title = _condense_title(text.splitlines()[0] if text else "")
    except OSError:
        return None
    if turn_count == 0:
        return None
    return ExternalSession(
        tool="claude-code", session_id=session_id, title=title or "(untitled)",
        cwd=cwd, updated_at=_mtime_iso(path), path=str(path), turn_count=turn_count,
    )


def _claude_code_read(session: ExternalSession, max_chars: int) -> list[ExternalTurn]:
    turns: list[ExternalTurn] = []
    try:
        with Path(session.path).open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") not in ("user", "assistant"):
                    continue
                message = entry.get("message") or {}
                text = _content_text(message.get("content")).strip()
                if not text:
                    continue
                turns.append(ExternalTurn(
                    role=str(message.get("role") or entry["type"]),
                    text=text, timestamp=str(entry.get("timestamp") or ""),
                ))
    except OSError:
        return []
    return _bound_turns(turns, max_chars)


# --------------------------------------------------------------------------
# Codex CLI: ~/.codex/sessions/YYYY/MM/DD/rollout-*-<id>.jsonl,
# titled via ~/.codex/session_index.jsonl (later lines override earlier
# ones for the same id -- the index is an append-only rename log).
# --------------------------------------------------------------------------

_CODEX_ID_RE = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F-]{27,})\.jsonl$")


def _codex_discover(home: Path) -> list[ExternalSession]:
    codex_home = Path(os.environ.get("CODEX_HOME") or home / ".codex").expanduser()
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.exists():
        return []
    titles: dict[str, str] = {}
    index_path = codex_home / "session_index.jsonl"
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid, name = entry.get("id"), str(entry.get("thread_name") or "").strip()
                    if sid and name:
                        titles[sid] = name
        except OSError:
            pass
    results: list[ExternalSession] = []
    for jsonl_path in _safe_iter(sessions_dir, "**/*.jsonl"):
        try:
            session = _codex_parse_session(jsonl_path, titles)
        except Exception:
            continue
        if session:
            results.append(session)
    return results


def _codex_parse_session(path: Path, titles: dict[str, str]) -> Optional[ExternalSession]:
    # Discovery only reads the first line (session_meta, confirmed always
    # ordinal 0 in real rollout files) -- a full parse just to list sessions
    # would mean reading every reasoning/tool-call event in every rollout
    # file on disk. _codex_read does the full parse, lazily, for one chosen
    # session.
    session_id = ""
    cwd = ""
    updated_at = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline().strip()
        if first_line:
            entry = json.loads(first_line)
            if entry.get("type") == "session_meta":
                payload = entry.get("payload") or {}
                session_id = str(payload.get("session_id") or payload.get("id") or "")
                cwd = str(payload.get("cwd") or "")
                updated_at = str(payload.get("timestamp") or entry.get("timestamp") or "")
    except (OSError, json.JSONDecodeError):
        pass
    if not session_id:
        match = _CODEX_ID_RE.search(path.name)
        session_id = match.group(1) if match else path.stem
    return ExternalSession(
        tool="codex", session_id=session_id, title=titles.get(session_id, ""),
        cwd=cwd, updated_at=updated_at or _mtime_iso(path), path=str(path), turn_count=0,
    )


def _codex_read(session: ExternalSession, max_chars: int) -> list[ExternalTurn]:
    turns: list[ExternalTurn] = []
    try:
        with Path(session.path).open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 20_000:  # bound pathologically long rollout files
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "response_item":
                    continue
                payload = entry.get("payload") or {}
                if payload.get("type") != "message":
                    continue
                text = _content_text(payload.get("content")).strip()
                # The very first user item in every Codex session is a
                # synthetic <environment_context> block, not something the
                # user actually said -- surfacing it as "what the user
                # asked" would be misleading in a continuation brief.
                if not text or text.startswith("<environment_context"):
                    continue
                turns.append(ExternalTurn(
                    role=str(payload.get("role") or ""), text=text,
                    timestamp=str(entry.get("timestamp") or ""),
                ))
    except OSError:
        return []
    return _bound_turns(turns, max_chars)


# --------------------------------------------------------------------------
# GitHub Copilot CLI: ~/.copilot/session-state/<id>/{workspace.yaml,events.jsonl}
# --------------------------------------------------------------------------

_YAML_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def _parse_flat_yaml(text: str) -> dict[str, str]:
    """Minimal single-level `key: value` reader for Copilot's small
    workspace.yaml sidecar -- avoids a YAML dependency for a file that's
    never more than a flat handful of scalar fields."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _YAML_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def _copilot_discover(home: Path) -> list[ExternalSession]:
    copilot_home = Path(os.environ.get("COPILOT_HOME") or home / ".copilot").expanduser()
    results: list[ExternalSession] = []
    for workspace_yaml in _safe_iter(copilot_home / "session-state", "*/workspace.yaml"):
        try:
            session = _copilot_parse_session(workspace_yaml)
        except Exception:
            continue
        if session:
            results.append(session)
    return results


def _copilot_parse_session(workspace_yaml: Path) -> Optional[ExternalSession]:
    try:
        values = _parse_flat_yaml(workspace_yaml.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    events_path = workspace_yaml.parent / "events.jsonl"
    return ExternalSession(
        tool="copilot",
        session_id=values.get("id") or workspace_yaml.parent.name,
        # Confirmed live: Copilot's own "name" field is often just the raw
        # first user message (sometimes thousands of characters, including
        # <pasted_content> markers) rather than a real title -- condense it
        # the same way this module condenses its own generic titles.
        title=_condense_title(values.get("name", "")),
        cwd=values.get("cwd", ""),
        updated_at=values.get("updated_at") or _mtime_iso(workspace_yaml),
        path=str(events_path if events_path.exists() else workspace_yaml),
        turn_count=0,
    )


def _copilot_read(session: ExternalSession, max_chars: int) -> list[ExternalTurn]:
    path = Path(session.path)
    if path.name != "events.jsonl":
        return []
    turns: list[ExternalTurn] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = entry.get("type", "")
                data = entry.get("data") or {}
                if etype == "user.message":
                    text = str(data.get("content") or "").strip()
                    if text:
                        turns.append(ExternalTurn(role="user", text=text, timestamp=str(entry.get("timestamp") or "")))
                elif etype in ("assistant.message", "model.turn_ended"):
                    text = str(data.get("content") or data.get("text") or data.get("message") or "").strip()
                    if text:
                        turns.append(ExternalTurn(role="assistant", text=text, timestamp=str(entry.get("timestamp") or "")))
    except OSError:
        return []
    return _bound_turns(turns, max_chars)


# --------------------------------------------------------------------------
# Generic best-effort adapter, used for tools without a verified on-disk
# format (OpenCode, Kimi Code) and for user-configured extra stores via
# TAMFIS_CODE_EXTERNAL_AGENT_DIRS ("name=path,name2=path2"). Looks for
# common field-name conventions across a directory of .json/.jsonl files
# and simply finds nothing if a given tool's real format doesn't match --
# see the module docstring's ground-truth note.
# --------------------------------------------------------------------------

def _generic_discover(root: Path, *, tool: str, subdirs: tuple[str, ...]) -> list[ExternalSession]:
    if not root.exists():
        return []
    candidates: list[Path] = []
    for sub in subdirs:
        candidates.extend(_safe_iter(root / sub, "**/*.json"))
        candidates.extend(_safe_iter(root / sub, "**/*.jsonl"))
    results: list[ExternalSession] = []
    for file_path in candidates:
        if _looks_sensitive(file_path):
            continue
        try:
            session = _generic_parse_session(file_path, tool=tool)
        except Exception:
            continue
        if session:
            results.append(session)
    return results


def _generic_parse_session(path: Path, *, tool: str) -> Optional[ExternalSession]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    data: Any = None
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    title = str(data.get("title") or data.get("name") or "").strip()
    if not title:
        for key in ("messages", "turns", "history"):
            items = data.get(key)
            if isinstance(items, list) and items and isinstance(items[0], dict):
                title = _condense_title(str(items[0].get("content") or items[0].get("text") or ""))
                break
    return ExternalSession(
        tool=tool,
        session_id=str(data.get("session_id") or data.get("id") or path.stem),
        title=title or "(untitled)",
        cwd=str(data.get("cwd") or data.get("workspace") or data.get("directory") or data.get("workspace_root") or ""),
        updated_at=str(data.get("updated_at") or data.get("updatedAt") or data.get("timestamp") or "") or _mtime_iso(path),
        path=str(path),
        turn_count=0,
    )


def _generic_read(session: ExternalSession, max_chars: int) -> list[ExternalTurn]:
    path = Path(session.path)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    entries: list[dict] = []
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("messages", "turns", "history"):
                if isinstance(data.get(key), list):
                    entries = [e for e in data[key] if isinstance(e, dict)]
                    break
    turns: list[ExternalTurn] = []
    for entry in entries:
        role = str(entry.get("role") or entry.get("type") or "")
        text = (_content_text(entry.get("content")) or str(entry.get("text") or "")).strip()
        if role in ("user", "assistant") and text:
            turns.append(ExternalTurn(role=role, text=text, timestamp=str(entry.get("timestamp") or "")))
    return _bound_turns(turns, max_chars)


def _kimi_code_discover(home: Path) -> list[ExternalSession]:
    root = Path(os.environ.get("KIMI_CODE_HOME") or home / ".kimi-code").expanduser()
    return _generic_discover(root, tool="kimi-code", subdirs=("sessions", "conversations", "history"))


def _opencode_discover(home: Path) -> list[ExternalSession]:
    roots = [home / ".local" / "share" / "opencode", home / ".config" / "opencode"]
    env_home = os.environ.get("OPENCODE_HOME")
    if env_home:
        roots.insert(0, Path(env_home))
    results: list[ExternalSession] = []
    seen: set[str] = set()
    for root in roots:
        for session in _generic_discover(Path(root).expanduser(), tool="opencode", subdirs=("session", "sessions", "storage/session")):
            if session.path in seen:
                continue
            seen.add(session.path)
            results.append(session)
    return results


def _extra_configured_adapters() -> dict[str, tuple[Callable[[Path], list[ExternalSession]], Callable[[ExternalSession, int], list[ExternalTurn]]]]:
    """User-configured additional tool stores: TAMFIS_CODE_EXTERNAL_AGENT_DIRS
    = "name=/path/to/sessions/dir,other=/path". Each is scanned with the
    same generic best-effort parser as OpenCode/Kimi Code."""
    raw = os.environ.get("TAMFIS_CODE_EXTERNAL_AGENT_DIRS", "")
    extra: dict[str, tuple[Callable, Callable]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, _, path_str = item.partition("=")
        name, path_str = name.strip(), path_str.strip()
        if not name or not path_str:
            continue

        def discover_fn(_home: Path, _root: str = path_str, _name: str = name) -> list[ExternalSession]:
            return _generic_discover(Path(_root).expanduser(), tool=_name, subdirs=("",))

        extra[name] = (discover_fn, _generic_read)
    return extra


_BUILTIN_ADAPTERS: dict[str, tuple[Callable[[Path], list[ExternalSession]], Callable[[ExternalSession, int], list[ExternalTurn]]]] = {
    "claude-code": (_claude_code_discover, _claude_code_read),
    "codex": (_codex_discover, _codex_read),
    "copilot": (_copilot_discover, _copilot_read),
    "opencode": (_opencode_discover, _generic_read),
    "kimi-code": (_kimi_code_discover, _generic_read),
}


def _all_adapters() -> dict[str, tuple[Callable, Callable]]:
    return {**_BUILTIN_ADAPTERS, **_extra_configured_adapters()}


def known_tools() -> tuple[str, ...]:
    return tuple(_all_adapters())


def discover_external_sessions(
    *, workspace_root: Optional[str] = None, tools: Optional[Iterable[str]] = None, limit: int = 50,
) -> list[ExternalSession]:
    """Best-effort listing of sessions recorded by other AI coding agents on
    this machine, newest first. Silently skips any tool whose store is
    missing, unreadable, or raises -- see the module docstring's safety
    contract. `workspace_root`, if given, keeps only sessions whose
    recorded cwd resolves to that same directory (a session with no
    recorded cwd is always kept, not dropped)."""
    home = _home()
    adapters = _all_adapters()
    wanted = set(tools) if tools else set(adapters)
    sessions: list[ExternalSession] = []
    for name in wanted:
        adapter = adapters.get(name)
        if not adapter:
            continue
        discover_fn, _ = adapter
        try:
            sessions.extend(discover_fn(home))
        except Exception:
            continue
    sessions = [s for s in sessions if _cwd_matches(s.cwd, workspace_root)]
    sessions.sort(key=lambda s: s.updated_at or "", reverse=True)
    return sessions[:limit]


def read_external_session(
    tool: str, session_id: str, *, max_chars: int = _MAX_TRANSCRIPT_CHARS,
) -> Optional[dict[str, Any]]:
    """Read one external session's turns by (tool, session_id), normalized
    to plain role/text/timestamp dicts. Re-runs discovery to resolve the id
    to its file -- these stores are flat files, not a database, so there's
    no persistent handle to look up directly; discovery itself is cheap
    (see _codex_parse_session's first-line-only read, in particular).
    Returns None if the tool is unknown or the session can't be found."""
    adapter = _all_adapters().get(tool)
    if not adapter:
        return None
    discover_fn, read_fn = adapter
    try:
        candidates = discover_fn(_home())
    except Exception:
        return None
    match = next((s for s in candidates if s.session_id == session_id), None)
    if match is None and session_id:
        # The terminal list deliberately shows a compact prefix so it remains
        # useful at 80 columns. Accept that prefix when it resolves to one
        # session, just as git accepts an unambiguous abbreviated object id.
        prefix_matches = [s for s in candidates if s.session_id.startswith(session_id)]
        if len(prefix_matches) == 1:
            match = prefix_matches[0]
    if match is None:
        return None
    try:
        turns = read_fn(match, max_chars)
    except Exception:
        turns = []
    return {
        "tool": match.tool,
        "session_id": match.session_id,
        "title": match.title,
        "cwd": match.cwd,
        "updated_at": match.updated_at,
        "turns": [{"role": t.role, "text": t.text, "timestamp": t.timestamp} for t in turns],
    }


def continuation_brief(record: dict[str, Any], *, max_chars: int = _MAX_TRANSCRIPT_CHARS) -> str:
    """Render a normalized session record (from read_external_session) as a
    compact prompt block that seeds a new tamfis-code objective -- what
    `continue-from` and the read_external_agent_session tool hand back so
    the model can pick up someone else's task without the user having to
    re-explain it."""
    lines = [f"Continuing a session from {record['tool']} (id {record['session_id']})."]
    if record.get("title"):
        lines.append(f"Title: {record['title']}")
    if record.get("cwd"):
        lines.append(f"Original working directory: {record['cwd']}")
    lines.append("")
    lines.append("Transcript (oldest to newest):")
    body = "\n\n".join(f"[{turn['role']}] {turn['text']}" for turn in record.get("turns", []))
    if len(body) > max_chars:
        body = "...(earlier turns truncated)...\n" + body[-max_chars:]
    lines.append(body or "(no transcript content recovered)")
    return "\n".join(lines)
