"""Multi-stage context compression cascade ("7-layer" context invincibility).

Single-pass summarization is not enough to survive a long agent turn: by the
time a generic compactor runs, the detail it throws away is exactly the detail
the model needs next round (confirmed live: a long exploration turn thrashed,
re-reading the same paths because the earlier read had been discarded).

This module implements the cascade in ordered, individually-observable stages,
cheapest first, so a turn only pays for the layers it actually needs:

  Stage 1 (micro)       -- local truncation of large tool/assistant/user
                           payloads: keep head + tail tokens and a pointer
                           (file path / re-run hint) instead of the whole body.
  Stage 2 (structured)  -- once context crosses the trigger ratio (default
                           80%), inject a bounded "State of the Union": files
                           touched, failed attempts, pending TODOs, and the
                           identifiers/state already established. This is what
                           lets a fact from 100 messages ago still be present
                           after everything around it was evicted.
  Stage 3 (elastic)     -- contextual pruning of *superseded* file reads:
                           inject only the signature/docstring view of a file
                           (see :func:`signature_view`) plus a pointer, instead
                           of the entire body that was read long ago.

Everything here is pure, synchronous and dependency-free (no provider calls),
so it can be unit-tested directly and reused by any caller. Failures never
raise into the agent loop: a stage that cannot do its job reports why in the
:class:`CompressionReport` and the cascade continues.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# A stable, greppable marker separating cacheable static instructions from
# volatile per-turn context (see :class:`CacheBoundary`).
CACHE_BOUNDARY_MARKER = "\n\n<!-- tamfis:cache-boundary -->\n\n"

# State-of-the-Union budget (Stage 2). Bounded on purpose: the summary exists
# to survive eviction, not to become a second full transcript.
STATE_OF_UNION_BUDGET_TOKENS = 20_000
STATE_OF_UNION_HEADER = "STATE OF THE UNION (compressed durable context):"

DEFAULT_MICRO_MAX_TOKENS = 2_000
DEFAULT_MICRO_HEAD_TOKENS = 500
DEFAULT_MICRO_TAIL_TOKENS = 500
DEFAULT_TRIGGER_RATIO = 0.80
# Superseded file reads older than this many most-recent tool results keep only
# their signature view.
DEFAULT_SIGNATURE_KEEP_RECENT = 4
DEFAULT_SIGNATURE_MIN_TOKENS = 800

_CHARS_PER_TOKEN = 4

_PATH_RE = re.compile(r"(?<![\w/.-])((?:/|\./|\.\./)?[\w.-]+(?:/[\w.-]+)+\.\w{1,8})")
_ASSIGNMENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*=\s*[^=\s]")
_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]*(.{0,120})", re.IGNORECASE)
_FAILURE_RE = re.compile(
    r"(?:Traceback|\bError\b|error:|failed|FAILED|AssertionError|Exception)"
    r"[^\n]{0,160}",
)
_SIGNATURE_START_RE = re.compile(
    r"^\s*(?:export\s+|public\s+|private\s+|protected\s+|static\s+|abstract\s+|final\s+|"
    r"async\s+|def\s+|class\s+|func\s+|function\s+|interface\s+|trait\s+|struct\s+|enum\s+|"
    r"impl\s+|module\s+|namespace\s+)*(?:def|class|func|function|interface|trait|struct|"
    r"enum|impl|module|namespace|const|var|let|type)\b"
)


def estimate_tokens(value: Any) -> int:
    """Cheap, provider-independent token estimate (~4 chars/token).

    Intentionally the same estimate the rest of the runtime uses for budget
    decisions, so compression thresholds and budgeting agree.
    """
    text = value if isinstance(value, str) else str(value or "")
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _one_line(value: str, limit: int = 200) -> str:
    collapsed = " ".join(str(value or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3].rstrip() + "..."


# --------------------------------------------------------------------------
# Stage 1: micro truncation
# --------------------------------------------------------------------------


def micro_compact(
    text: str,
    *,
    pointer: str = "",
    max_tokens: int = DEFAULT_MICRO_MAX_TOKENS,
    head_tokens: int = DEFAULT_MICRO_HEAD_TOKENS,
    tail_tokens: int = DEFAULT_MICRO_TAIL_TOKENS,
    label: str = "output",
) -> tuple[str, bool]:
    """Stage 1 -- bound one oversized payload, preserving head, tail and a
    pointer to where the full value still lives.

    Returns ``(text, changed)``. A payload at or under ``max_tokens`` is
    returned untouched (so this is always safe to call unconditionally).
    """
    raw = text or ""
    if not raw or estimate_tokens(raw) <= max_tokens:
        return raw, False
    head_chars = max(1, head_tokens * _CHARS_PER_TOKEN)
    tail_chars = max(0, tail_tokens * _CHARS_PER_TOKEN)
    omitted = max(0, len(raw) - head_chars - tail_chars)
    hint = pointer or "re-read the source path or re-run the command to restore full detail"
    marker = f"...[{label} compacted: {omitted} characters omitted; {hint}]..."
    compacted = raw[:head_chars].rstrip() + f"\n{marker}\n" + (raw[-tail_chars:] if tail_chars else "")
    return compacted, True


# --------------------------------------------------------------------------
# Stage 3 plumbing (used here, and by the elastic stage)
# --------------------------------------------------------------------------


def extract_signatures(source: str, *, max_chars: int = 6_000, path: str = "") -> str:
    """Return the signature/docstring view of a source file -- the part of a
    file another agent turn genuinely needs to reason about, without the
    bodies.

    Deliberately language-agnostic (Python, JS/TS, Go, Rust, Java/C#, PHP):
    declaration lines are kept, their indented docstring/comment first lines
    are kept, everything else is replaced by an elision marker. Non-source or
    unparseable text degrades to a bounded head preview rather than an error.
    """
    lines = (source or "").splitlines()
    kept: list[str] = []
    total = len(lines)
    index = 0
    while index < total:
        line = lines[index]
        stripped = line.strip()
        # Module/file docstring or leading header comment: kept only at the top.
        if not kept and (stripped.startswith(("'''", '"""')) or stripped.startswith(("#", "//", "/*", "*"))):
            kept.append(line.rstrip())
            if stripped.startswith(("'''", '"""')):
                quote = stripped[:3]
                if quote not in stripped[3:]:
                    index += 1
                    while index < total:
                        kept.append(lines[index].rstrip())
                        if quote in lines[index]:
                            break
                        index += 1
            index += 1
            continue
        if _SIGNATURE_START_RE.match(line):
            kept.append(line.rstrip())
            indent = line[: len(line) - len(line.lstrip())]
            step = 1
            following = lines[index + 1].strip() if index + 1 < total else ""
            if following.startswith(("'''", '"""')):
                quote = following[:3]
                kept.append(lines[index + 1].rstrip())
                step = 2
                if quote not in following[3:]:
                    while index + step < total and quote not in lines[index + step]:
                        kept.append(lines[index + step].rstrip())
                        step += 1
                    if index + step < total:
                        kept.append(lines[index + step].rstrip())
                        step += 1
            elif following.startswith(("#", "//")):
                kept.append(lines[index + 1].rstrip())
                step = 2
            kept.append(f"{indent}    <body elided>")
            index += step
            continue
        index += 1
    if not kept:
        # Not recognizably source: keep a bounded head so the caller still
        # gets *something* structural instead of nothing.
        preview = "\n".join(lines[:40])
        return _one_line(preview, max_chars)
    rendered = "\n".join(kept)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n... [signature view truncated]"
    return rendered


def signature_view(path: str | Path, *, max_chars: int = 6_000) -> Optional[str]:
    """Stage 3 helper -- signature view of a file on disk, or None when the
    file cannot be read (the cascade then leaves that message untouched)."""
    try:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            return None
        if candidate.stat().st_size > 4_000_000:
            return None
        source = candidate.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError, RuntimeError):
        return None
    view = extract_signatures(source, max_chars=max_chars, path=str(path))
    return view or None


# --------------------------------------------------------------------------
# Stage 2: structured State of the Union
# --------------------------------------------------------------------------


@dataclass
class StateOfUnionFacts:
    """Everything Stage 2 knows, as structured data (so it is inspectable and
    testable independently of its rendered form)."""

    objective: str = ""
    files_touched: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    pending_todos: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    turn_count: int = 0

    def is_empty(self) -> bool:
        return not any((
            self.files_touched, self.identifiers, self.failed_attempts,
            self.pending_todos, self.evidence_ids,
        ))


def extract_facts(
    messages: Iterable[dict[str, Any]],
    *,
    session_state: Any = None,
    cap: int = 40,
) -> StateOfUnionFacts:
    """Derive the durable facts of a conversation from the messages themselves.

    This is the mechanism that makes a 100-message-old fact survive: files,
    identifiers, failures and TODOs are extracted *before* anything is
    evicted, then re-injected as a single bounded system message.
    """
    facts = StateOfUnionFacts()
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, default=str) if content is not None else ""
        if role == "user" and not facts.objective and content.strip():
            facts.objective = _one_line(content, 300)
        facts.turn_count += 1
        for match in _PATH_RE.finditer(content):
            candidate = match.group(1)
            if candidate not in facts.files_touched and not candidate.startswith(("http", "//")):
                facts.files_touched.append(candidate)
        for match in _ASSIGNMENT_RE.finditer(content):
            name = match.group(1)
            if name not in facts.identifiers and len(facts.identifiers) < cap:
                facts.identifiers.append(name)
        for match in _FAILURE_RE.finditer(content):
            snippet = _one_line(match.group(0), 160)
            if snippet not in facts.failed_attempts and len(facts.failed_attempts) < cap:
                facts.failed_attempts.append(snippet)
        for match in _TODO_RE.finditer(content):
            snippet = _one_line(f"{match.group(1).upper()}: {match.group(2)}", 160)
            if snippet not in facts.pending_todos and len(facts.pending_todos) < cap:
                facts.pending_todos.append(snippet)
        for match in re.finditer(r"evidence_[a-f0-9]{6,}", content):
            if match.group(0) not in facts.evidence_ids and len(facts.evidence_ids) < cap:
                facts.evidence_ids.append(match.group(0))
    if session_state is not None:
        for path in list(getattr(session_state, "inspected_files", []) or [])[-cap:]:
            if str(path) not in facts.files_touched:
                facts.files_touched.append(str(path))
        for entry in list(getattr(session_state, "completed_actions", []) or [])[-cap:]:
            summary = _one_line(str(getattr(entry, "summary", "") or ""), 120)
            if summary and summary not in facts.failed_attempts and "fail" in summary.lower():
                facts.failed_attempts.append(summary)
    facts.files_touched = facts.files_touched[:cap]
    return facts


def render_state_of_union(facts: StateOfUnionFacts, *, max_tokens: int = STATE_OF_UNION_BUDGET_TOKENS) -> str:
    """Render the facts as bounded Markdown for injection into the prompt."""
    if facts.is_empty():
        return ""
    sections: list[str] = [STATE_OF_UNION_HEADER]
    if facts.objective:
        sections.append(f"- Objective: {facts.objective}")
    if facts.files_touched:
        sections.append("- Files touched / inspected:\n" + "\n".join(f"  - {p}" for p in facts.files_touched))
    if facts.identifiers:
        sections.append(
            "- Established identifiers / state:\n"
            + "\n".join(f"  - {name}" for name in facts.identifiers)
        )
    if facts.failed_attempts:
        sections.append("- Failed attempts (do not repeat blindly):\n" + "\n".join(f"  - {f}" for f in facts.failed_attempts))
    if facts.pending_todos:
        sections.append("- Pending TODOs:\n" + "\n".join(f"  - {t}" for t in facts.pending_todos))
    if facts.evidence_ids:
        sections.append("- Durable evidence ids: " + ", ".join(facts.evidence_ids))
    sections.append("Treat the above as already-established state; do not re-discover it.")
    rendered = "\n".join(sections)
    max_chars = max(1_000, max_tokens * _CHARS_PER_TOKEN)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n... [state of the union truncated at budget]"
    return rendered


def build_state_of_union(
    messages: Iterable[dict[str, Any]],
    *,
    session_state: Any = None,
    max_tokens: int = STATE_OF_UNION_BUDGET_TOKENS,
) -> tuple[str, StateOfUnionFacts]:
    """Stage 2 entry point -- extract then render. Returns (text, facts)."""
    facts = extract_facts(messages, session_state=session_state)
    return render_state_of_union(facts, max_tokens=max_tokens), facts


# --------------------------------------------------------------------------
# Cache boundary
# --------------------------------------------------------------------------


@dataclass
class CacheBoundary:
    """Split a system prompt into a cacheable static prefix and a volatile
    suffix.

    Provider-side prompt caching (OpenAI/Anthropic/DeepSeek-style) is
    prefix-based: the cache hits only while the *beginning* of the request is
    byte-identical. Appending per-turn state (plan, fingerprint, validation
    evidence) to the same system message therefore invalidates the cache every
    single turn. Keeping the static instructions first and the volatile state
    in its own message preserves both the ordering the model already saw and
    the identical prefix the provider can cache.
    """

    static_prefix: str
    volatile_suffix: str = ""

    def as_system_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.static_prefix}]
        if self.volatile_suffix:
            messages.append({"role": "system", "content": self.volatile_suffix})
        return messages

    def joined(self) -> str:
        if not self.volatile_suffix:
            return self.static_prefix
        return f"{self.static_prefix}{CACHE_BOUNDARY_MARKER}{self.volatile_suffix}"

    @classmethod
    def split(cls, text: str) -> "CacheBoundary":
        if CACHE_BOUNDARY_MARKER in (text or ""):
            static, _, volatile = text.partition(CACHE_BOUNDARY_MARKER)
            return cls(static, volatile)
        return cls(text or "", "")


# --------------------------------------------------------------------------
# The cascade
# --------------------------------------------------------------------------


@dataclass
class CompressionReport:
    """Observability for one cascade run -- which layers fired, and why the
    ones that did not fire stayed idle."""

    tokens_before: int = 0
    tokens_after: int = 0
    target_tokens: int = 0
    stage1_compactions: int = 0
    stage2_summary_injected: bool = False
    stage2_summary_tokens: int = 0
    stage3_signature_prunes: int = 0
    evicted_cycles: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    facts: Optional[StateOfUnionFacts] = None

    @property
    def layers_applied(self) -> list[str]:
        applied = []
        if self.stage1_compactions:
            applied.append("micro")
        if self.stage2_summary_injected:
            applied.append("structured")
        if self.stage3_signature_prunes:
            applied.append("elastic")
        if self.evicted_cycles:
            applied.append("evict")
        return applied

    @property
    def changed(self) -> bool:
        return self.tokens_after < self.tokens_before


def _path_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if name not in {"read_file", "inspect_artifact"}:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            path = arguments.get("path")
            call_id = str(call.get("id") or "")
            if call_id and isinstance(path, str) and path:
                mapping[call_id] = path
    return mapping


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, default=str)


class CompressionCascade:
    """Ordered multi-stage compression over a working message list.

    ``compact`` mutates ``messages`` in place (the agent loop owns the list)
    and returns a :class:`CompressionReport` describing exactly which layers
    ran. Nothing raises: a stage that cannot act records itself in
    ``report.skipped`` and the cascade moves on.
    """

    def __init__(
        self,
        *,
        trigger_ratio: float = DEFAULT_TRIGGER_RATIO,  # context full-ratio at which the structured layer fires
        micro_max_tokens: int = DEFAULT_MICRO_MAX_TOKENS,
        micro_head_tokens: int = DEFAULT_MICRO_HEAD_TOKENS,
        micro_tail_tokens: int = DEFAULT_MICRO_TAIL_TOKENS,
        summary_tokens: int = STATE_OF_UNION_BUDGET_TOKENS,
        signature_keep_recent: int = DEFAULT_SIGNATURE_KEEP_RECENT,
        signature_min_tokens: int = DEFAULT_SIGNATURE_MIN_TOKENS,
    ) -> None:
        self.trigger_ratio = max(0.05, min(0.99, trigger_ratio))
        self.micro_max_tokens = micro_max_tokens
        self.micro_head_tokens = micro_head_tokens
        self.micro_tail_tokens = micro_tail_tokens
        self.summary_tokens = summary_tokens
        self.signature_keep_recent = max(0, signature_keep_recent)
        self.signature_min_tokens = max(1, signature_min_tokens)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _latest_user_index(messages: list[dict[str, Any]]) -> int:
        return max(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )

    @staticmethod
    def _leading_system_index(messages: list[dict[str, Any]]) -> int:
        return 0 if messages and messages[0].get("role") == "system" else -1

    @staticmethod
    def _pointer_for(message: dict[str, Any], paths: dict[str, str]) -> str:
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            path = paths.get(call_id)
            if path:
                return f"full output of read_file({path}) is recoverable by re-reading that path"
            return "re-run the tool to restore full output"
        if message.get("role") == "user":
            return "the user's original request remains in their own transcript"
        return "the full content is preserved in the session transcript"

    # -- stages ----------------------------------------------------------
    def stage1_micro(self, messages: list[dict[str, Any]], report: CompressionReport) -> None:
        """Truncate oversized payloads, newest-last, never the leading system
        message and never the current user request."""
        paths = _path_by_call_id(messages)
        latest_user = self._latest_user_index(messages)
        leading_system = self._leading_system_index(messages)
        for index, message in enumerate(messages):
            if index in (latest_user, leading_system) or not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"tool", "assistant"}:
                continue
            # Assistant messages carrying tool calls are needed verbatim enough
            # that only their prose is bounded (see runner_local's own compactor
            # for the argument-level handling); skip the rest.
            if role == "assistant" and message.get("tool_calls"):
                continue
            content = _message_text(message)
            if not content:
                continue
            compacted, changed = micro_compact(
                content,
                pointer=self._pointer_for(message, paths),
                max_tokens=self.micro_max_tokens,
                head_tokens=self.micro_head_tokens,
                tail_tokens=self.micro_tail_tokens,
                label=f"{role} output",
            )
            if changed:
                message["content"] = compacted
                report.stage1_compactions += 1

    def stage2_structured(
        self, messages: list[dict[str, Any]], report: CompressionReport, *, session_state: Any = None,
    ) -> None:
        """Inject the bounded State-of-the-Union summary directly after the
        leading system message (it must precede the detail it replaces)."""
        text, facts = build_state_of_union(messages, session_state=session_state, max_tokens=self.summary_tokens)
        report.facts = facts
        if not text:
            report.skipped["structured"] = "no durable facts found yet"
            return
        summary_message = {
            "role": "system",
            "content": text,
            "_tamfis_compression": "state_of_union",
        }
        insert_at = 1 if self._leading_system_index(messages) == 0 else 0
        messages.insert(insert_at, summary_message)
        report.stage2_summary_injected = True
        report.stage2_summary_tokens = estimate_tokens(text)

    def stage3_elastic(
        self,
        messages: list[dict[str, Any]],
        report: CompressionReport,
        *,
        signature_viewer: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        """Replace superseded file-read bodies with their signature view."""
        viewer = signature_viewer or signature_view
        paths = _path_by_call_id(messages)
        if not paths:
            report.skipped["elastic"] = "no file reads in context"
            return
        tool_indexes = [
            index for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "tool"
            and str(message.get("tool_call_id") or "") in paths
        ]
        candidates = tool_indexes[: max(0, len(tool_indexes) - self.signature_keep_recent)]
        if not candidates:
            report.skipped["elastic"] = "all file reads are still recent"
            return
        for index in candidates:
            message = messages[index]
            content = _message_text(message)
            if estimate_tokens(content) < self.signature_min_tokens:
                continue
            path = paths.get(str(message.get("tool_call_id") or ""))
            if not path:
                continue
            try:
                view = viewer(path)
            except Exception as exc:  # a viewer must never break the loop
                report.skipped["elastic"] = f"signature view failed: {type(exc).__name__}"
                return
            if not view:
                report.skipped["elastic"] = f"could not read {path}"
                continue
            message["content"] = (
                f"[signature view of {path} -- file body pruned from context]\n"
                f"{view}\n"
                f"[re-read {path} for the full body]"
            )
            report.stage3_signature_prunes += 1

    def _evict_cycles(self, messages: list[dict[str, Any]], report: CompressionReport, *, target_tokens: int, keep_recent: int) -> None:
        """Last resort inside the cascade: drop oldest completed cycles,
        protocol-safe (a tool result is never orphaned from its call)."""
        leading_system = self._leading_system_index(messages)
        # Count the injected State-of-the-Union message as durable: never evict
        # the layer that exists to survive eviction.
        protected = {leading_system}
        for index, message in enumerate(messages):
            if isinstance(message, dict) and message.get("_tamfis_compression") == "state_of_union":
                protected.add(index)
        guard = 0
        while estimate_tokens(_messages_text(messages)) > target_tokens and guard < 500:
            guard += 1
            latest_user = self._latest_user_index(messages)
            limit = max(1, len(messages) - keep_recent)
            removed = False
            index = 0
            while index < limit:
                if index in protected or index == latest_user:
                    index += 1
                    continue
                message = messages[index]
                if message.get("role") == "assistant" and message.get("tool_calls"):
                    call_ids = {str(call.get("id") or "") for call in message.get("tool_calls") or []}
                    end = index + 1
                    while end < len(messages):
                        candidate = messages[end]
                        if candidate.get("role") != "tool":
                            break
                        if call_ids and str(candidate.get("tool_call_id") or "") not in call_ids:
                            break
                        end += 1
                    del messages[index:end]
                else:
                    del messages[index]
                removed = True
                report.evicted_cycles += 1
                break
            if not removed:
                break

    # -- entry point -----------------------------------------------------
    def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        token_budget: int,
        target_tokens: Optional[int] = None,
        keep_recent: int = 6,
        session_state: Any = None,
        signature_viewer: Optional[Callable[[str], Optional[str]]] = None,
        final_trim: Optional[Callable[[list[dict[str, Any]], int], bool]] = None,
    ) -> CompressionReport:
        """Run the cascade until the context fits ``target_tokens`` (default:
        85% of ``token_budget``). Never raises; see the report for what ran."""
        report = CompressionReport()
        report.tokens_before = estimate_tokens(_messages_text(messages))
        report.target_tokens = int(target_tokens if target_tokens is not None else token_budget * 0.85)
        if not messages:
            report.tokens_after = 0
            return report

        # Stage 2 fires once the context is "full" by ratio (default 80%),
        # never later than the target we must reach -- whichever is higher.
        structured_trigger = max(report.target_tokens, int(token_budget * self.trigger_ratio))
        try:
            self.stage1_micro(messages, report)
            if estimate_tokens(_messages_text(messages)) > structured_trigger:
                self.stage2_structured(messages, report, session_state=session_state)
            else:
                report.skipped["structured"] = "context below trigger threshold"
            if estimate_tokens(_messages_text(messages)) > report.target_tokens:
                self.stage3_elastic(messages, report, signature_viewer=signature_viewer)
            else:
                report.skipped["elastic"] = "context below target budget"
            if estimate_tokens(_messages_text(messages)) > report.target_tokens:
                if final_trim is not None:
                    # A caller-supplied compactor (runner_local's own
                    # _trim_tool_outputs) already owns protocol-safe eviction;
                    # defer to it rather than double-evicting here.
                    try:
                        final_trim(messages, report.target_tokens)
                    except Exception as exc:
                        report.skipped["final_trim"] = f"{type(exc).__name__}: {exc}"
                else:
                    self._evict_cycles(
                        messages, report, target_tokens=report.target_tokens, keep_recent=keep_recent,
                    )
        except Exception as exc:  # pragma: no cover - defensive: never break a turn
            report.skipped["cascade"] = f"{type(exc).__name__}: {exc}"
        report.tokens_after = estimate_tokens(_messages_text(messages))
        return report


def _messages_text(messages: Iterable[dict[str, Any]]) -> str:
    return "\n".join(_message_text(message) for message in messages if isinstance(message, dict))
