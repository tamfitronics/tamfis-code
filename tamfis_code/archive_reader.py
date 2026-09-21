"""Read-only inspection of ZIP/TAR archives, nested to any depth, without extracting anything.

`extract_archive` writes into the workspace and refuses anything over 5,000 files / 250 MB, so a large
archive (or a zip inside a zip) was unreadable in an audit: `read_file` says "binary" and the extractor is
a mutating tool that read-only turns are not offered. This module lists and reads members straight out of
the archive:

  * no total-size limit -- ZIPs are random access, so a multi-gigabyte archive lists instantly and only the
    member asked for is decompressed; output is paged so the model always gets a continuation offset;
  * nested archives to any depth -- name the chain with ``!/`` (``pack.zip!/data/inner.tar.gz!/notes.txt``);
    an inner archive is streamed to a temp file (memory up to SPOOL_MEMORY_BYTES) that is deleted on return;
  * nothing is ever written to the workspace, member names are lookup keys and never filesystem paths, so
    traversal / symlink / overwrite attacks do not apply.

Bounds exist per CALL (bytes scanned, characters returned, one inner archive), never on archive size.
"""
from __future__ import annotations

import fnmatch
import gzip
import io
import posixpath
import tarfile
import tempfile
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar", ".zip")
NEST_SEPARATOR = "!/"
MAX_NEST_DEPTH = 32
SPOOL_MEMORY_BYTES = 64 * 1024 * 1024
MAX_INNER_ARCHIVE_BYTES = 8 * 1024 ** 3      # one nested archive copied to a temp file
MAX_SCAN_BYTES = 1024 ** 3                   # decompressed bytes examined by ONE read call
DEFAULT_LIST_LIMIT = 200
MAX_LIST_LIMIT = 1000
DEFAULT_PAGE_LINES = 800
MAX_PAGE_LINES = 2000
MAX_RETURN_CHARS = 200_000
MAX_LINE_CHARS = 20_000


def is_archive_name(name: str) -> bool:
    return str(name or "").lower().endswith(ARCHIVE_SUFFIXES)


def split_chain(path: str) -> list[str]:
    """``a.zip!/b.zip!/c.txt`` -> ["a.zip", "b.zip", "c.txt"] (empty pieces dropped)."""
    return [piece for piece in str(path or "").split(NEST_SEPARATOR) if piece]


@dataclass(frozen=True)
class Entry:
    name: str
    size: int
    packed: int
    is_dir: bool


def _norm(name: str) -> str:
    text = posixpath.normpath(str(name or "").replace("\\", "/").lstrip("./") or ".")
    return "" if text == "." else text.lstrip("/")


class Archive:
    """One opened archive (zip or tar variant). Members are addressed by normalised name."""

    def __init__(self, source: "str | Path | BinaryIO", *, label: str) -> None:
        self.label = label
        self._zip: Optional[zipfile.ZipFile] = None
        self._tar: Optional[tarfile.TarFile] = None
        self._tar_members: dict[str, tarfile.TarInfo] = {}
        name = label.lower()
        try:
            is_zip = name.endswith(".zip") or zipfile.is_zipfile(source)  # type: ignore[arg-type]
            if not isinstance(source, (str, Path)):
                source.seek(0)  # is_zipfile leaves the stream mid-file
            if is_zip:
                self._zip = zipfile.ZipFile(source)  # type: ignore[arg-type]
            else:
                self._tar = tarfile.open(source if isinstance(source, (str, Path)) else None,  # type: ignore[arg-type]
                                         fileobj=None if isinstance(source, (str, Path)) else source, mode="r:*")
        except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as exc:
            raise ValueError(f"{label} could not be opened as an archive ({exc})") from exc

    def close(self) -> None:
        for handle in (self._zip, self._tar):
            if handle is not None:
                handle.close()

    def entries(self) -> Iterator[Entry]:
        if self._zip is not None:
            for info in self._zip.infolist():
                yield Entry(_norm(info.filename), info.file_size, info.compress_size, info.is_dir())
            return
        assert self._tar is not None
        if not self._tar_members:
            for member in self._tar.getmembers():
                self._tar_members[_norm(member.name)] = member
        for key, member in self._tar_members.items():
            yield Entry(key, member.size, member.size, member.isdir())

    def find(self, member: str) -> Optional[Entry]:
        wanted = _norm(member)
        return next((e for e in self.entries() if e.name == wanted), None)

    def open(self, member: str) -> BinaryIO:
        wanted = _norm(member)
        if self._zip is not None:
            for info in self._zip.infolist():
                if _norm(info.filename) == wanted:
                    try:
                        return self._zip.open(info)  # type: ignore[return-value]
                    except RuntimeError as exc:  # password-protected
                        raise ValueError(f"'{member}' is encrypted ({exc}); it cannot be read without the password") from exc
            raise FileNotFoundError(member)
        assert self._tar is not None
        list(self.entries())
        info = self._tar_members.get(wanted)
        handle = self._tar.extractfile(info) if info is not None and info.isfile() else None
        if handle is None:
            raise FileNotFoundError(member)
        return handle  # type: ignore[return-value]


def open_chain(root: Path, inner: list[str], stack: ExitStack) -> Archive:
    """Open ``root`` and every nested archive named in ``inner`` (each stays open until ``stack`` closes)."""
    if len(inner) > MAX_NEST_DEPTH:
        raise ValueError(f"archive nesting deeper than {MAX_NEST_DEPTH} levels")
    current = Archive(root, label=root.name)
    stack.callback(current.close)
    walked = root.name
    for name in inner:
        entry = current.find(name)
        if entry is None or entry.is_dir:
            raise FileNotFoundError(f"'{name}' is not a file inside {walked}")
        if not is_archive_name(entry.name):
            raise ValueError(f"'{name}' inside {walked} is not a ZIP/TAR archive")
        if entry.size > MAX_INNER_ARCHIVE_BYTES:
            raise ValueError(f"'{name}' is {entry.size:,} bytes, over the {MAX_INNER_ARCHIVE_BYTES:,}-byte nested-archive limit")
        spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_MEMORY_BYTES)
        stack.callback(spool.close)
        with current.open(name) as incoming:
            copied = 0
            while True:
                block = incoming.read(1 << 20)
                if not block:
                    break
                copied += len(block)
                if copied > MAX_INNER_ARCHIVE_BYTES:
                    raise ValueError(f"'{name}' expands past the nested-archive limit")
                spool.write(block)
        spool.seek(0)
        walked = f"{walked}{NEST_SEPARATOR}{name}"
        current = Archive(spool, label=entry.name)  # type: ignore[arg-type]
        stack.callback(current.close)
    return current


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def list_members(archive: Archive, chain_label: str, *, pattern: str = "", offset: int = 1,
                 limit: int = DEFAULT_LIST_LIMIT) -> str:
    limit = max(1, min(MAX_LIST_LIMIT, int(limit or DEFAULT_LIST_LIMIT)))
    offset = max(1, int(offset or 1))
    files = [e for e in archive.entries() if not e.is_dir]
    if pattern:
        needle = pattern.lower()
        files = [e for e in files if fnmatch.fnmatch(e.name.lower(), needle) or needle in e.name.lower()]
    total_bytes = sum(e.size for e in files)
    page = files[offset - 1:offset - 1 + limit]
    lines = [
        f"{_human(e.size):>10}  {e.name}" + ("   [archive: open it with path=\"" + chain_label + NEST_SEPARATOR + e.name + "\"]"
                                             if is_archive_name(e.name) else "")
        for e in page
    ]
    end = offset - 1 + len(page)
    head = (
        f"[{chain_label}: {len(files):,} files{' matching ' + repr(pattern) if pattern else ''}, "
        f"{_human(total_bytes)} uncompressed. Showing {offset}-{max(end, offset)}."
    )
    if end < len(files):
        head += f" Continue with offset={end + 1}."
    else:
        head += " End of listing."
    return head + " Read a member with member=\"<name>\".]\n" + "\n".join(lines)


def read_member(archive: Archive, chain_label: str, member: str, *, offset: int = 1,
                limit: int = DEFAULT_PAGE_LINES) -> str:
    entry = archive.find(member)
    if entry is None or entry.is_dir:
        near = [e.name for e in archive.entries() if not e.is_dir and posixpath.basename(e.name) == posixpath.basename(_norm(member))][:5]
        hint = f" Did you mean: {', '.join(near)}?" if near else " List the archive first (omit member)."
        return f"Error: '{member}' is not a file in {chain_label}.{hint}"
    if is_archive_name(entry.name):
        return (f"'{entry.name}' is itself an archive ({_human(entry.size)}). Open it by passing "
                f"path=\"{chain_label}{NEST_SEPARATOR}{entry.name}\" (omit member to list it).")
    limit = max(1, min(MAX_PAGE_LINES, int(limit or DEFAULT_PAGE_LINES)))
    offset = max(1, int(offset or 1))
    scanned = 0
    chars = 0
    shown: list[str] = []
    line_no = 0
    more = False
    with archive.open(entry.name) as packed:
        # A lone .gz member (e.g. data-000.jsonl.gz) is read decompressed, like zcat.
        raw: BinaryIO = gzip.GzipFile(fileobj=packed) if entry.name.lower().endswith(".gz") else packed  # type: ignore[assignment]
        try:
            head = raw.read(8000)
        except (OSError, EOFError) as exc:
            return f"Error: '{entry.name}' could not be decompressed ({exc})"
        if b"\x00" in head:
            return (f"'{entry.name}' in {chain_label} is a binary file ({_human(entry.size)}); only text members can be "
                    "read. Its name, size and place in the archive are all that can be reported.")
        text = io.TextIOWrapper(io.BufferedReader(_Prefixed(head, raw)), encoding="utf-8", errors="replace", newline=None)
        while True:
            line = text.readline(MAX_LINE_CHARS)
            if not line:
                break
            if len(line) == MAX_LINE_CHARS and not line.endswith("\n"):
                # An over-long line (minified JSON/HTML): keep its head, skip the rest without holding it.
                skipped = 0
                while True:
                    rest = text.readline(1 << 20)
                    skipped += len(rest)
                    scanned += len(rest)
                    if not rest or rest.endswith("\n") or scanned > MAX_SCAN_BYTES:
                        break
                line = f"{line} [... line continues, {skipped:,} more chars ...]\n"
            line_no += 1
            scanned += len(line)
            if scanned > MAX_SCAN_BYTES:
                more = True
                break
            if line_no < offset:
                continue
            if len(shown) >= limit or chars + len(line) > MAX_RETURN_CHARS:
                more = True
                break
            shown.append(f"{line_no}: {line}" if line.endswith("\n") else f"{line_no}: {line}\n")
            chars += len(line)
    if not shown:
        return f"[Offset {offset} is beyond the end of {entry.name} ({line_no} lines).]" if line_no else f"[{entry.name} is empty.]"
    end = offset + len(shown) - 1
    tail = f" Continue with offset={end + 1}." if more else " End of file."
    return f"[{chain_label}{NEST_SEPARATOR}{entry.name}: lines {offset}-{end}.{tail}]\n{''.join(shown)}"


class _Prefixed(io.RawIOBase):
    """``head`` (already read for binary sniffing) followed by the rest of ``raw``."""

    def __init__(self, head: bytes, raw: BinaryIO) -> None:
        self._head = memoryview(head)
        self._raw = raw

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        if len(self._head):
            n = min(len(buffer), len(self._head))
            buffer[:n] = self._head[:n]
            self._head = self._head[n:]
            return n
        data = self._raw.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def read_archive(root: Path, inner: list[str], member: str = "", *, pattern: str = "", offset: int = 1,
                 limit: Optional[int] = None) -> str:
    """List (no ``member``) or read one text ``member`` of the archive at ``root`` -> ``inner`` chain."""
    label = root.name + "".join(NEST_SEPARATOR + part for part in inner)
    with ExitStack() as stack:
        try:
            archive = open_chain(root, inner, stack)
        except (ValueError, FileNotFoundError) as exc:
            return f"Error: {exc}"
        if member:
            return read_member(archive, label, member, offset=offset, limit=limit or DEFAULT_PAGE_LINES)
        return list_members(archive, label, pattern=pattern, offset=offset, limit=limit or DEFAULT_LIST_LIMIT)
