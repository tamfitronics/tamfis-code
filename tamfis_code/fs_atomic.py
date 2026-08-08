"""Atomic-write helper that preserves an existing file's mode/ownership.

os.replace()-based atomic writes (write to a sibling temp file, then rename
over the target) are the right way to avoid partial-write corruption and to
follow symlinks safely -- but the rename swaps inodes outright. Unless the
temp file is explicitly given the target's mode/owner first, the replaced
file silently inherits tempfile.mkstemp's restrictive 0600 mode and the
current process's uid/gid instead of the original file's. Confirmed live:
user-reported permissions/ownership loss on files edited by write_file/
edit_file (mode 0600, owner "nobody:nobody" -- the process identity, not
the original file's).
"""
from __future__ import annotations

import os
import stat
from pathlib import Path


def _process_default_file_mode() -> int:
    """The mode an ordinary open()/Path.write_text() call would produce:
    0666 minus the process umask. os.umask() has no direct getter -- the
    standard trick is to set a throwaway value and read back the previous
    one, then immediately restore it. Cheap and safe (no window where a
    real file is created under the wrong umask; this never touches a file,
    only the process-wide umask register)."""
    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def preserve_existing_metadata(temp_path: str | Path, target_path: str | Path) -> None:
    """Normalize temp_path's mode/owner to what the final file should have,
    before os.replace(temp_path, target_path) swaps it in. Call this AFTER
    writing temp_path's content and BEFORE the replace -- os.replace()
    preserves whatever mode/owner the source (temp_path) already has, so
    this is the only point where it can be corrected.

    Two cases:
    - target_path already exists: copy its current mode/owner onto
      temp_path, so editing a file never silently changes its permissions
      or ownership.
    - target_path is brand new: mkstemp() deliberately always uses 0600
      regardless of umask (the right call for an actual temp file, since
      temp files are private-by-default for security) -- but once that
      inode becomes the permanent file via replace, 0600 would silently
      make every newly-created file owner-only, which is not what any
      other file-creation path in this tool (or a plain `open()` call)
      does. Reset to the process's ordinary umask-derived default instead.

    Ownership restoration (os.chown) requires root or CAP_CHOWN and is
    best-effort: a non-privileged process cannot change a file's owner to
    someone else's, full stop, so failure here is expected and silent
    rather than surfaced as a tool error -- the mode restoration (which
    virtually always succeeds, since it only requires owning the file or
    running as root) is the fix that matters for the overwhelmingly common
    case of a single-user workspace.

    Confirmed live: user-reported permissions/ownership loss on files
    edited by write_file/edit_file (mode 0600, owner "nobody:nobody" --
    the process identity, not the original file's).
    """
    target = Path(target_path)
    try:
        original = target.stat()
    except OSError:
        try:
            os.chmod(temp_path, _process_default_file_mode())
        except OSError:
            pass
        return

    try:
        os.chmod(temp_path, stat.S_IMODE(original.st_mode))
    except OSError:
        pass

    try:
        os.chown(temp_path, original.st_uid, original.st_gid)
    except OSError:
        pass
