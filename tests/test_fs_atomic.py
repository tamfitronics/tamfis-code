"""Regression coverage for fs_atomic.preserve_existing_metadata.

Live-reported bug: files edited via write_file/edit_file (mcp.py's
_atomic_write_text, an os.replace()-based atomic write) silently lost their
original mode/owner -- os.replace() swaps inodes, so the replacement
inherited tempfile.mkstemp's restrictive 0600 mode and the running
process's uid/gid ("nobody:nobody") instead of the original file's.
"""
import os
import stat
import tempfile
from pathlib import Path

from tamfis_code.fs_atomic import preserve_existing_metadata


def test_preserves_mode_of_an_existing_target(tmp_path: Path):
    target = tmp_path / "existing.txt"
    target.write_text("original")
    os.chmod(target, 0o640)

    fd, temp_name = tempfile.mkstemp(dir=str(tmp_path))
    os.write(fd, b"new content")
    os.close(fd)
    assert stat.S_IMODE(os.stat(temp_name).st_mode) == 0o600  # mkstemp's default

    preserve_existing_metadata(temp_name, target)
    os.replace(temp_name, target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert target.read_text() == "new content"


def test_new_target_gets_process_default_mode_not_mkstemp_0600(tmp_path: Path):
    target = tmp_path / "brand_new.txt"  # does not exist yet

    fd, temp_name = tempfile.mkstemp(dir=str(tmp_path))
    os.close(fd)

    preserve_existing_metadata(temp_name, target)

    mode = stat.S_IMODE(os.stat(temp_name).st_mode)
    assert mode != 0o600, f"new file left at mkstemp's restrictive default: {oct(mode)}"
    # Matches what an ordinary open()/Path.write_text() call would produce.
    umask = os.umask(0)
    os.umask(umask)
    assert mode == (0o666 & ~umask)
    os.unlink(temp_name)


def test_nonexistent_temp_path_does_not_raise(tmp_path: Path):
    # Defensive: a caller passing a path that vanished between write and
    # replace (e.g. a concurrent cleanup) must not crash the whole write --
    # this helper's failures are always best-effort/silent by design.
    target = tmp_path / "existing.txt"
    target.write_text("x")
    preserve_existing_metadata(str(tmp_path / "does-not-exist.tmp"), target)
