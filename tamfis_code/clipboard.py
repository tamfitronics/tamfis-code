"""Copy text to, and read an image from, the user's system clipboard.

Text OUT uses OSC 52, a terminal escape sequence most modern terminal
emulators support (iTerm2, kitty, Alacritty, Windows Terminal, VTE-based
GNOME Terminal, tmux) that sets the system clipboard through the terminal
itself. Critically, this works over a plain SSH session with no GUI, X11,
or Wayland session on the machine running tamfis-code -- unlike
xclip/wl-copy/pbcopy, which all require local clipboard tooling a remote
box usually doesn't have.

Image IN has no equivalent terminal-escape-sequence trick: no widely
supported escape sequence lets a bare TTY program read arbitrary binary
(non-text) clipboard content back from the terminal, and image bytes
never flow through the TTY character stream the way pasted text does --
they only ever exist in the OS clipboard, reachable through each
platform's own native clipboard tool. read_clipboard_image therefore
shells out to pbpaste (macOS), wl-paste/xclip (Linux, whichever
compositor is running), or PowerShell (Windows) -- the same well-
established approach every desktop AI coding tool uses for "paste an
image" -- and simply reports "not supported here" over a session with no
attached GUI/X11/Wayland session (a plain headless SSH box), rather than
guessing at a fragile terminal-protocol workaround.
"""

from __future__ import annotations

import base64
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console

# Some terminals (older tmux in particular) cap how much a single OSC 52
# payload they'll accept and silently drop anything larger, with no ack
# channel to detect that -- there's nothing to negotiate a real limit
# against, so this just caps to a size known to work broadly rather than
# gambling a large paste does nothing at all.
MAX_CLIPBOARD_CHARS = 100_000


def copy_to_clipboard(console: Console, text: str) -> bool:
    """Write `text` to the system clipboard via an OSC 52 escape sequence.

    Returns True if the sequence was written -- there's no ack channel, so
    this is best-effort and does not guarantee the terminal actually applied
    it. Returns False if there's no attached terminal (e.g. piped/redirected
    output), since firing the escape sequence at a non-terminal would just
    corrupt whatever's consuming that stream instead of copying anything.
    """
    if not getattr(console, "is_terminal", False):
        return False
    truncated = text[:MAX_CLIPBOARD_CHARS]
    encoded = base64.b64encode(truncated.encode("utf-8")).decode("ascii")
    console.file.write(f"\x1b]52;c;{encoded}\x07")
    console.file.flush()
    return True


# Mirrors runner_local.py's MAX_VISION_ATTACHMENT_BYTES -- a clipboard
# image feeds the exact same vision-attachment pipeline (build_vision_
# content_blocks), so a larger capture is rejected here, immediately and
# with a clear reason, rather than being silently dropped several layers
# deeper once the turn actually runs.
MAX_CLIPBOARD_IMAGE_BYTES = 10 * 1024 * 1024

_SUBPROCESS_TIMEOUT_SECONDS = 10.0


def _run(args: list[str], *, input_bytes: Optional[bytes] = None) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(
        args, input=input_bytes, capture_output=True, timeout=_SUBPROCESS_TIMEOUT_SECONDS, check=False,
    )


def _read_clipboard_image_macos() -> tuple[Optional[bytes], str]:
    # AppleScript's `the clipboard as «class PNGf»` raises (a non-zero
    # osascript exit, no image bytes on stdout) when the clipboard holds
    # no image -- caught here as the ordinary "nothing to paste" case,
    # not a tool-missing error, since osascript itself always exists on
    # macOS. Writes straight to a temp file from within the script
    # (rather than parsing AppleScript's own textual hex-dump form on
    # stdout) so no separate encoding step can corrupt the bytes.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    script = (
        'try\n'
        f'  set outFile to (open for access POSIX file "{tmp_path}" with write permission)\n'
        '  set eof outFile to 0\n'
        '  write (the clipboard as «class PNGf») to outFile\n'
        '  close access outFile\n'
        'on error\n'
        '  try\n'
        f'    close access (open for access POSIX file "{tmp_path}")\n'
        '  end try\n'
        '  return "NO_IMAGE"\n'
        'end try\n'
    )
    try:
        result = _run(["osascript", "-e", script])
        data = Path(tmp_path).read_bytes()
        if b"NO_IMAGE" in result.stdout or not data:
            return None, "no image on the clipboard"
        return data, ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"osascript failed: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _read_clipboard_image_linux() -> tuple[Optional[bytes], str]:
    # A Wayland session (XDG_SESSION_TYPE/WAYLAND_DISPLAY) uses wl-paste;
    # anything else (X11, or an ambiguous/unset session type) tries xclip,
    # matching how each tool actually talks to its respective display
    # protocol -- wl-paste against an X11-only session (or vice versa)
    # just hangs or errors rather than transparently falling back.
    is_wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or os.environ.get("XDG_SESSION_TYPE") == "wayland"
    if is_wayland and shutil.which("wl-paste"):
        try:
            types = _run(["wl-paste", "--list-types"]).stdout.decode("utf-8", "replace")
            if "image/" not in types:
                return None, "no image on the clipboard"
            mime = next((line for line in types.splitlines() if line.startswith("image/")), "image/png")
            result = _run(["wl-paste", "--type", mime])
            if result.returncode != 0 or not result.stdout:
                return None, "no image on the clipboard"
            return result.stdout, ""
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"wl-paste failed: {exc}"
    if shutil.which("xclip"):
        try:
            targets = _run(["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"]).stdout.decode("utf-8", "replace")
            available = [line for line in targets.splitlines() if line.startswith("image/")]
            if not available:
                return None, "no image on the clipboard"
            mime = "image/png" if "image/png" in available else available[0]
            result = _run(["xclip", "-selection", "clipboard", "-t", mime, "-o"])
            if result.returncode != 0 or not result.stdout:
                return None, "no image on the clipboard"
            return result.stdout, ""
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"xclip failed: {exc}"
    return None, (
        "no clipboard image tool found -- install wl-clipboard (Wayland) "
        "or xclip (X11), e.g. `apt install xclip`"
    )


def _read_clipboard_image_windows() -> tuple[Optional[bytes], str]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    script = (
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
        "if ([System.Windows.Forms.Clipboard]::ContainsImage()) { "
        f"[System.Windows.Forms.Clipboard]::GetImage().Save('{tmp_path}', "
        "[System.Drawing.Imaging.ImageFormat]::Png); "
        "Write-Output 'OK' "
        "} else { Write-Output 'NO_IMAGE' }"
    )
    try:
        result = _run(["powershell", "-NoProfile", "-Command", script])
        data = Path(tmp_path).read_bytes()
        if b"NO_IMAGE" in result.stdout or not data:
            return None, "no image on the clipboard"
        return data, ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"PowerShell clipboard read failed: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def read_clipboard_image() -> tuple[Optional[bytes], str]:
    """Best-effort read of an image (normalized to PNG bytes) off the
    user's OS clipboard, dispatched by platform. Returns (data, "") on
    success, or (None, reason) -- never raises. `reason` is meant to be
    shown to the user directly (e.g. "no clipboard image tool found...",
    "no image on the clipboard", or a subprocess failure detail), so it
    is always a complete, human-readable sentence fragment.

    A plain headless SSH session with no GUI/X11/Wayland attached simply
    has no OS clipboard to read at all -- pbpaste/wl-paste/xclip/
    PowerShell all fail or hang the same way any other locally-installed
    tool would on a box with nothing to talk to, which surfaces here as
    an ordinary "tool not found"/timeout reason rather than a special
    case, since there genuinely is nothing more specific to say.
    """
    system = platform.system()
    if system == "Darwin":
        return _read_clipboard_image_macos()
    if system == "Windows":
        return _read_clipboard_image_windows()
    return _read_clipboard_image_linux()
