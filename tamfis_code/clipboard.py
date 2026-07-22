"""Copy text to the user's system clipboard from a terminal session.

Uses OSC 52, a terminal escape sequence most modern terminal emulators
support (iTerm2, kitty, Alacritty, Windows Terminal, VTE-based GNOME
Terminal, tmux) that sets the system clipboard through the terminal itself.
Critically, this works over a plain SSH session with no GUI, X11, or
Wayland session on the machine running tamfis-code -- unlike xclip/wl-copy/
pbcopy, which all require local clipboard tooling a remote box usually
doesn't have.
"""

from __future__ import annotations

import base64

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
