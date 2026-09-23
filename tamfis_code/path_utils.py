"""Small, shared path-input normalisation helpers.

Tool arguments are data, not shell syntax.  Models nevertheless commonly
return a quoted absolute path (``\"/home/project\"``).  Cleaning that once at
the tool boundary prevents the workspace root from being prepended to a path
that was absolute before its transport quotes were removed.
"""

from __future__ import annotations


def clean_path_argument(value: object) -> str:
    """Return a path argument with transport quotes removed, without rewriting it."""
    text = str(value or "").strip()
    # Remove balanced wrappers repeatedly: JSON/string adapters can leave an
    # extra pair around an already quoted path.
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text
