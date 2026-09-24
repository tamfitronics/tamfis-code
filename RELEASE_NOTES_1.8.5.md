# Tamfis-Code 1.8.5

## Session history aliases and slash-command parity

- `/resume` without an id now opens the existing-session picker inside the
  interactive REPL instead of silently selecting the newest session.
- `/history`, `/sessions`, and `/chats` now correctly browse the same picker;
  `/continue` remains an explicit resume alias.
- Resume aliases are listed in `/help` as well as tab completion.
- Added end-to-end regression coverage for selecting an existing session via
  `/history`.

## Verification

- Resume and slash-command suites: 78 passed, 27 subtests passed.
