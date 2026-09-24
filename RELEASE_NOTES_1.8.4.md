# Tamfis-Code 1.8.4

## Deep discovery and resilient streaming

- Allow `list_directory` to recurse to the deepest reachable directory with
  `depth=0`; the previous hard depth-3 ceiling is removed.
- Remove the automatic external `find` depth-20 ceiling while retaining the
  approved-path and same-filesystem safety checks.
- Ignore empty OpenAI-compatible stream choice envelopes instead of raising
  `list index out of range` and aborting provider recovery.
- Add regression coverage for deep directory discovery, malformed depth input,
  and empty stream chunks.

## Verification

- Focused MCP, provider protocol, workspace-scope, and malformed-tool suites
  pass.
