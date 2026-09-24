# Tamfis-Code 1.7.77

## Provider-safe questions and repository orientation

- Decode provider tool-call JSON strings for `ask_user_question` arrays at
  the local and remote capability gateways, including nested options, before
  strict schema validation and dispatch.
- Preserve fail-closed validation for malformed or unrelated tool arguments.
- Require bounded repository-tree orientation before reading an unresolved
  path, so agents use discovered paths instead of guessing filenames.
- Added regression coverage for native and remote question dispatch and for
  top-level JSON-encoded question arrays.
