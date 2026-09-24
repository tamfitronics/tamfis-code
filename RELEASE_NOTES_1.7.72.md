# Tamfis-Code 1.7.72

## Markdown-path checkpoint recovery

- Normalized Markdown backticks and quotes around paths in saved plan steps.
- Resumed plans such as `Inventory `/home`` now reach the deterministic
  `list_directory` recovery instead of being treated as unsupported provider
  prose.
- Added regression coverage for the exact rendered-plan format.

## Verification

- 224 focused tests passed, plus 17 subtests.
- Python compilation passed.
