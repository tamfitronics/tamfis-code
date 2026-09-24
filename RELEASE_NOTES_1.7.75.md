# Tamfis-Code 1.7.75

## Validation and resume recovery

- Repository-wide shell validation now uses the safety-approved `find -exec`
  syntax-check form instead of an unsupported `xargs` interpreter fan-out.
- A completed plan with failed validation remains resumable as a
  validation-only checkpoint; `continue` no longer reports completion without
  running the missing check or replays the completed edit.
