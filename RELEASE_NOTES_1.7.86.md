# Tamfis-Code 1.7.86

## Fixed

- Make approved external reads effective through both the capability gateway
  and its wrapped MCP server. Approved commands such as `cat /etc/cron.d/...`
  no longer fail again with the original workspace-root error at dispatch.
- Handle missing provider configuration as an actionable task failure instead
  of raising an uncaught `ValueError` or endlessly replaying a saved task.
- Recovery suggestions now direct the user to `/doctor`, `/model`, and
  `/retry` rather than creating a new synthetic stream-fix objective.

## Verification

- Focused suite: 139 passed, 3 subtests passed.
