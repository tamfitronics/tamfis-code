# Tamfis-Code 1.8.2

## Approved external diagnostics and WordPress CLI recovery

- Allow explicitly approved external diagnostic commands to run after the
  workspace-scope approval gate, while keeping unapproved commands blocked.
- Permit approved direct reads such as `/etc/cron.d/*` without routing them
  through the workspace-only file resolver.
- Detect WP-CLI's root safety refusal for read-only queries and retry the same
  approved query with `--allow-root`, reporting the recovery in tool evidence.
- Add regression coverage for root-guard recovery, approved external reads,
  and continued workspace-boundary enforcement.

## Verification

- `tests/test_mcp.py`: 71 passed.
- Workspace, cwd, and sandbox suites: 63 passed, 3 subtests passed.
