# Tamfis-Code 1.7.81

## Cron inspection and malformed-call recovery

- `crontab -l` is now recognized as a safe read-only inspection command.
- Cron inspection no longer gets trapped behind repeated read-only execution
  refusals.
- Malformed provider tool arguments now add explicit recovery guidance and
  prohibit verbatim retries.
- Added safety regression coverage.
