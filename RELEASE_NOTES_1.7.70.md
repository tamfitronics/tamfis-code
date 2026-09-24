# Tamfis-Code 1.7.70

## Read-only audit and recovery hardening

- Exposed `execute_command` during read-only inspection turns with the same
  strict non-mutating command classifier and dispatch guard.
- Safe Python reports, checkpoint inspection, compile checks, and test
  discovery can now produce real observed evidence instead of being rejected
  as unavailable or ending with an evidence-validation failure.
- Preserved blocking for writes, shell control, command substitution,
  arbitrary script execution, and destructive commands.
- Added regression coverage for the read-only tool contract.

## Release verification

- Version metadata is synchronized at `1.7.70`.
- The release must pass the focused recovery, safety, rendering, and routing
  suites before installation.
