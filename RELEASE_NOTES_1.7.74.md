# Tamfis-Code 1.7.74

## Safety and audit execution

- Advice-only repository audits now explicitly prohibit unrequested training,
  builds, installs, servers, migrations, and other long-running workloads.
- The coding contract requires bounded execution only when the user has
  explicitly requested an expensive workload, and forbids claiming a smoke
  test from an unobserved or timed-out command.

This release is a prompt-contract hardening change; it does not alter the
workspace permission boundary or authorize additional commands.
