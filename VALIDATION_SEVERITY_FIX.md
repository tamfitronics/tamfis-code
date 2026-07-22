# Tamfis-Code validation severity repair

This patch preserves the standalone Tier IV boundary and changes only completion validation severity.

- `pass`: evidence checks pass.
- `warning`: evidence is incomplete; the task completes with the existing validation caveat.
- `error`: the response claims that repository inspection/review occurred but no successful tool evidence supports the claim; the task fails and its checkpoint is preserved.

This restores established behaviours for denied tools, hook-blocked operations, fake textual tool calls, no-mutation caveats, planning fixtures, and legacy resume summaries, while preventing unsupported cross-workspace audit claims from being marked completed.
