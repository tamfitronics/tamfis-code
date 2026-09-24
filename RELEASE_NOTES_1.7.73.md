# Tamfis-Code 1.7.73

## Active-task follow-up routing

- Classify live follow-up messages before queueing them.
- Additive requests such as “also add tests” are marked as `append` and
  delivered to the active task at the next safe orchestration boundary.
- Direction changes are marked as `replace` so the running agent revises its
  plan instead of merely acknowledging the message.
- Explicitly separate/new tasks are marked as `deferred` and remain for the
  next task rather than being injected into the current task.
- Updated terminal feedback so users can distinguish “added to the active
  task” from a genuinely deferred follow-up.

## Verification

- Live-input, in-flight branching, and stale-stop regression suites pass.
