# Tamfis-Code 1.8.1

## Production hardening

- Normalize provider variants of interactive question payloads before schema
  validation, preventing malformed `ask_user_question` responses from causing
  failover loops.
- Preserve interrupted and cancelled terminal checkpoints so `continue`
  resumes verification instead of silently reporting a completed task.
- Keep resumable checkpoints during idle-state compaction when they represent
  failed or interrupted execution.
- Add regression coverage for malformed interactive tool arguments and
  interrupted-plan recovery.

## Verification

- Focused Tamfis-Code agent, swarm, resume, and capability-gateway tests pass.
