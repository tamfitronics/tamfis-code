# Tamfis-Code 0.6.0 — OpenHands-Class Runtime

Built directly from the uploaded Tamfis-Code 0.4.43 baseline.

## Preserved

- NVIDIA → OpenRouter → Hugging Face standalone routing.
- Tier IV excluded from standalone execution.
- Ollama remains intentionally removed.
- Approval policies, workspace scope enforcement, checkpoints, evidence validation and resume behaviour.

## Added

- Append-only typed event log and deterministic replay.
- Conversation lifecycle with pause, resume and cancellation.
- Local no-Docker workspace, SSH workspace and Remote API workspace.
- Workspace snapshots and restore.
- Tool registry with terminal, file editing, browser, Git and snapshot tools.
- Skills loader for `.tamfis/skills` project extensions.
- Security analyser and file-backed secret vault.
- Multi-agent delegation events and bounded concurrency.
- Exclusive conversation/workspace leases.
- Scheduled automation store and scheduler.
- FastAPI REST and WebSocket agent server.
- Adapter from the existing proven Tamfis local agent loop into the new event runtime.
- Buffered professional Markdown streaming and structured execution plans.

## Server

```bash
tamfis-code-server --host 127.0.0.1 --port 9600
```

The server exposes health, conversations, events, messages, agent execution, tools, pause/resume/cancel, workspace browsing, snapshots and WebSocket event streaming.
