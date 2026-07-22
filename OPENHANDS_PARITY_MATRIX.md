# Tamfis-Code OpenHands Parity Matrix

Version 0.6.0 consolidates OpenHands-class architecture into Tamfis-Code 0.4.43.

| Capability | Implementation |
|---|---|
| Immutable typed events and replay | `tamfis_code.openhands.events` |
| Conversations and lifecycle | `conversation.py` |
| Local no-Docker workspace | `workspace.LocalWorkspace` |
| SSH and remote workspaces | `SSHWorkspace`, `RemoteWorkspace` |
| Terminal, file, browser and Git tools | `tools.py` |
| Snapshots and restore | `LocalWorkspace.snapshot/restore` |
| Security analysis and approvals | `security.py` plus existing Tamfis approval engine |
| Secret vault | `security.SecretVault` |
| Skills and project extensions | `skills.py` |
| Multi-agent delegation | `delegation.py` plus existing swarm |
| Conversation leases | `leases.py` |
| Scheduled automations | `automation.py` |
| REST and WebSocket agent server | `agent_server.py` |
| MCP integration | existing `tamfis_code.mcp` and MCP stdio server |
| Context condensation | existing `runner_local._trim_tool_outputs` and checkpoints |
| Professional streaming renderer | buffered renderer in `render.py` |
| Provider boundary | NVIDIA, OpenRouter, Hugging Face only; Tier IV excluded from standalone |

Docker is deliberately not included because the deployment constraint is no Docker. Local, SSH and Remote API workspaces cover the supported runtime modes.
