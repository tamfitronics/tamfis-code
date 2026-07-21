"""OpenHands-class runtime for Tamfis-Code.

This package provides the production primitives used by the CLI and agent
server: immutable events, conversations, workspaces, tools, skills, leases,
secrets, delegation, automations and replay.
"""
from .events import Event, EventKind, EventStore
from .conversation import Conversation, ConversationManager, ConversationState
from .workspace import LocalWorkspace, SSHWorkspace, RemoteWorkspace
from .tools import Tool, ToolRegistry, ToolResult
from .skills import Skill, SkillRegistry

__all__ = [
    "Event", "EventKind", "EventStore", "Conversation", "ConversationManager",
    "ConversationState", "LocalWorkspace", "SSHWorkspace", "RemoteWorkspace",
    "Tool", "ToolRegistry", "ToolResult", "Skill", "SkillRegistry",
]
