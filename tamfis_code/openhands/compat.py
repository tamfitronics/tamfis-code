"""Compatibility names mirroring the public OpenHands SDK vocabulary."""
from .conversation import Conversation, ConversationManager
from .events import Event, EventStore
from .skills import Skill as AgentSkill, SkillRegistry
from .tools import Tool, ToolRegistry
from .workspace import BaseWorkspace, LocalWorkspace, RemoteWorkspace, SSHWorkspace

Agent = object
LLM = object
__all__=["Conversation","ConversationManager","Event","EventStore","AgentSkill","SkillRegistry","Tool","ToolRegistry","BaseWorkspace","LocalWorkspace","RemoteWorkspace","SSHWorkspace"]
