from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from .events import Event, EventKind, EventStore
from .leases import Lease, LeaseManager
from .security import SecurityAnalyzer
from .skills import SkillRegistry
from .tools import ToolRegistry, default_registry
from .workspace import LocalWorkspace


class ConversationState(str, Enum):
    CREATED="created"; RUNNING="running"; WAITING="waiting"; PAUSED="paused"; COMPLETED="completed"; FAILED="failed"; CANCELLED="cancelled"


@dataclass(slots=True)
class Conversation:
    id: str
    workspace: LocalWorkspace
    event_store: EventStore
    tools: ToolRegistry
    state: ConversationState = ConversationState.CREATED
    objective: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str,Any] = field(default_factory=dict)
    _pause: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _cancelled: bool = False

    def __post_init__(self): self._pause.set()
    def emit(self,kind:EventKind,payload:dict[str,Any],**kwargs)->Event:
        self.updated_at=time.time(); return self.event_store.emit(self.id,kind,payload,**kwargs)
    def set_state(self,state:ConversationState,reason:str=""):
        self.state=state; self.emit(EventKind.STATE_CHANGED,{"state":state.value,"reason":reason})
    def send_message(self,message:str,actor:str="user"):
        self.objective=message; return self.emit(EventKind.USER_MESSAGE,{"content":message},actor=actor)
    def pause(self): self._pause.clear(); self.set_state(ConversationState.PAUSED)
    def resume(self): self._pause.set(); self.set_state(ConversationState.RUNNING)
    def cancel(self): self._cancelled=True; self._pause.set(); self.set_state(ConversationState.CANCELLED)
    async def wait_if_paused(self): await self._pause.wait();
    async def invoke_tool(self,name:str,arguments:dict[str,Any],*,approval_policy:str="ask"):
        await self.wait_if_paused()
        if self._cancelled: raise asyncio.CancelledError
        analyzer=SecurityAnalyzer(); decision=analyzer.analyze(name,arguments,approval_policy=approval_policy)
        if not decision.allowed:
            self.emit(EventKind.ERROR,{"tool":name,"error":decision.reason,"risk":decision.risk}); return {"ok":False,"error":decision.reason}
        correlation=uuid.uuid4().hex; self.emit(EventKind.TOOL_STARTED,{"tool":name,"arguments":arguments,"risk":decision.risk},correlation_id=correlation)
        result=await self.tools.invoke(name,arguments)
        self.emit(EventKind.TOOL_FINISHED,{"tool":name,"result":result.as_dict()},correlation_id=correlation)
        return result.as_dict()
    def replay(self,after:int=0): return self.event_store.read(self.id,after=after)


class ConversationManager:
    def __init__(self,state_root:str|Path):
        self.state_root=Path(state_root).expanduser().resolve(); self.state_root.mkdir(parents=True,exist_ok=True)
        self.events=EventStore(self.state_root/'events'); self.leases=LeaseManager(self.state_root/'leases.json'); self._active:dict[str,Conversation]={}
    def create(self,workspace:str|Path,*,conversation_id:str|None=None,metadata:dict[str,Any]|None=None)->Conversation:
        cid=conversation_id or uuid.uuid4().hex; ws=LocalWorkspace(workspace,state_dir=self.state_root/'workspaces'/cid); conv=Conversation(cid,ws,self.events,default_registry(ws),metadata=metadata or {}); self._active[cid]=conv; conv.emit(EventKind.STATE_CHANGED,{"state":conv.state.value}); return conv
    def get(self,conversation_id:str)->Conversation:
        if conversation_id in self._active:return self._active[conversation_id]
        raise KeyError(conversation_id)
    def list(self): return [{"id":c.id,"state":c.state.value,"workspace":str(c.workspace.root),"objective":c.objective,"updated_at":c.updated_at} for c in sorted(self._active.values(),key=lambda c:c.updated_at,reverse=True)]
