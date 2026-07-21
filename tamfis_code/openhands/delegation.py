from __future__ import annotations
import asyncio, uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from .events import EventKind
from .conversation import Conversation

@dataclass(slots=True)
class DelegatedTask:
    objective:str; role:str="general"; id:str=field(default_factory=lambda:uuid.uuid4().hex); result:Any=None; error:str|None=None

class DelegationManager:
    def __init__(self,conversation:Conversation,max_concurrency:int=4): self.conversation=conversation; self.semaphore=asyncio.Semaphore(max(1,max_concurrency))
    async def run(self,tasks:list[DelegatedTask],executor:Callable[[DelegatedTask],Awaitable[Any]]):
        async def one(task):
            async with self.semaphore:
                self.conversation.emit(EventKind.DELEGATION_STARTED,{"task_id":task.id,"role":task.role,"objective":task.objective})
                try: task.result=await executor(task)
                except Exception as exc: task.error=f"{type(exc).__name__}: {exc}"
                self.conversation.emit(EventKind.DELEGATION_FINISHED,{"task_id":task.id,"role":task.role,"result":task.result,"error":task.error})
                return task
        return await asyncio.gather(*(one(t) for t in tasks))
