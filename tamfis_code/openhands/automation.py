from __future__ import annotations
import asyncio, json, time, uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

@dataclass(slots=True)
class Automation:
    name:str; objective:str; workspace:str; interval_seconds:float; id:str=field(default_factory=lambda:uuid.uuid4().hex); enabled:bool=True; last_run:float|None=None; next_run:float|None=None

class AutomationStore:
    def __init__(self,path:str|Path):self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def load(self):
        if not self.path.exists():return []
        return [Automation(**x) for x in json.loads(self.path.read_text())]
    def save(self,items):self.path.write_text(json.dumps([asdict(x) for x in items],indent=2))

class AutomationScheduler:
    def __init__(self,store:AutomationStore,runner:Callable[[Automation],Awaitable[None]]):self.store=store;self.runner=runner;self._stop=False
    async def run_forever(self,poll:float=1.0):
        while not self._stop:
            now=time.time();items=self.store.load();changed=False
            for item in items:
                due=item.enabled and (item.next_run is None or item.next_run<=now)
                if due:
                    await self.runner(item);item.last_run=now;item.next_run=now+max(60,item.interval_seconds);changed=True
            if changed:self.store.save(items)
            await asyncio.sleep(poll)
    def stop(self):self._stop=True
