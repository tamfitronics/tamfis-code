from __future__ import annotations
import json, os, time, uuid
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(slots=True)
class Lease:
    resource: str; owner: str; token: str; expires_at: float
    @property
    def expired(self): return time.time() >= self.expires_at

class LeaseManager:
    def __init__(self,path:str|Path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def _load(self):
        if not self.path.exists(): return {}
        try: return json.loads(self.path.read_text())
        except Exception: return {}
    def _save(self,data):
        temp=self.path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2)); os.replace(temp,self.path)
    def acquire(self,resource:str,owner:str,ttl:float=300)->Lease:
        data=self._load(); current=data.get(resource)
        if current and current['expires_at']>time.time() and current['owner']!=owner: raise RuntimeError(f"resource leased by {current['owner']}")
        lease=Lease(resource,owner,uuid.uuid4().hex,time.time()+ttl); data[resource]=asdict(lease); self._save(data); return lease
    def renew(self,lease:Lease,ttl:float=300)->Lease:
        data=self._load(); current=data.get(lease.resource)
        if not current or current['token']!=lease.token: raise RuntimeError('lease lost')
        renewed=Lease(lease.resource,lease.owner,lease.token,time.time()+ttl); data[lease.resource]=asdict(renewed); self._save(data); return renewed
    def release(self,lease:Lease):
        data=self._load(); current=data.get(lease.resource)
        if current and current['token']==lease.token: data.pop(lease.resource,None); self._save(data)
