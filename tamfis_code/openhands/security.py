from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SecurityDecision:
    allowed: bool
    risk: str
    reason: str
    requires_approval: bool = False


class SecurityAnalyzer:
    DESTRUCTIVE=(r"\brm\s+-rf\b",r"\bmkfs\b",r"\bdd\s+if=",r"\bshutdown\b",r"\breboot\b",r"git\s+reset\s+--hard",r"git\s+clean\s+-[a-z]*f")
    NETWORK=(r"\bcurl\b",r"\bwget\b",r"\bssh\b",r"\bscp\b")
    def analyze(self, tool_name:str, arguments:dict[str,Any], *, approval_policy:str="ask")->SecurityDecision:
        text=json.dumps(arguments,default=str).lower()
        if any(re.search(p,text,re.I) for p in self.DESTRUCTIVE): return SecurityDecision(approval_policy in {"auto","full-auto"},"critical","destructive operation",True)
        if tool_name in {"restore_snapshot","git_commit","git_branch","write_file","edit_file"}: return SecurityDecision(approval_policy not in {"never","read-only","plan-only"},"medium","workspace mutation",approval_policy in {"ask","safe"})
        if any(re.search(p,text,re.I) for p in self.NETWORK): return SecurityDecision(approval_policy!="never","medium","network operation",approval_policy=="ask")
        return SecurityDecision(True,"low","read-only or low-risk action",False)


class SecretVault:
    """File-backed secret names with redacted access; values never enter events."""
    def __init__(self,path:str|Path): self.path=Path(path).expanduser(); self.path.parent.mkdir(parents=True,exist_ok=True)
    def _load(self):
        if not self.path.exists(): return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
    def set(self,name:str,value:str):
        data=self._load(); data[name]=value; self.path.write_text(json.dumps(data),encoding="utf-8"); os.chmod(self.path,0o600)
    def names(self): return sorted(self._load())
    def environment(self,names:list[str])->dict[str,str]:
        data=self._load(); missing=[n for n in names if n not in data]
        if missing: raise KeyError(f"missing secrets: {', '.join(missing)}")
        return {n:data[n] for n in names}
