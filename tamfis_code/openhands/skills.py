from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    instructions: str
    required_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source: str = ""
    enabled: bool = True


class SkillRegistry:
    def __init__(self, roots: list[str | Path] | None = None):
        self.roots=[Path(p).expanduser().resolve() for p in (roots or [])]; self._skills: dict[str, Skill]={}
    def load(self) -> dict[str, Skill]:
        self._skills={}
        for root in self.roots:
            if not root.exists(): continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name.lower() in {"skill.md","skill.toml","skill.json"}:
                    skill=self._parse(path); self._skills[skill.name]=skill
        return dict(self._skills)
    def _parse(self, path: Path) -> Skill:
        if path.suffix==".toml": data=tomllib.loads(path.read_text(encoding="utf-8")); instructions=str(data.get("instructions", ""))
        elif path.suffix==".json": data=json.loads(path.read_text(encoding="utf-8")); instructions=str(data.get("instructions", ""))
        else:
            text=path.read_text(encoding="utf-8"); data={}; instructions=text
            match=re.match(r"^---\n(.*?)\n---\n", text, re.S)
            if match:
                for line in match.group(1).splitlines():
                    if ":" in line:
                        key,value=line.split(":",1); data[key.strip()]=value.strip()
                instructions=text[match.end():]
        name=str(data.get("name") or path.parent.name); description=str(data.get("description") or instructions.splitlines()[0] if instructions.splitlines() else name)
        tools=data.get("required_tools", ()); tags=data.get("tags", ())
        if isinstance(tools,str): tools=tuple(x.strip() for x in tools.strip("[]").split(",") if x.strip())
        if isinstance(tags,str): tags=tuple(x.strip() for x in tags.strip("[]").split(",") if x.strip())
        return Skill(name, description, instructions.strip(), tuple(tools), tuple(tags), str(path), bool(data.get("enabled", True)))
    def get(self,name:str)->Skill: return self._skills[name]
    def list(self)->list[Skill]: return sorted((s for s in self._skills.values() if s.enabled), key=lambda s:s.name)
    def match(self, objective:str, limit:int=5)->list[Skill]:
        words=set(re.findall(r"[a-z0-9_-]+", objective.lower())); scored=[]
        for skill in self.list():
            hay=set(re.findall(r"[a-z0-9_-]+", f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower())); scored.append((len(words&hay),skill))
        return [skill for score,skill in sorted(scored,key=lambda x:(-x[0],x[1].name)) if score>0][:limit]
