from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import CONFIG_DIR


_MATCH_STOPWORDS = {
    "and", "are", "but", "caused", "for", "from", "into", "please", "that",
    "the", "this", "through", "use", "when", "with", "your", "as", "at", "be",
    "by", "do", "if", "in", "is", "it", "of", "on", "or", "to", "we",
}


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    instructions: str
    required_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source: str = ""
    enabled: bool = True
    model_invocable: bool = True


class SkillRegistry:
    def __init__(self, roots: list[str | Path] | None = None):
        self.roots=[Path(p).expanduser().resolve() for p in (roots or [])]; self._skills: dict[str, Skill]={}
    def load(self) -> dict[str, Skill]:
        self._skills={}
        for root in self.roots:
            if not root.exists(): continue
            for path in sorted(root.rglob("*")):
                # Kimi also supports a flat `<name>.md` directly inside a
                # skill root. Do not treat nested reference Markdown files as
                # independent skills.
                supported = (
                    path.name.lower() in {"skill.md", "skill.toml", "skill.json"}
                    or (path.parent == root and path.suffix.lower() == ".md")
                )
                if path.is_file() and supported:
                    try:
                        skill=self._parse(path)
                    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError):
                        continue
                    self._skills[skill.name]=skill
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
                        key,value=line.split(":",1)
                        value = value.strip()
                        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                            value = value[1:-1]
                        data[key.strip()]=value
                instructions=text[match.end():]
        fallback_name = path.stem if path.name.lower() != "skill.md" else path.parent.name
        name=str(data.get("name") or fallback_name); description=str(data.get("description") or instructions.splitlines()[0] if instructions.splitlines() else name)
        tools=data.get("required_tools", ()); tags=data.get("tags", ())
        if isinstance(tools,str): tools=tuple(x.strip() for x in tools.strip("[]").split(",") if x.strip())
        if isinstance(tags,str): tags=tuple(x.strip() for x in tags.strip("[]").split(",") if x.strip())
        enabled_value = data.get("enabled", True)
        enabled = enabled_value if isinstance(enabled_value, bool) else str(enabled_value).lower() not in {"false", "0", "no"}
        disabled_value = data.get("disableModelInvocation", data.get("disable_model_invocation", False))
        disabled_invocation = disabled_value is True or str(disabled_value).lower() in {"true", "1", "yes"}
        skill_type = str(data.get("type") or "prompt").lower()
        return Skill(
            name, description, instructions.strip(), tuple(tools), tuple(tags), str(path), enabled,
            not disabled_invocation and skill_type != "flow",
        )
    def get(self,name:str)->Skill: return self._skills[name]
    def list(self)->list[Skill]: return sorted((s for s in self._skills.values() if s.enabled), key=lambda s:s.name)
    def match(self, objective:str, limit:int=5)->list[Skill]:
        objective_lower = objective.lower()
        words = {
            word for word in re.findall(r"[a-z0-9_-]+", objective_lower)
            if len(word) > 1 and word not in _MATCH_STOPWORDS
        }
        scored=[]
        for skill in self.list():
            if not skill.model_invocable:
                continue
            name = skill.name.lower().strip('"\'')
            name_words = set(re.findall(r"[a-z0-9]+", name))
            hay=set(re.findall(r"[a-z0-9_-]+", f"{name} {skill.description} {' '.join(skill.tags)}".lower()))
            overlap = words & hay
            explicit = name in objective_lower or bool(words & name_words)
            score = len(overlap)
            if explicit or score >= 2:
                scored.append((score + (10 if explicit else 0), skill))
        return [skill for score,skill in sorted(scored,key=lambda x:(-x[0],x[1].name)) if score>0][:limit]


def workspace_skill_registry(workspace_root: str | Path) -> SkillRegistry:
    """Return layered Kimi/Codex/Claude/shared/Tamfis skills.

    Roots are ordered from least to most specific; a later same-named skill
    replaces an earlier one. This makes project Tamfis definitions the final
    authority without requiring users to copy their existing vendor skills.
    """
    root = Path(workspace_root).expanduser().resolve()
    home = Path.home()
    kimi_home = Path(os.environ.get("KIMI_CODE_HOME") or home / ".kimi-code").expanduser()
    from ..plugins import plugin_skill_roots
    return SkillRegistry([
        CONFIG_DIR / "skills",
        *plugin_skill_roots(),
        home / ".agents" / "skills",
        kimi_home / "skills",
        home / ".claude" / "skills",
        home / ".codex" / "skills",
        root / ".agents" / "skills",
        root / ".kimi-code" / "skills",
        root / ".claude" / "skills",
        root / ".codex" / "skills",
        root / ".tamfis" / "skills",
    ])


def skill_prompt(workspace_root: str | Path, objective: str, *, max_chars: int = 3_000) -> str:
    registry = workspace_skill_registry(workspace_root)
    registry.load()
    available = registry.list()
    if not available:
        return ""
    matched = registry.match(objective)
    lines = ["Available skills (use when the request names one or clearly matches its description):"]
    # User-level vendor stores may contain dozens of skills. Keep their
    # discovery catalog bounded so the immutable system prompt cannot crowd
    # the actual request out of context. Matched instructions receive the
    # remaining budget below.
    catalog_budget = max(400, min(max_chars // 2, 1_500))
    for skill in available:
        description = skill.description.strip().replace("\n", " ")
        if len(description) > 180:
            description = description[:177] + "..."
        entry = f"- {skill.name}: {description}"
        if len("\n".join([*lines, entry])) > catalog_budget:
            remaining = len(available) - (len(lines) - 1)
            lines.append(f"- ... {remaining} more installed skill(s)")
            break
        lines.append(entry)
    if matched:
        lines.append("\nMatched skill instructions for this objective:")
        remaining = max_chars - len("\n".join(lines))
        for skill in matched:
            marker = f"\n--- skill: {skill.name} ({skill.source}) ---\n"
            if remaining <= len(marker):
                break
            body = skill.instructions[:remaining - len(marker)]
            lines.append(marker + body)
            remaining -= len(marker) + len(body)
    return "\n".join(lines)[:max_chars]
