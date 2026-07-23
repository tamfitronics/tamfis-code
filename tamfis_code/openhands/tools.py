from __future__ import annotations

import asyncio
import html
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin

import httpx

from .workspace import LocalWorkspace


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "output": self.output, "error": self.error, "metadata": self.metadata}


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[..., Any]
    mutating: bool = False
    dangerous: bool = False

    async def invoke(self, **arguments: Any) -> ToolResult:
        started = time.monotonic()
        try:
            value = self.handler(**arguments)
            if asyncio.iscoroutine(value): value = await value
            result = value if isinstance(value, ToolResult) else ToolResult(True, value)
        except Exception as exc:
            result = ToolResult(False, error=f"{type(exc).__name__}: {exc}")
        result.metadata.setdefault("duration_seconds", time.monotonic() - started)
        return result


class ToolRegistry:
    def __init__(self): self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools: raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool
    def get(self, name: str) -> Tool:
        if name not in self._tools: raise KeyError(f"unknown tool: {name}")
        return self._tools[name]
    def list(self) -> list[Tool]: return list(self._tools.values())
    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult: return await self.get(name).invoke(**arguments)


class _TextExtractor(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.links=[]; self._href=None
    def handle_starttag(self, tag, attrs):
        if tag == "a": self._href = dict(attrs).get("href")
    def handle_endtag(self, tag):
        if tag == "a": self._href=None
    def handle_data(self, data):
        text=" ".join(data.split())
        if text: self.parts.append(text)
        if text and self._href: self.links.append({"text": text, "href": self._href})


class BrowserSession:
    """Stateful HTTP browser with text extraction, links, downloads and optional Playwright actions."""
    def __init__(self, *, timeout: float = 30.0, user_agent: str = "Tamfis-Code/0.6"):
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": user_agent})
        self.current_url: str | None = None; self.history: list[str] = []
    async def close(self): await self.client.aclose()
    async def open(self, url: str) -> dict[str, Any]:
        response=await self.client.get(url); response.raise_for_status(); self.current_url=str(response.url); self.history.append(self.current_url)
        content_type=response.headers.get("content-type", "")
        if "html" in content_type:
            parser=_TextExtractor(); parser.feed(response.text)
            links=[{"text": item["text"], "href": urljoin(self.current_url, item["href"])} for item in parser.links[:500]]
            return {"url": self.current_url, "status": response.status_code, "title": self._title(response.text), "text": "\n".join(parser.parts)[:200000], "links": links, "content_type": content_type}
        return {"url": self.current_url, "status": response.status_code, "content_type": content_type, "bytes": len(response.content), "text": response.text[:200000]}
    @staticmethod
    def _title(document: str) -> str:
        match=re.search(r"<title[^>]*>(.*?)</title>", document, re.I|re.S)
        return html.unescape(" ".join(match.group(1).split())) if match else ""
    async def download(self, url: str, path: str) -> dict[str, Any]:
        response=await self.client.get(url); response.raise_for_status(); target=Path(path).expanduser().resolve(); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(response.content)
        return {"path": str(target), "bytes": len(response.content), "url": str(response.url)}


class GitTools:
    def __init__(self, workspace: LocalWorkspace): self.workspace=workspace
    async def _run(self, *args: str) -> dict[str, Any]:
        cmd="git " + " ".join(shlex.quote(a) for a in args); return (await self.workspace.execute(cmd)).as_dict()
    async def status(self): return await self._run("status", "--short", "--branch")
    async def diff(self, staged: bool=False): return await self._run("diff", "--cached" if staged else "--")
    async def log(self, limit: int=20): return await self._run("log", f"-{max(1,min(limit,100))}", "--oneline", "--decorate")
    async def branch(self, name: str, checkout: bool=True): return await self._run("checkout" if checkout else "branch", "-b" if checkout else name, name) if checkout else await self._run("branch", name)
    async def commit(self, message: str): return await self._run("commit", "-am", message)


def default_registry(workspace: LocalWorkspace) -> ToolRegistry:
    registry=ToolRegistry(); browser=BrowserSession(); git=GitTools(workspace)
    registry.register(Tool("terminal", "Execute a shell command inside the workspace", {"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"},"timeout":{"type":"number"}},"required":["command"]}, workspace.execute, dangerous=True))
    registry.register(Tool("read_file", "Read a UTF-8 text file", {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}, workspace.read_text))
    registry.register(Tool("write_file", "Create or replace a UTF-8 text file", {"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}, workspace.write_text, mutating=True))
    registry.register(Tool("edit_file", "Replace one exact occurrence", {"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}, workspace.replace_text, mutating=True))
    registry.register(Tool("list_files", "List files and directories", {"type":"object","properties":{"path":{"type":"string"}}}, workspace.list_files))
    registry.register(Tool("browser_open", "Open a web page and extract text and links", {"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}, browser.open))
    registry.register(Tool("browser_download", "Download a URL to a file", {"type":"object","properties":{"url":{"type":"string"},"path":{"type":"string"}},"required":["url","path"]}, browser.download, mutating=True))
    registry.register(Tool("git_status", "Show Git status", {"type":"object","properties":{}}, git.status))
    registry.register(Tool("git_diff", "Show Git diff", {"type":"object","properties":{"staged":{"type":"boolean"}}}, git.diff))
    registry.register(Tool("git_log", "Show recent Git commits", {"type":"object","properties":{"limit":{"type":"integer"}}}, git.log))
    registry.register(Tool("git_branch", "Create a Git branch", {"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}, git.branch, mutating=True))
    registry.register(Tool("git_commit", "Commit tracked changes", {"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}, git.commit, mutating=True))
    registry.register(Tool("snapshot", "Create a reversible workspace snapshot", {"type":"object","properties":{"label":{"type":"string"}}}, workspace.snapshot, mutating=False))
    registry.register(Tool("restore_snapshot", "Restore a workspace snapshot", {"type":"object","properties":{"snapshot_id":{"type":"string"}},"required":["snapshot_id"]}, workspace.restore, mutating=True, dangerous=True))
    return registry
