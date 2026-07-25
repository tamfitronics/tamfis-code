import asyncio
from pathlib import Path

import pytest

from tamfis_code.live_input import LiveInputListener
from tamfis_code.mcp import MCPServer


class _Renderer:
    def __init__(self):
        self.live_input_listener = None
        self.resumed = False

    def suspend_live(self):
        pass

    def resume_live(self):
        self.resumed = True

    def handle_event(self, event):
        pass

    def live_input_status(self):
        return "running"


class _Config:
    approval_policy = "ask"


@pytest.mark.asyncio
async def test_stop_awaits_prompt_task_and_releases_owner():
    renderer = _Renderer()
    listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_Config())
    renderer.live_input_listener = listener
    listener._active = True
    blocker = asyncio.Event()

    async def waiting_prompt():
        try:
            await blocker.wait()
        finally:
            listener._prompt_session = None

    listener._input_task = asyncio.create_task(waiting_prompt())
    await asyncio.sleep(0)
    await listener.stop()

    assert listener._input_task is None
    assert listener._prompt_session is None
    assert renderer.live_input_listener is None
    assert renderer.resumed


@pytest.mark.asyncio
async def test_edit_file_content_alias_performs_full_atomic_replacement(tmp_path: Path):
    target = tmp_path / "functions.php"
    target.write_text("<?php\necho 'old';\n", encoding="utf-8")
    server = MCPServer(workspace_root=str(tmp_path))

    content = "<?php\n$wpdb->get_var(\"SELECT COUNT(*)\");\n"
    result = await server._edit_file("functions.php", content=content)

    assert result.startswith("✅")
    assert target.read_text(encoding="utf-8") == content
    assert not list(tmp_path.glob(".functions.php.*.tmp"))


@pytest.mark.asyncio
async def test_edit_file_accepts_old_text_and_new_text_aliases(tmp_path: Path):
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    server = MCPServer(workspace_root=str(tmp_path))

    result = await server._edit_file(
        "sample.py", old_text="value = 1", new_text="value = 2"
    )

    assert result.startswith("✅")
    assert target.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_atomic_write_replaces_existing_file_without_temp_leak(tmp_path: Path):
    target = tmp_path / "data.txt"
    target.write_text("before", encoding="utf-8")
    server = MCPServer(workspace_root=str(tmp_path))

    result = await server._write_file("data.txt", text="after")

    assert result.startswith("✅")
    assert target.read_text(encoding="utf-8") == "after"
    assert not list(tmp_path.glob(".data.txt.*.tmp"))
