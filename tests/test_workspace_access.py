from __future__ import annotations

import pytest

from tamfis_code import workspace_access


@pytest.mark.asyncio
async def test_accessible_workspace_skips_privileged_helper(monkeypatch, tmp_path):
    called = False

    async def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("helper must not run")

    monkeypatch.setattr(workspace_access.asyncio, "create_subprocess_exec", forbidden)
    ok, error = await workspace_access.ensure_workspace_access(str(tmp_path), read_only=False)
    assert ok is True
    assert error == ""
    assert called is False


@pytest.mark.asyncio
async def test_external_denied_workspace_requires_host_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace_access.os, "access", lambda *_args: False)
    monkeypatch.setattr(workspace_access, "MANAGED_ROOT", tmp_path / "managed")
    ok, error = await workspace_access.ensure_workspace_access(
        str(tmp_path / "external"), read_only=False,
    )
    assert ok is False
    assert "requires host approval outside" in error
