from pathlib import Path

import pytest

from tamfis_code.mcp import MCPServer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "content", "detail_key"),
    [
        ("docx", {"title": "Enterprise Theme", "sections": [{"heading": "Scope", "content": "Complete build"}]}, "paragraphs"),
        ("xlsx", {"sheets": [{"name": "Budget", "rows": [["Item", "Cost"], ["Theme", 5000]]}]}, "sheets"),
        ("pptx", {"title": "Launch", "slides": [{"title": "Plan", "bullets": ["Build", "Test", "Ship"]}]}, "slides"),
        ("pdf", {"title": "Project Report", "sections": [{"heading": "Result", "content": "Verified delivery"}]}, "pages"),
    ],
)
async def test_create_and_inspect_real_office_artifacts(tmp_path: Path, kind: str, content: dict, detail_key: str):
    server = MCPServer(workspace_root=str(tmp_path), session_id=500)
    created = await server.call_tool("create_artifact", {
        "path": f"deliverables/output.{kind}", "format": kind, "content": content,
    })
    assert created["success"] is True
    assert created["result"]["success"] is True
    output = tmp_path / "deliverables" / f"output.{kind}"
    assert output.is_file() and output.stat().st_size > 0

    inspected = await server.call_tool("inspect_artifact", {"path": str(output)})
    assert inspected["result"]["success"] is True
    assert detail_key in inspected["result"]
    assert inspected["result"]["text"]


@pytest.mark.asyncio
async def test_spreadsheet_formula_injection_is_escaped_by_default(tmp_path: Path):
    server = MCPServer(workspace_root=str(tmp_path))
    await server.call_tool("create_artifact", {
        "path": "safe.xlsx", "format": "xlsx",
        "content": {"rows": [["Input"], ["=HYPERLINK(\"https://evil.example\")"]]},
    })
    inspected = await server.call_tool("inspect_artifact", {"path": "safe.xlsx"})
    assert "'=HYPERLINK" in inspected["result"]["text"]


@pytest.mark.asyncio
async def test_artifact_extension_must_match_format(tmp_path: Path):
    server = MCPServer(workspace_root=str(tmp_path))
    result = await server.call_tool("create_artifact", {
        "path": "fake.pdf", "format": "docx", "content": {},
    })
    assert result["success"] is False
    assert "must end in .docx" in result["error"]
