import json
from pathlib import Path

import pytest

from tamfis_code.mcp import MCPServer


MANIFEST = Path(__file__).parent / "fixtures" / "artifact_benchmark_manifest.json"


@pytest.mark.asyncio
async def test_synthetic_artifact_corpus_creates_and_inspects_all_formats(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    server = MCPServer(workspace_root=str(tmp_path), session_id=901)
    for scenario in manifest["scenarios"]:
        relative = f"deliverables/{scenario['filename']}"
        created = await server.call_tool("create_artifact", {
            "path": relative, "format": scenario["format"], "content": scenario["content"],
        })
        assert created["success"] is True, scenario["id"]
        inspected = await server.call_tool("inspect_artifact", {"path": relative})
        assert inspected["result"]["success"] is True, scenario["id"]
        assert scenario["marker"] in inspected["result"]["text"], scenario["id"]


def test_benchmark_manifest_is_provider_and_domain_neutral():
    serialized = MANIFEST.read_text(encoding="utf-8").lower()
    assert all(term not in serialized for term in ("openai", "anthropic", "nvidia", "housekeeping", "sap", "pacer"))
