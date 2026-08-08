import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import remote_agent


class RemoteAgentIdentityTests(unittest.TestCase):
    def test_device_and_workspace_identity_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(remote_agent, "CONFIG_DIR", root), patch.object(
                remote_agent, "DEVICE_PATH", root / "device.json"
            ):
                first = remote_agent.load_or_create_device_identity()
                second = remote_agent.load_or_create_device_identity()

            self.assertEqual(first, second)
            self.assertEqual(len(first["device_id"]), 32)
            self.assertEqual(
                remote_agent.workspace_identity(first["device_id"], str(root)),
                remote_agent.workspace_identity(first["device_id"], str(root / ".")),
            )
            self.assertNotEqual(
                remote_agent.workspace_identity(first["device_id"], str(root)),
                remote_agent.workspace_identity(first["device_id"], str(root / "other")),
            )

    def test_websocket_url_preserves_reverse_proxy_path_without_token(self):
        url = remote_agent._websocket_url(
            "https://example.test/gateway", "device-1234567890",
        )
        self.assertEqual(
            url,
            "wss://example.test/gateway/api/v1/remote/agent/ws/device-1234567890",
        )
        self.assertNotIn("token", url)


class RemoteAgentExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_is_scoped_to_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = object.__new__(remote_agent.RemoteAgentBridge)
            bridge.workspace_root = tmp
            result = await bridge._execute({"command": "pwd"})
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(Path(result["stdout"].strip()), Path(tmp).resolve())

    async def test_write_rejects_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = object.__new__(remote_agent.RemoteAgentBridge)
            bridge.workspace_root = tmp
            result = await bridge._write_text_file({
                "path": "../outside.txt", "content": "not allowed",
            })
            self.assertIn("error", result)
            self.assertFalse((Path(tmp).parent / "outside.txt").exists())
