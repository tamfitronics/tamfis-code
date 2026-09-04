import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

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


class RemoteAgentReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_waits_for_transient_registration_failure(self):
        """Startup stays alive until a recoverable remote connection succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempts = 0

            class Client:
                config = SimpleNamespace(api_base="https://example.test")
                credentials = SimpleNamespace(access_token="token")

                async def register_agent_device(self, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise httpx.ConnectError("offline")
                    return {"id": 9}

                async def me(self):
                    return {}

            class Socket:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return None

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    await asyncio.Event().wait()

                async def send(self, _message):
                    return None

            with patch.object(remote_agent, "CONFIG_DIR", root), patch.object(
                remote_agent, "DEVICE_PATH", root / "device.json"
            ), patch.object(remote_agent, "OUTBOX_PATH", root / "outbox.json"), patch(
                "tamfis_code.remote_agent.websockets.connect", return_value=Socket()
            ), patch.object(remote_agent, "RECONNECT_DELAY_SECONDS", 0.001):
                bridge = remote_agent.RemoteAgentBridge(Client(), str(root))
                server = await asyncio.wait_for(bridge.start(), timeout=1)
                self.assertEqual(server, {"id": 9})
                self.assertEqual(attempts, 2)
                self.assertTrue(bridge._connected.is_set())
                await bridge.stop()

    async def test_cancelled_start_stops_unregistered_bridge_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class Client:
                config = SimpleNamespace(api_base="https://example.test")
                credentials = SimpleNamespace(access_token="token")

                async def register_agent_device(self, **_kwargs):
                    await asyncio.Event().wait()

            with patch.object(remote_agent, "CONFIG_DIR", root), patch.object(
                remote_agent, "DEVICE_PATH", root / "device.json"
            ), patch.object(remote_agent, "OUTBOX_PATH", root / "outbox.json"):
                bridge = remote_agent.RemoteAgentBridge(Client(), str(root))
                startup = asyncio.create_task(bridge.start())
                await asyncio.sleep(0)
                startup.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await startup
                self.assertTrue(bridge._task.done())

    def test_outbox_write_is_valid_after_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = object.__new__(remote_agent.RemoteAgentBridge)
            bridge._outbox = {"request-1": {"type": "rpc_result", "id": "request-1"}}
            with patch.object(remote_agent, "CONFIG_DIR", root), patch.object(
                remote_agent, "OUTBOX_PATH", root / "agent-outbox.json"
            ):
                bridge._save_outbox()
                self.assertEqual(bridge._load_outbox(), bridge._outbox)
                self.assertEqual((root / "agent-outbox.json").stat().st_mode & 0o777, 0o600)
