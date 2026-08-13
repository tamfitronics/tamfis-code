from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from tamfis_code.mcp import MCPServer
from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import (
    _has_port_conflict,
    _looks_like_forced_listener_reclamation,
    _user_authorized_service_disruption,
    run_local_agent_turn,
)

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _RecordingRenderer,
    _StatePatchMixin,
    _chunk,
    _delta,
    _tool_call_delta,
)


class PortConflictRecoveryTests(unittest.TestCase):
    def test_recognizes_nested_node_eaddrinuse_output(self):
        result = {
            "success": False,
            "result": {
                "stderr": "Error: listen EADDRINUSE: address already in use 0.0.0.0:8090",
                "return_code": 1,
            },
        }
        self.assertTrue(_has_port_conflict(result))

    def test_recognizes_forced_port_reclamation_commands(self):
        commands = (
            "kill -9 3534595",
            "pkill -f 'node dist/boot.js'",
            "fuser -k 8090/tcp",
            "lsof -t -i:8090 | xargs kill -9",
            "systemctl restart tamfisseo.service",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(_looks_like_forced_listener_reclamation(command))

    def test_read_only_listener_diagnostics_are_allowed(self):
        commands = (
            "ss -ltnp 'sport = :8090'",
            "lsof -i :8090",
            "systemctl status tamfisseo.service",
            "curl -fsS http://127.0.0.1:8090/api/health",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(_looks_like_forced_listener_reclamation(command))

    def test_service_disruption_requires_explicit_user_language(self):
        self.assertFalse(_user_authorized_service_disruption("fix the startup validation failure"))
        self.assertTrue(_user_authorized_service_disruption("restart the tamfisseo service"))
        self.assertTrue(_user_authorized_service_disruption("kill the stale server process"))


class PortConflictRunnerTests(_StatePatchMixin, unittest.TestCase):
    def test_blocks_blind_kill_after_bind_conflict(self):
        start_args = json.dumps({"command": "npm start"})
        kill_args = json.dumps({"command": "kill -9 3534595"})
        rounds = [
            [_chunk(_delta(tool_calls=[
                _tool_call_delta(0, call_id="start", name="execute_command", arguments=start_args),
            ]))],
            [_chunk(_delta(tool_calls=[
                _tool_call_delta(0, call_id="kill", name="execute_command", arguments=kill_args),
            ]))],
            [_chunk(_delta(content="The existing listener must be health-checked and left in place."))],
        ]
        client = _FakeClient(rounds)
        manager = _FakeManager(client)
        renderer = _RecordingRenderer()
        dispatched = []

        async def fake_call_tool(_server, name, parameters, **_kwargs):
            dispatched.append((name, parameters))
            return {
                "success": False,
                "result": {
                    "stderr": "Error: listen EADDRINUSE: address already in use 0.0.0.0:8090",
                    "return_code": 1,
                },
            }

        with tempfile.TemporaryDirectory() as workspace, patch(
            "tamfis_code.runner_local.should_plan", return_value=False,
        ), patch(
            "tamfis_code.runner_local.detect_validation_commands", return_value=[],
        ), patch.object(MCPServer, "call_tool", new=fake_call_tool):
            outcome = asyncio.run(run_local_agent_turn(
                manager,
                ProviderType.NVIDIA,
                None,
                [{"role": "user", "content": "fix the startup validation failure"}],
                Console(file=StringIO(), no_color=True, width=200),
                renderer,
                workspace_root=workspace,
                session_id=1,
                approval_policy="auto",
                interactive=False,
            ))

        # The synthetic task may still fail the mutation/completion contract
        # because it intentionally edits no file; this regression is about
        # preventing the destructive command from ever reaching the shell.
        self.assertIn(outcome.status, {"completed", "failed"})
        self.assertEqual([parameters["command"] for _, parameters in dispatched], ["npm start"])
        all_messages = [message for call in client.calls for message in call.get("messages", [])]
        self.assertTrue(any(
            message.get("role") == "system" and "health/readiness endpoint" in str(message.get("content"))
            for message in all_messages
        ))
        self.assertTrue(any(
            message.get("role") == "tool" and "Refused blind listener reclamation" in str(message.get("content"))
            for message in all_messages
        ))

    def test_blocks_blind_kill_after_bind_conflict_in_the_same_round(self):
        # Regression: round-loop concurrency defers a call's actual
        # dispatch (and the port_conflict_seen flag it sets) until
        # _flush_dispatch_queue runs -- if BOTH the conflicting start and
        # the forced-reclamation kill are requested in the SAME model turn,
        # the kill's guard check must still see the conflict the start
        # produced moments earlier in this same round, not a stale
        # port_conflict_seen read before the start ever actually executed.
        start_args = json.dumps({"command": "npm start"})
        kill_args = json.dumps({"command": "kill -9 3534595"})
        rounds = [
            [_chunk(_delta(tool_calls=[
                _tool_call_delta(0, call_id="start", name="execute_command", arguments=start_args),
                _tool_call_delta(1, call_id="kill", name="execute_command", arguments=kill_args),
            ]))],
            [_chunk(_delta(content="The existing listener must be health-checked and left in place."))],
        ]
        client = _FakeClient(rounds)
        manager = _FakeManager(client)
        renderer = _RecordingRenderer()
        dispatched = []

        async def fake_call_tool(_server, name, parameters, **_kwargs):
            dispatched.append((name, parameters))
            return {
                "success": False,
                "result": {
                    "stderr": "Error: listen EADDRINUSE: address already in use 0.0.0.0:8090",
                    "return_code": 1,
                },
            }

        with tempfile.TemporaryDirectory() as workspace, patch(
            "tamfis_code.runner_local.should_plan", return_value=False,
        ), patch(
            "tamfis_code.runner_local.detect_validation_commands", return_value=[],
        ), patch.object(MCPServer, "call_tool", new=fake_call_tool):
            outcome = asyncio.run(run_local_agent_turn(
                manager,
                ProviderType.NVIDIA,
                None,
                [{"role": "user", "content": "fix the startup validation failure"}],
                Console(file=StringIO(), no_color=True, width=200),
                renderer,
                workspace_root=workspace,
                session_id=1,
                approval_policy="auto",
                interactive=False,
            ))

        self.assertIn(outcome.status, {"completed", "failed"})
        # The kill must never actually reach the shell -- only "npm start"
        # (the call that produced the conflict) should have dispatched.
        self.assertEqual([parameters["command"] for _, parameters in dispatched], ["npm start"])
        all_messages = [message for call in client.calls for message in call.get("messages", [])]
        self.assertTrue(any(
            message.get("role") == "tool" and "Refused blind listener reclamation" in str(message.get("content"))
            for message in all_messages
        ))


if __name__ == "__main__":
    unittest.main()
