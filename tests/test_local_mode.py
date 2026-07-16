"""Tests for tamfis-code's offline/local chat mode (providers.py wired up
for real, local_tools.py's read-only boundary, local_chat.py's tool loop,
and the `tamfis-code local` CLI command)."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from tamfis_code.cli import cli
from tamfis_code.local_chat import resolve_provider_type, run_local_turn
from tamfis_code.local_tools import READ_ONLY_TOOL_SCHEMAS, LocalReadOnlyTools
from tamfis_code.providers import ProviderManager, ProviderType


class ResolveProviderTypeTests(unittest.TestCase):
    def test_known_aliases_resolve(self):
        self.assertEqual(resolve_provider_type("ollama"), ProviderType.OLLAMA)
        self.assertEqual(resolve_provider_type("nvidia"), ProviderType.NVIDIA)
        self.assertEqual(resolve_provider_type("hf"), ProviderType.HF)
        self.assertEqual(resolve_provider_type("or"), ProviderType.OPENROUTER)
        self.assertEqual(resolve_provider_type(None), ProviderType.AUTO)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            resolve_provider_type("not-a-real-provider")


class LocalReadOnlyToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_only_tool_is_dispatched(self):
        tools = LocalReadOnlyTools()
        result = await tools.call("get_git_info", {"path": "."})
        self.assertIn("is_git_repo", result)

    async def test_write_file_is_refused(self):
        tools = LocalReadOnlyTools()
        with self.assertRaises(ValueError):
            await tools.call("write_file", {"path": "x.txt", "content": "y"})

    async def test_execute_command_is_refused(self):
        tools = LocalReadOnlyTools()
        with self.assertRaises(ValueError):
            await tools.call("execute_command", {"command": "echo hi"})

    async def test_browser_is_refused(self):
        tools = LocalReadOnlyTools()
        with self.assertRaises(ValueError):
            await tools.call("browser", {"url": "https://example.com", "action": "navigate"})

    def test_schemas_only_advertise_read_only_names(self):
        names = {schema["function"]["name"] for schema in READ_ONLY_TOOL_SCHEMAS}
        self.assertEqual(names, {"read_file", "list_directory", "search_code", "get_git_info"})
        self.assertNotIn("write_file", names)
        self.assertNotIn("execute_command", names)


def _fake_openai_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _fake_tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class RunLocalTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_answer_with_no_tool_calls(self):
        manager = ProviderManager.__new__(ProviderManager)
        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(
            return_value=_fake_openai_response(content="hello there")
        )
        manager.get_client = MagicMock(return_value=fake_client)
        manager.PROVIDERS = ProviderManager.PROVIDERS

        console = MagicMock()
        answer = await run_local_turn(
            manager, ProviderType.OLLAMA, [{"role": "user", "content": "hi"}], None, console, use_tools=False,
        )
        self.assertEqual(answer, "hello there")

    async def test_tool_call_round_trip_then_final_answer(self):
        manager = ProviderManager.__new__(ProviderManager)
        fake_client = MagicMock()
        first = _fake_openai_response(
            content="", tool_calls=[_fake_tool_call("call_1", "get_git_info", "{\"path\": \".\"}")]
        )
        second = _fake_openai_response(content="This repo is not a git repo, or has no commits yet.")
        fake_client.chat.completions.create = AsyncMock(side_effect=[first, second])
        manager.get_client = MagicMock(return_value=fake_client)
        manager.PROVIDERS = ProviderManager.PROVIDERS

        console = MagicMock()
        with patch("tamfis_code.local_chat.LocalReadOnlyTools") as mock_tools_cls:
            mock_tools = MagicMock()
            mock_tools.call = AsyncMock(return_value={"is_git_repo": False})
            mock_tools_cls.return_value = mock_tools
            answer = await run_local_turn(
                manager, ProviderType.OLLAMA, [{"role": "user", "content": "is this a git repo?"}], None, console,
            )
        self.assertIn("not a git repo", answer)
        self.assertEqual(fake_client.chat.completions.create.await_count, 2)
        mock_tools.call.assert_awaited_once_with("get_git_info", {"path": "."})

    async def test_no_available_client_raises(self):
        manager = ProviderManager.__new__(ProviderManager)
        manager.get_client = MagicMock(return_value=None)
        manager.PROVIDERS = ProviderManager.PROVIDERS
        console = MagicMock()
        with self.assertRaises(RuntimeError):
            await run_local_turn(
                manager, ProviderType.OLLAMA, [{"role": "user", "content": "hi"}], None, console, use_tools=False,
            )


class LocalCommandCliTests(unittest.TestCase):
    def test_local_help_lists_no_login_requirement(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["local", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("no TamfisGPT account", result.output.replace("\n", " "))

    def test_local_without_objective_or_repl_is_a_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["local"])
        self.assertNotEqual(result.exit_code, 0)

    def test_local_rejects_unknown_provider(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["local", "--provider", "bogus", "hello"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown local provider", result.output)

    def test_local_single_turn_runs_without_backend_auth(self):
        runner = CliRunner()
        with patch("tamfis_code.local_chat.run_local_turn", new=AsyncMock(return_value="ok")):
            result = runner.invoke(cli, ["local", "hello there"])
        self.assertEqual(result.exit_code, 0, result.output)


if __name__ == "__main__":
    unittest.main()
