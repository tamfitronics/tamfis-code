"""Slash commands added from the Claude Code / Codex / Kimi Code / Freebuff comparison.

Owner request 2026-09-19: "check for /*** commands missing when compared to
freebuff, claude code, codex and kimi code; consolidate and add any missing".
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from tamfis_code import slash_registry as sr
from tamfis_code import state as state_module
from tamfis_code.runtime import ledger as ledger_module
from tamfis_code.workspace import WorkspaceContext


class _Isolated:
    def setUp(self):
        self._originals = (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
            ledger_module.LEDGER_DIR,
        )
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        state_module.CONFIG_DIR = self.base / ".config"
        state_module.STATE_PATH = self.base / ".config" / "state.json"
        state_module._LOCK_PATH = self.base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None
        ledger_module.LEDGER_DIR = self.base / "ledgers"
        state_module.save_session_state(1, workspace_root=str(self.root), primary_workspace=str(self.root))

    def tearDown(self):
        (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
            ledger_module.LEDGER_DIR,
        ) = self._originals
        state_module._STATE_CACHE = None
        self._tmp.cleanup()

    def ctx(self, **kw):
        self.buffer = StringIO()
        defaults = dict(
            console=Console(file=self.buffer, no_color=True, width=160),
            config=SimpleNamespace(approval_policy="ask"),
            workspace=WorkspaceContext(session_id=1, workspace_root=str(self.root)),
            conversation_history=[],
            version="9.9.9",
        )
        defaults.update(kw)
        return sr.SlashContext(**defaults)

    def run_cmd(self, text, ctx=None):
        ctx = ctx or self.ctx()
        return asyncio.run(sr.dispatch(text, ctx)), ctx

    @property
    def out(self):
        return self.buffer.getvalue()


class DispatchTests(_Isolated, unittest.TestCase):
    def test_text_that_is_not_a_slash_command_is_not_handled(self):
        for text in ("fix the bug", "", "  hello /new"):
            result, _ = self.run_cmd(text)
            self.assertIsNone(result, text)

    def test_an_unknown_slash_command_falls_through_to_the_repl(self):
        result, _ = self.run_cmd("/definitely-not-a-command")
        self.assertIsNone(result)

    def test_aliases_rewrite_to_the_existing_command_keeping_arguments(self):
        cases = {
            "/history": "/resume", "/sessions": "/resume", "/chats": "/resume", "/branch": "/fork",
            "/cost": "/usage", "/pwd": "/cwd", "/q": "/exit", "/?": "/help",
            "/rewind": "/undo", "/diagnostics": "/debug", "/release-notes": "/version", "/plugin": "/plugins",
        }
        for alias, target in cases.items():
            with self.subTest(alias=alias):
                result, _ = self.run_cmd(alias)
                self.assertEqual(result, sr.Rewrite(target))
        result, _ = self.run_cmd("/history 12")
        self.assertEqual(result, sr.Rewrite("/resume 12"))

    def test_a_users_own_custom_command_wins_over_a_builtin(self):
        result, _ = self.run_cmd("/review please", self.ctx(custom_commands={"review": object()}))
        self.assertIsNone(result)

    def test_a_crashing_handler_never_takes_the_repl_down(self):
        async def boom(ctx, arg):
            raise RuntimeError("kaput")

        broken = sr.SlashCommand("/boom", "x", boom)
        with patch.dict(sr._BY_NAME, {"/boom": broken}):
            result, _ = self.run_cmd("/boom")
        self.assertIs(result, sr.HANDLED)
        self.assertIn("/boom failed: RuntimeError: kaput", self.out)


class SessionCommandTests(_Isolated, unittest.TestCase):
    def test_new_starts_a_fresh_session_and_keeps_the_old_one(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        ctx = self.ctx(conversation_history=history, last_response_text="hello", last_turn=("x", "coding"))
        result, ctx = self.run_cmd("/new", ctx)
        self.assertIs(result, sr.HANDLED)
        self.assertGreater(ctx.workspace.session_id, 1)
        self.assertEqual(ctx.workspace.workspace_root, str(self.root))
        self.assertEqual(ctx.conversation_history, [])
        self.assertIsNone(ctx.last_response_text)
        self.assertIsNone(ctx.last_turn)
        self.assertIn("/resume 1", self.out)
        self.assertIn(1, state_module.all_known_session_ids())

    def test_reset_is_an_alias_of_new(self):
        result, ctx = self.run_cmd("/reset")
        self.assertIs(result, sr.HANDLED)
        self.assertGreater(ctx.workspace.session_id, 1)

    def test_export_writes_the_conversation_as_markdown(self):
        history = [{"role": "user", "content": "Add a login page"}, {"role": "assistant", "content": "Done -- added /login."}]
        result, _ = self.run_cmd("/export notes/session.md", self.ctx(conversation_history=history))
        text = (self.root / "notes" / "session.md").read_text()
        self.assertIn("## You", text)
        self.assertIn("Add a login page", text)
        self.assertIn("## Assistant", text)
        self.assertIn("Done -- added /login.", text)
        self.assertIn("Exported", self.out)

    def test_export_of_an_empty_session_says_so_and_writes_nothing(self):
        self.run_cmd("/export out.md")
        self.assertIn("Nothing to export", self.out)
        self.assertFalse((self.root / "out.md").exists())

    def test_rename_sets_a_user_title_the_ai_never_overwrites(self):
        self.run_cmd("/rename My Own Name")
        state = state_module.get_session_state(1)
        self.assertEqual(state.session_title, "My Own Name")
        self.assertEqual(state.title_source, "user")

    def test_rename_without_a_name_shows_the_current_one(self):
        self.run_cmd("/rename")
        self.assertIn("Usage: /rename", self.out)

    def test_archive_hides_the_session(self):
        self.run_cmd("/archive")
        self.assertTrue(state_module.get_session_state(1).archived)

    def test_undo_drops_the_last_turn_from_memory_and_from_disk(self):
        history = [
            {"role": "user", "content": "first"}, {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"}, {"role": "assistant", "content": "two"},
        ]
        state = state_module.get_session_state(1)
        state.conversation_history = list(history)
        state_module.put_session_state(state)
        result, ctx = self.run_cmd("/undo", self.ctx(conversation_history=list(history)))
        self.assertEqual([m["content"] for m in ctx.conversation_history], ["first", "one"])
        self.assertEqual([m["content"] for m in state_module.get_session_state(1).conversation_history], ["first", "one"])
        self.assertIn("not reverted", self.out)

    def test_undo_with_nothing_to_undo_says_so(self):
        self.run_cmd("/undo")
        self.assertIn("Nothing to undo", self.out)


class PromptShapedCommandTests(_Isolated, unittest.TestCase):
    def test_init_asks_the_agent_to_create_agents_md(self):
        result, _ = self.run_cmd("/init")
        self.assertIsInstance(result, sr.Rewrite)
        self.assertTrue(result.text.startswith("/agent "))
        self.assertIn("create an AGENTS.md", result.text)

    def test_init_updates_an_existing_agents_md(self):
        (self.root / "AGENTS.md").write_text("# existing")
        result, _ = self.run_cmd("/init focus on the API")
        self.assertIn("update the existing AGENTS.md", result.text)
        self.assertIn("focus on the API", result.text)

    def test_review_is_a_read_only_audit_of_the_diff(self):
        result, _ = self.run_cmd("/review the auth changes")
        self.assertTrue(result.text.startswith("/audit "))
        self.assertIn("git diff", result.text)
        self.assertIn("Do not modify any files", result.text)
        self.assertIn("the auth changes", result.text)

    def test_security_review_is_read_only_and_security_focused(self):
        result, _ = self.run_cmd("/security-review")
        self.assertTrue(result.text.startswith("/audit "))
        self.assertIn("injection", result.text)
        self.assertIn("Do not modify any files", result.text)

    def test_interview_needs_a_request_and_uses_the_probing_tool(self):
        result, _ = self.run_cmd("/interview")
        self.assertIs(result, sr.HANDLED)
        self.assertIn("Usage: /interview", self.out)
        result, _ = self.run_cmd("/interview add SSO login")
        self.assertIn("ask_user_question", result.text)
        self.assertIn("add SSO login", result.text)


class InspectCommandTests(_Isolated, unittest.TestCase):
    def test_mcp_lists_configured_servers(self):
        (self.root / ".mcp.json").write_text(json.dumps({"mcpServers": {"files": {"command": "npx", "args": ["-y", "srv"]}}}))
        self.run_cmd("/mcp")
        self.assertIn("files", self.out)
        self.assertIn("stdio", self.out)

    def test_mcp_with_none_configured_says_where_to_add_them(self):
        with patch("tamfis_code.mcp_client.load_mcp_servers", return_value={}):  # not the host's real config
            self.run_cmd("/mcp")
        self.assertIn("No MCP servers configured", self.out)

    def test_hooks_lists_configured_hooks(self):
        (self.root / ".tamfis").mkdir()
        (self.root / ".tamfis" / "hooks.toml").write_text(
            '[[pre_tool_use]]\nmatcher = "execute_command"\ncommand = "echo checking"\n'
        )
        self.run_cmd("/hooks")
        self.assertIn("pre_tool_use", self.out)
        self.assertIn("execute_command", self.out)

    def test_hooks_with_none_says_so(self):
        with patch("tamfis_code.hooks.load_hooks", return_value=[]):  # not the host's real config
            self.run_cmd("/hooks")
        self.assertIn("No hooks configured", self.out)

    def test_skills_and_plugins_never_crash_when_empty(self):
        for command in ("/skills", "/plugins"):
            with self.subTest(command=command):
                result, _ = self.run_cmd(command)
                self.assertIs(result, sr.HANDLED)
                self.assertTrue(self.out.strip())

    def test_memory_shows_the_instruction_files_in_play(self):
        (self.root / "AGENTS.md").write_text("# rules\nbe careful\n")
        self.run_cmd("/memory")
        self.assertIn("Instruction files loaded", self.out)
        self.assertIn("AGENTS.md", self.out)

    def test_tasks_and_stop_handle_the_empty_and_unknown_cases(self):
        with patch("tamfis_code.background.list_jobs", return_value=[]):  # not the host's real jobs
            self.run_cmd("/tasks")
        self.assertIn("No background tasks", self.out)
        self.run_cmd("/stop")
        self.assertIn("Usage: /stop", self.out)
        self.run_cmd("/stop nope-123")
        self.assertIn("No running background task 'nope-123'", self.out)

    def test_version_shows_the_version_and_install_location(self):
        self.run_cmd("/version")
        self.assertIn("tamfis-code 9.9.9", self.out)
        self.assertIn("installed at", self.out)

    def test_debug_summarises_the_session(self):
        history = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y" * 400}]
        self.run_cmd("/debug", self.ctx(conversation_history=history))
        self.assertIn("messages           2", self.out)
        self.assertIn("~200 tokens", self.out)
        self.assertIn("session            1", self.out)


class SettingsCommandTests(_Isolated, unittest.TestCase):
    def test_add_dir_grants_the_session_another_directory(self):
        extra = self.base / "other"
        extra.mkdir()
        self.run_cmd(f"/add-dir {extra}")
        self.assertIn(str(extra.resolve()), state_module.get_session_state(1).allowed_workspaces)

    def test_add_dir_rejects_a_missing_directory(self):
        self.run_cmd(f"/add-dir {self.base / 'nope'}")
        self.assertIn("is not a directory", self.out)

    def test_effort_shows_sets_and_validates(self):
        from tamfis_code import runner_local

        original = runner_local.DEFAULT_REASONING_EFFORT
        try:
            self.run_cmd("/effort high")
            self.assertEqual(runner_local.DEFAULT_REASONING_EFFORT, "high")
            self.run_cmd("/effort")
            self.assertIn("high", self.out)
            self.run_cmd("/effort turbo")
            self.assertIn("Unknown effort", self.out)
            self.assertEqual(runner_local.DEFAULT_REASONING_EFFORT, "high")
        finally:
            runner_local.DEFAULT_REASONING_EFFORT = original

    def test_vim_toggles_the_prompts_editing_mode(self):
        from prompt_toolkit.enums import EditingMode

        session = SimpleNamespace(editing_mode=EditingMode.EMACS)
        self.run_cmd("/vim", self.ctx(session=session))
        self.assertEqual(session.editing_mode, EditingMode.VI)
        self.run_cmd("/vim", self.ctx(session=session))
        self.assertEqual(session.editing_mode, EditingMode.EMACS)
        self.run_cmd("/vim on", self.ctx(session=session))
        self.assertEqual(session.editing_mode, EditingMode.VI)

    def test_reload_reports_the_custom_command_count(self):
        self.run_cmd("/reload", self.ctx(reload_custom_commands=lambda: 3))
        self.assertIn("3 custom commands loaded", self.out)

    def test_feedback_is_saved_locally_and_nothing_is_sent(self):
        self.run_cmd("/feedback the footer is great")
        line = (state_module.CONFIG_DIR / "feedback.jsonl").read_text().splitlines()[-1]
        entry = json.loads(line)
        self.assertEqual(entry["feedback"], "the footer is great")
        self.assertEqual(entry["session"], 1)
        self.assertIn("nothing was sent anywhere", self.out)
        self.run_cmd("/feedback")
        self.assertIn("Usage: /feedback", self.out)


class ConsolidationTests(unittest.TestCase):
    """The comparison against the four other agents, pinned: every command concept
    they share that makes sense for a standalone agent is reachable here."""

    def _known(self):
        from tamfis_code.interactive import SLASH_COMMANDS

        return {name for name, _ in SLASH_COMMANDS}

    def test_every_shared_concept_has_a_command_or_an_alias(self):
        known = self._known()
        # concept -> commands the other agents use for it; at least one must exist here
        concepts = {
            "new conversation": {"/new", "/reset", "/clear"},
            "resume / history": {"/resume", "/history", "/sessions", "/chats"},
            "fork / branch": {"/fork", "/branch"},
            "rename": {"/rename", "/title", "/retitle"},
            "export": {"/export"},
            "compact": {"/compact"},
            "recap / summary": {"/recap", "/summary"},
            "init instructions": {"/init"},
            "review": {"/review", "/security-review"},
            "plan": {"/plan"},
            "diff": {"/diff"},
            "model": {"/model"},
            "permissions": {"/permissions"},
            "effort / reasoning": {"/effort"},
            "mcp": {"/mcp"},
            "hooks": {"/hooks"},
            "skills": {"/skills"},
            "plugins": {"/plugins", "/plugin"},
            "memory": {"/memory"},
            "add directory": {"/add-dir"},
            "cd / pwd": {"/cd", "/pwd", "/cwd"},
            "undo / rewind": {"/undo", "/rewind"},
            "tasks / background": {"/tasks", "/background"},
            "stop": {"/stop"},
            "usage / cost": {"/usage", "/cost"},
            "doctor / debug": {"/doctor", "/debug", "/diagnostics"},
            "version / release notes": {"/version", "/release-notes", "/changelog"},
            "feedback": {"/feedback"},
            "vim mode": {"/vim"},
            "queue": {"/queue"},
            "copy": {"/copy"},
            "side question": {"/btw"},
            "goal": {"/goal"},
            "image": {"/image", "/paste-image"},
            "exit": {"/exit", "/quit", "/q"},
            "help": {"/help", "/?"},
            "status": {"/status"},
            "reload": {"/reload"},
            "interview / spec": {"/interview"},
            "archive": {"/archive"},
        }
        missing = {concept: sorted(names) for concept, names in concepts.items() if not (names & known)}
        self.assertEqual(missing, {})

    def test_no_command_is_listed_twice_and_all_have_descriptions(self):
        from tamfis_code.interactive import SLASH_COMMANDS

        names = [name for name, _ in SLASH_COMMANDS]
        self.assertEqual(len(names), len(set(names)))
        for name, description in SLASH_COMMANDS:
            self.assertTrue(description.strip(), name)

    def test_every_alias_targets_a_real_command(self):
        known = self._known()
        for alias, (target, _description) in sr.ALIAS_REWRITES.items():
            self.assertIn(target, known, f"{alias} -> {target}")

    def test_the_help_text_lists_every_new_command(self):
        from tamfis_code.interactive import HELP_TEXT

        for command in sr.COMMANDS:
            self.assertIn(command.name, HELP_TEXT)

    def test_vendor_specific_commands_are_deliberately_absent(self):
        known = self._known()
        for vendor_only in ("/login", "/subscribe", "/passes", "/stickers", "/radio", "/ads:enable", "/byok", "/desktop", "/mobile"):
            self.assertNotIn(vendor_only, known)


class ReplIntegrationTests(_Isolated, unittest.TestCase):
    """Through the real REPL loop."""

    def _repl(self, inputs):
        import asyncio as _asyncio
        import io
        from unittest.mock import AsyncMock

        from tamfis_code.config import Config
        from tamfis_code.interactive import run_interactive

        buf = io.StringIO()
        console = Console(file=buf, no_color=True, width=200)
        workspace = WorkspaceContext(session_id=1, workspace_root=str(self.root))
        with patch("tamfis_code.interactive.Console", return_value=console), \
                patch("tamfis_code.interactive.PromptSession") as session_cls, \
                patch("tamfis_code.interactive.print_banner"):
            session_cls.return_value.prompt_async = AsyncMock(side_effect=inputs)
            _asyncio.run(run_interactive(client=None, config=Config(), workspace=workspace))
        return buf.getvalue()

    def test_a_registry_command_runs_inside_the_repl(self):
        out = self._repl(["/version", EOFError()])
        self.assertIn("tamfis-code", out)
        self.assertIn("installed at", out)

    def test_an_alias_that_rewrites_to_a_registry_command_resolves_in_one_go(self):
        out = self._repl(["/release-notes", EOFError()])  # -> /version
        self.assertIn("installed at", out)

    def test_the_q_alias_exits_the_repl(self):
        out = self._repl(["/q"])  # no EOFError needed: /q -> /exit ends the loop
        self.assertNotIn("is not a command", out)

    def test_new_switches_the_session_for_the_rest_of_the_repl(self):
        out = self._repl(["/new", "/debug", EOFError()])
        self.assertIn("New session", out)
        new_id = max(state_module.all_known_session_ids())
        self.assertIn(f"session            {new_id}", out)

    def test_a_typo_still_gets_a_suggestion_from_the_merged_list(self):
        out = self._repl(["/exprot", EOFError()])
        self.assertIn("is not a command", out)
        self.assertIn("/export", out)


if __name__ == "__main__":
    unittest.main()
