"""Tests for reading other AI coding agents' session history (Claude Code,
Codex CLI, GitHub Copilot CLI, and the generic best-effort adapter used for
OpenCode/Kimi Code/user-configured stores).

Fixture shapes below were copied from real on-disk stores (~/.claude,
~/.codex, ~/.copilot) rather than guessed -- see external_agents.py's module
docstring for where that ground truth came from.
"""
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import external_agents as ea
from tamfis_code.render import render_external_sessions, render_update_notice
from rich.console import Console


class _HomeFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self._patcher = patch.object(ea, "_home", return_value=self.home)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()


class ClaudeCodeAdapterTests(_HomeFixture):
    def _write_session(self, session_id: str, lines: list[dict]) -> Path:
        project_dir = self.home / ".claude" / "projects" / "-home"
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / f"{session_id}.jsonl"
        with path.open("w") as fh:
            for entry in lines:
                fh.write(json.dumps(entry) + "\n")
        return path

    def test_discovers_a_session_with_its_ai_title_and_cwd(self):
        self._write_session("abc-123", [
            {"type": "ai-title", "aiTitle": "Fix the flaky auth test", "sessionId": "abc-123"},
            {"type": "user", "cwd": "/home", "message": {"role": "user", "content": "Fix the flaky auth test"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}},
        ])
        sessions = ea._claude_code_discover(self.home)
        self.assertEqual(len(sessions), 1)
        session = sessions[0]
        self.assertEqual(session.tool, "claude-code")
        self.assertEqual(session.session_id, "abc-123")
        self.assertEqual(session.title, "Fix the flaky auth test")
        self.assertEqual(session.cwd, "/home")
        self.assertEqual(session.turn_count, 2)

    def test_falls_back_to_the_first_user_message_when_theres_no_ai_title(self):
        self._write_session("no-title", [
            {"type": "user", "cwd": "/home", "message": {"role": "user", "content": "Investigate the timeout bug"}},
        ])
        sessions = ea._claude_code_discover(self.home)
        self.assertEqual(sessions[0].title, "Investigate the timeout bug")

    def test_a_session_with_no_user_or_assistant_turns_is_skipped(self):
        self._write_session("empty", [{"type": "mode", "mode": "normal"}])
        self.assertEqual(ea._claude_code_discover(self.home), [])

    def test_read_extracts_text_from_content_block_lists(self):
        self._write_session("blocks", [
            {"type": "user", "cwd": "/home", "message": {"role": "user", "content": "Add a health check"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Added "}, {"type": "tool_use", "id": "t1", "name": "write_file"},
                {"type": "text", "text": "/health."},
            ]}},
        ])
        session = ea._claude_code_discover(self.home)[0]
        turns = ea._claude_code_read(session, 20_000)
        self.assertEqual(turns[0].role, "user")
        self.assertEqual(turns[0].text, "Add a health check")
        self.assertEqual(turns[1].role, "assistant")
        self.assertEqual(turns[1].text, "Added \n/health.")

    def test_a_malformed_line_does_not_break_discovery_of_the_rest_of_the_file(self):
        path = self._write_session("bad-line", [
            {"type": "user", "cwd": "/home", "message": {"role": "user", "content": "Real turn"}},
        ])
        with path.open("a") as fh:
            fh.write("{not json\n")
        sessions = ea._claude_code_discover(self.home)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].turn_count, 1)


class CodexAdapterTests(_HomeFixture):
    def _write_rollout(self, session_id: str, items: list[dict]) -> Path:
        day_dir = self.home / ".codex" / "sessions" / "2026" / "09" / "09"
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"rollout-2026-09-09T12-00-00-{session_id}.jsonl"
        meta = {
            "timestamp": "2026-09-09T12:00:00.000Z", "type": "session_meta",
            "payload": {"session_id": session_id, "cwd": "/home", "timestamp": "2026-09-09T12:00:00.000Z"},
        }
        with path.open("w") as fh:
            fh.write(json.dumps(meta) + "\n")
            for item in items:
                fh.write(json.dumps(item) + "\n")
        return path

    def _write_index(self, entries: list[dict]) -> None:
        codex_home = self.home / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        with (codex_home / "session_index.jsonl").open("w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")

    def test_discovers_a_session_and_prefers_the_latest_index_title(self):
        self._write_rollout("01a0-1", [])
        self._write_index([
            {"id": "01a0-1", "thread_name": "first draft title"},
            {"id": "01a0-1", "thread_name": "Fix serialized training status"},
        ])
        sessions = ea._codex_discover(self.home)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].title, "Fix serialized training status")
        self.assertEqual(sessions[0].cwd, "/home")

    def test_read_skips_the_synthetic_environment_context_preamble(self):
        path = self._write_rollout("01a0-2", [
            {"type": "response_item", "timestamp": "t1", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "<environment_context>\n<cwd>/home</cwd>\n</environment_context>"}],
            }},
            {"type": "response_item", "timestamp": "t2", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "Actually fix the retry loop"}],
            }},
            {"type": "response_item", "timestamp": "t3", "payload": {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Fixed the retry loop."}],
            }},
        ])
        session = ea.ExternalSession(tool="codex", session_id="01a0-2", title="", cwd="/home", updated_at="", path=str(path))
        turns = ea._codex_read(session, 20_000)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].text, "Actually fix the retry loop")
        self.assertEqual(turns[1].role, "assistant")

    def test_a_session_with_no_index_entry_still_discovers_with_an_empty_title(self):
        self._write_rollout("01a0-3", [])
        sessions = ea._codex_discover(self.home)
        self.assertEqual(sessions[0].title, "")


class CopilotAdapterTests(_HomeFixture):
    def _write_session(self, session_id: str, workspace_extra: str = "", events: list[dict] | None = None) -> Path:
        session_dir = self.home / ".copilot" / "session-state" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "workspace.yaml").write_text(
            f"id: {session_id}\ncwd: /home\nname: 'Continue what Claude was doing'\n"
            f"updated_at: 2026-09-11T11:31:03.659Z\n{workspace_extra}"
        )
        if events is not None:
            with (session_dir / "events.jsonl").open("w") as fh:
                for event in events:
                    fh.write(json.dumps(event) + "\n")
        return session_dir

    def test_discovers_a_session_from_its_workspace_yaml(self):
        self._write_session("cp-1", events=[])
        sessions = ea._copilot_discover(self.home)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].tool, "copilot")
        self.assertEqual(sessions[0].session_id, "cp-1")
        self.assertEqual(sessions[0].cwd, "/home")
        self.assertEqual(sessions[0].title, "Continue what Claude was doing")

    def test_condenses_a_pasted_content_marker_in_the_title(self):
        session_dir = self.home / ".copilot" / "session-state" / "cp-2"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "workspace.yaml").write_text(
            "id: cp-2\ncwd: /home\n"
            'name: \'fix it: <pasted_content file="x.txt" size="1KB" lines="10" />\'\n'
        )
        sessions = ea._copilot_discover(self.home)
        self.assertIn("[pasted content]", sessions[0].title)

    def test_read_extracts_user_and_assistant_events(self):
        session_dir = self._write_session("cp-3", events=[
            {"type": "session.start", "data": {}},
            {"type": "user.message", "data": {"content": "Continue the work"}, "timestamp": "t1"},
            {"type": "model.turn_ended", "data": {"content": "Continuing now."}, "timestamp": "t2"},
        ])
        session = ea.ExternalSession(tool="copilot", session_id="cp-3", title="", cwd="/home", updated_at="", path=str(session_dir / "events.jsonl"))
        turns = ea._copilot_read(session, 20_000)
        self.assertEqual([(t.role, t.text) for t in turns], [("user", "Continue the work"), ("assistant", "Continuing now.")])


class GenericAdapterTests(_HomeFixture):
    def test_discovers_and_reads_a_kimi_code_style_json_session(self):
        sessions_dir = self.home / ".kimi-code" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "s1.json").write_text(json.dumps({
            "id": "s1", "title": "Refactor the router", "cwd": "/home",
            "updated_at": "2026-09-01T00:00:00Z",
            "messages": [
                {"role": "user", "content": "Refactor the router"},
                {"role": "assistant", "content": "Done."},
            ],
        }))
        sessions = ea._kimi_code_discover(self.home)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].title, "Refactor the router")
        turns = ea._generic_read(sessions[0], 20_000)
        self.assertEqual(len(turns), 2)

    def test_skips_files_that_look_like_credentials(self):
        sessions_dir = self.home / ".kimi-code" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "auth_token.json").write_text(json.dumps({"id": "secret", "title": "should never load"}))
        self.assertEqual(ea._kimi_code_discover(self.home), [])

    def test_a_malformed_json_file_is_silently_skipped(self):
        sessions_dir = self.home / ".kimi-code" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "broken.json").write_text("{not valid json")
        self.assertEqual(ea._kimi_code_discover(self.home), [])


class DiscoverExternalSessionsTests(_HomeFixture):
    def _claude_session(self, session_id: str, cwd: str) -> None:
        project_dir = self.home / ".claude" / "projects" / "-home"
        project_dir.mkdir(parents=True, exist_ok=True)
        with (project_dir / f"{session_id}.jsonl").open("w") as fh:
            fh.write(json.dumps({"type": "user", "cwd": cwd, "message": {"role": "user", "content": "hi"}}) + "\n")

    def test_filters_by_workspace_root_but_keeps_sessions_with_no_recorded_cwd(self):
        self._claude_session("match", "/home/project-a")
        self._claude_session("mismatch", "/home/project-b")
        self._claude_session("unknown-cwd", "")
        sessions = ea.discover_external_sessions(workspace_root="/home/project-a")
        ids = {s.session_id for s in sessions}
        self.assertIn("match", ids)
        self.assertIn("unknown-cwd", ids)
        self.assertNotIn("mismatch", ids)

    def test_a_session_launched_from_the_workspace_parent_is_relevant(self):
        self._claude_session("parent", "/home")
        sessions = ea.discover_external_sessions(workspace_root="/home/project-a")
        self.assertIn("parent", {s.session_id for s in sessions})

    def test_a_child_repo_session_is_not_included_for_its_parent(self):
        self._claude_session("child", "/home/project-a")
        sessions = ea.discover_external_sessions(workspace_root="/home")
        self.assertNotIn("child", {s.session_id for s in sessions})

    def test_an_unknown_tool_name_is_ignored_rather_than_raising(self):
        self._claude_session("s1", "/home")
        sessions = ea.discover_external_sessions(tools=["not-a-real-tool"])
        self.assertEqual(sessions, [])

    def test_limit_is_respected(self):
        for i in range(5):
            self._claude_session(f"s{i}", "/home")
        sessions = ea.discover_external_sessions(limit=2)
        self.assertEqual(len(sessions), 2)

    def test_sorts_newest_first(self):
        self._claude_session("older", "/home")
        import time
        time.sleep(0.01)
        self._claude_session("newer", "/home")
        sessions = ea.discover_external_sessions()
        self.assertEqual(sessions[0].session_id, "newer")


class ReadExternalSessionTests(_HomeFixture):
    def test_returns_none_for_an_unknown_tool(self):
        self.assertIsNone(ea.read_external_session("not-a-tool", "x"))

    def test_returns_none_for_a_missing_session_id(self):
        self.assertIsNone(ea.read_external_session("claude-code", "does-not-exist"))

    def test_reads_a_real_session_end_to_end(self):
        project_dir = self.home / ".claude" / "projects" / "-home"
        project_dir.mkdir(parents=True, exist_ok=True)
        with (project_dir / "e2e.jsonl").open("w") as fh:
            fh.write(json.dumps({"type": "ai-title", "aiTitle": "End to end test"}) + "\n")
            fh.write(json.dumps({"type": "user", "cwd": "/home", "message": {"role": "user", "content": "Do the thing"}}) + "\n")
        record = ea.read_external_session("claude-code", "e2e")
        self.assertEqual(record["title"], "End to end test")
        self.assertEqual(record["cwd"], "/home")
        self.assertEqual(record["turns"], [{"role": "user", "text": "Do the thing", "timestamp": ""}])

    def test_accepts_an_unambiguous_session_id_prefix(self):
        project_dir = self.home / ".claude" / "projects" / "-home"
        project_dir.mkdir(parents=True, exist_ok=True)
        session_id = "dd1f76ec-20f9-41aa-8133-b9f4e036a1c9"
        with (project_dir / f"{session_id}.jsonl").open("w") as fh:
            fh.write(json.dumps({"type": "user", "cwd": "/home", "message": {"role": "user", "content": "Continue"}}) + "\n")
        record = ea.read_external_session("claude-code", "dd1f76ec-20f")
        self.assertEqual(record["session_id"], session_id)

    def test_rejects_an_ambiguous_session_id_prefix(self):
        project_dir = self.home / ".claude" / "projects" / "-home"
        project_dir.mkdir(parents=True, exist_ok=True)
        for session_id in ("same-prefix-one", "same-prefix-two"):
            with (project_dir / f"{session_id}.jsonl").open("w") as fh:
                fh.write(json.dumps({"type": "user", "cwd": "/home", "message": {"role": "user", "content": "Continue"}}) + "\n")
        self.assertIsNone(ea.read_external_session("claude-code", "same-prefix"))


class ContinuationBriefTests(unittest.TestCase):
    def test_renders_title_cwd_and_turns(self):
        record = {
            "tool": "codex", "session_id": "abc", "title": "Fix the retry loop",
            "cwd": "/home", "updated_at": "",
            "turns": [{"role": "user", "text": "Please continue", "timestamp": ""}],
        }
        brief = ea.continuation_brief(record)
        self.assertIn("codex", brief)
        self.assertIn("Fix the retry loop", brief)
        self.assertIn("/home", brief)
        self.assertIn("[user] Please continue", brief)

    def test_truncates_an_oversized_transcript_from_the_front(self):
        record = {
            "tool": "codex", "session_id": "abc", "title": "", "cwd": "", "updated_at": "",
            "turns": [{"role": "user", "text": "x" * 100, "timestamp": ""}],
        }
        brief = ea.continuation_brief(record, max_chars=20)
        self.assertIn("truncated", brief)
        self.assertLessEqual(len(brief.splitlines()[-1]), 20)

    def test_empty_transcript_says_so_explicitly(self):
        record = {"tool": "codex", "session_id": "abc", "title": "", "cwd": "", "updated_at": "", "turns": []}
        self.assertIn("no transcript content recovered", ea.continuation_brief(record))


class KnownToolsTests(unittest.TestCase):
    def test_includes_every_built_in_adapter(self):
        self.assertEqual(
            set(ea.known_tools()),
            {"claude-code", "codex", "copilot", "opencode", "kimi-code"},
        )


class ExternalSessionRenderingTests(unittest.TestCase):
    def test_compact_picker_keeps_a_copyable_prefix_at_80_columns(self):
        stream = io.StringIO()
        console = Console(file=stream, width=80, no_color=True)
        session = ea.ExternalSession(
            tool="claude-code",
            session_id="dd1f76ec-20f9-41aa-8133-b9f4e036a1c9",
            title="Continue a detailed cross-agent implementation",
            cwd="/home/tamfiscode",
            updated_at="",
            path="/tmp/session.jsonl",
        )
        render_external_sessions(console, [session], title="External sessions · current workspace")
        rendered = stream.getvalue()
        self.assertIn("Claude Code", rendered)
        self.assertIn("dd1f76ec-20f", rendered)
        self.assertIn("Continue a detailed cross-agent implementation", rendered)
        self.assertIn("Session prefix works with continue-from and show", rendered)

    def test_update_notice_exposes_click_keyboard_and_command_actions(self):
        stream = io.StringIO()
        console = Console(file=stream, width=100, no_color=True)
        render_update_notice(console, current="1.0.0", available="1.1.0")
        rendered = stream.getvalue()
        self.assertIn("Update available", rendered)
        self.assertIn("v1.0.0", rendered)
        self.assertIn("v1.1.0", rendered)
        self.assertIn("Install & restart", rendered)
        self.assertIn("Press Ctrl+U or type /update", rendered)


if __name__ == "__main__":
    unittest.main()
