"""Tests for user-defined custom slash commands (custom_commands.py) -- a
real Claude Code/Codex-style parity gap: SLASH_COMMANDS used to be a
hardcoded tuple with no way for a user to add their own /command without
editing source."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamfis_code import custom_commands as custom_commands_module
from tamfis_code.custom_commands import CustomCommand, expand_custom_command, load_custom_commands


class TestLoadCustomCommands:
    def setup_method(self):
        self._original_dir = custom_commands_module.USER_COMMANDS_DIR
        self.tmp = tempfile.TemporaryDirectory()
        custom_commands_module.USER_COMMANDS_DIR = Path(self.tmp.name) / "user" / "commands"

    def teardown_method(self):
        custom_commands_module.USER_COMMANDS_DIR = self._original_dir
        self.tmp.cleanup()

    def test_missing_directories_return_no_commands(self):
        assert load_custom_commands(str(Path(self.tmp.name) / "project")) == {}

    def test_loads_a_plain_command_with_no_frontmatter(self):
        custom_commands_module.USER_COMMANDS_DIR.mkdir(parents=True)
        (custom_commands_module.USER_COMMANDS_DIR / "review.md").write_text(
            "Review the current diff for bugs.\n"
        )
        loaded = load_custom_commands()
        assert set(loaded) == {"review"}
        assert loaded["review"].template == "Review the current diff for bugs."
        assert loaded["review"].source == "user config"
        assert "review.md" in loaded["review"].description

    def test_loads_frontmatter_description(self):
        custom_commands_module.USER_COMMANDS_DIR.mkdir(parents=True)
        (custom_commands_module.USER_COMMANDS_DIR / "secreview.md").write_text(
            "---\ndescription: Review a diff for security issues\n---\n"
            "Review the diff for security issues: $ARGUMENTS\n"
        )
        loaded = load_custom_commands()
        assert loaded["secreview"].description == "Review a diff for security issues"
        assert loaded["secreview"].template == "Review the diff for security issues: $ARGUMENTS"

    def test_project_command_overrides_same_named_user_command(self):
        custom_commands_module.USER_COMMANDS_DIR.mkdir(parents=True)
        (custom_commands_module.USER_COMMANDS_DIR / "review.md").write_text("user version")
        project_root = Path(self.tmp.name) / "project"
        (project_root / ".tamfis" / "commands").mkdir(parents=True)
        (project_root / ".tamfis" / "commands" / "review.md").write_text("project version")
        loaded = load_custom_commands(str(project_root))
        assert loaded["review"].template == "project version"
        assert loaded["review"].source == "project config"

    def test_non_md_files_are_ignored(self):
        custom_commands_module.USER_COMMANDS_DIR.mkdir(parents=True)
        (custom_commands_module.USER_COMMANDS_DIR / "notes.txt").write_text("not a command")
        assert load_custom_commands() == {}

    def test_invalid_command_name_characters_are_rejected(self):
        # Filenames can't be regex-unsafe/path-unsafe in ways that would
        # make a "/name" dispatch check ambiguous or unsafe.
        custom_commands_module.USER_COMMANDS_DIR.mkdir(parents=True)
        (custom_commands_module.USER_COMMANDS_DIR / "not a valid name!.md").write_text("hi")
        assert load_custom_commands() == {}

    def test_empty_body_after_frontmatter_is_rejected(self):
        custom_commands_module.USER_COMMANDS_DIR.mkdir(parents=True)
        (custom_commands_module.USER_COMMANDS_DIR / "empty.md").write_text("---\ndescription: x\n---\n")
        assert load_custom_commands() == {}


class TestExpandCustomCommand:
    def test_substitutes_arguments_placeholder(self):
        command = CustomCommand(name="x", description="", template="Look at $ARGUMENTS please.", source="user config")
        assert expand_custom_command(command, "app.py") == "Look at app.py please."

    def test_appends_arguments_when_no_placeholder_present(self):
        command = CustomCommand(name="x", description="", template="Review the diff.", source="user config")
        assert expand_custom_command(command, "focus on auth") == "Review the diff.\n\nfocus on auth"

    def test_no_arguments_and_no_placeholder_returns_template_unchanged(self):
        command = CustomCommand(name="x", description="", template="Review the diff.", source="user config")
        assert expand_custom_command(command, "") == "Review the diff."
