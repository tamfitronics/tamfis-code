import unittest

from tamfis_code.custom_commands import CustomCommand
from tamfis_code.interactive import parse_intent


class ParseIntentTests(unittest.TestCase):
    def test_dollar_prefix_is_shell(self):
        intent = parse_intent("$ pwd")
        self.assertEqual(intent.kind, "shell")
        self.assertEqual(intent.command, "pwd")

    def test_run_prefix_is_shell(self):
        intent = parse_intent("/run npm test")
        self.assertEqual(intent.kind, "shell")
        self.assertEqual(intent.command, "npm test")

    def test_shell_prefix_is_shell(self):
        intent = parse_intent("/shell ls -la")
        self.assertEqual(intent.kind, "shell")
        self.assertEqual(intent.command, "ls -la")

    def test_audit_prefix_sets_mode(self):
        intent = parse_intent("/audit find bugs")
        self.assertEqual(intent.kind, "ai")
        self.assertEqual(intent.mode, "audit")
        self.assertEqual(intent.objective, "find bugs")

    def test_plan_prefix_sets_mode(self):
        intent = parse_intent("/plan fix the bug")
        self.assertEqual(intent.mode, "plan")

    def test_execute_prefix_sets_mode(self):
        intent = parse_intent("/execute do the thing")
        self.assertEqual(intent.mode, "execute")

    def test_chat_prefix_sets_read_only_chat_mode(self):
        intent = parse_intent("/chat explain this module")
        self.assertEqual(intent.kind, "ai")
        self.assertEqual(intent.mode, "chat")

    def test_agent_prefix_sets_execute_mode(self):
        intent = parse_intent("/agent implement the fix")
        self.assertEqual(intent.mode, "execute")

    def test_execute_plan_selects_saved_plan(self):
        intent = parse_intent("/execute-plan plan_123")
        self.assertEqual(intent.kind, "saved_plan")
        self.assertEqual(intent.command, "plan_123")

    def test_ask_prefix_is_coding_mode(self):
        intent = parse_intent("/ask what is this")
        self.assertEqual(intent.kind, "ai")
        self.assertEqual(intent.mode, "coding")
        self.assertEqual(intent.objective, "what is this")

    def test_plain_text_defaults_to_ai_coding(self):
        intent = parse_intent("hello there, please help")
        self.assertEqual(intent.kind, "ai")
        self.assertEqual(intent.mode, "coding")
        self.assertEqual(intent.objective, "hello there, please help")

    def test_unrecognised_slash_command_falls_through_to_ai_not_shell(self):
        # A typo'd slash command must not silently become a shell
        # execution attempt -- unrecognised prefixes fall through to the
        # AI path, same as plain prose.
        intent = parse_intent("/unknown-command do a thing")
        self.assertEqual(intent.kind, "ai")

    def test_custom_command_expands_into_an_ai_objective(self):
        commands = {
            "review": CustomCommand(
                name="review", description="", template="Review $ARGUMENTS for bugs.", source="user config",
            )
        }
        intent = parse_intent("/review app.py", custom_commands=commands)
        self.assertEqual(intent.kind, "ai")
        self.assertEqual(intent.mode, "coding")
        self.assertEqual(intent.objective, "Review app.py for bugs.")

    def test_custom_command_with_no_arguments_still_expands(self):
        commands = {"standup": CustomCommand(name="standup", description="", template="Summarize today's changes.", source="user config")}
        intent = parse_intent("/standup", custom_commands=commands)
        self.assertEqual(intent.objective, "Summarize today's changes.")

    def test_built_in_command_wins_over_a_same_named_custom_command(self):
        commands = {"plan": CustomCommand(name="plan", description="", template="should never be used", source="user config")}
        intent = parse_intent("/plan fix the bug", custom_commands=commands)
        self.assertEqual(intent.mode, "plan")
        self.assertEqual(intent.objective, "fix the bug")

    def test_unknown_slash_command_with_custom_commands_present_still_falls_through(self):
        commands = {"review": CustomCommand(name="review", description="", template="x", source="user config")}
        intent = parse_intent("/totally-unknown", custom_commands=commands)
        self.assertEqual(intent.kind, "ai")
        self.assertEqual(intent.objective, "/totally-unknown")


if __name__ == "__main__":
    unittest.main()
