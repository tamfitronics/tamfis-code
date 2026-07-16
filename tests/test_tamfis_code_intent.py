import unittest

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


if __name__ == "__main__":
    unittest.main()
