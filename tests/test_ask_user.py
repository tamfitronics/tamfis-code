"""Probing: the agent can ask the person a real decision, with options.

Owner request 2026-09-19 ("add such probing abilities to tamfis-code"), modelled
on Claude Code's structured questions: before a hard-to-reverse or outward-facing
action (restart a service that drops connections, decide what happens to data)
the agent asks, with concrete options and a recommended default, and continues
with the answer.
"""
from __future__ import annotations

import asyncio
import unittest
from io import StringIO
from unittest.mock import AsyncMock, patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from tamfis_code import ask_user as au
from tamfis_code.mcp import MCPServer
from tamfis_code.tool_display import display_target, summarize_result

QUESTIONS = [
    {
        "question": "Restart Caddy now? It drops connections on every site for a few seconds.",
        "header": "Restart",
        "options": [
            {"label": "Keep offline for now", "description": "Leave everything as it is"},
            {"label": "Restart now (Recommended)", "description": "A few seconds of dropped connections"},
        ],
    },
    {
        "question": "Which tooling should be cleaned?",
        "header": "Scope",
        "multiSelect": True,
        "options": [{"label": "seo script"}, {"label": "permissions script"}, {"label": "site list"}],
    },
]


async def _select(question: au.Question, keys: str, *, position=1, total=1):
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        state = au.SelectorState(question, position=position, total=total)
        return await au._run_selector(state, input=pipe, output=DummyOutput())


class NormalizeTests(unittest.TestCase):
    def test_the_recommended_option_is_moved_first(self):
        parsed = au.normalize_questions(questions=QUESTIONS)
        self.assertEqual(parsed[0].options[0].label, "Restart now (Recommended)")
        self.assertTrue(parsed[0].options[0].recommended)

    def test_the_original_single_question_form_still_works(self):
        parsed = au.normalize_questions(question="Which env?", options=["staging", "prod"])
        self.assertEqual(len(parsed), 1)
        self.assertEqual([o.label for o in parsed[0].options], ["staging", "prod"])

    def test_json_encoded_option_arrays_stay_as_whole_clickable_options(self):
        parsed = au.normalize_questions(question="Continue?", options='["Yes", "No"]')
        self.assertEqual([o.label for o in parsed[0].options], ["Yes", "No"])

    def test_plain_string_option_stays_one_option(self):
        parsed = au.normalize_questions(question="Environment?", options="staging")
        self.assertEqual([o.label for o in parsed[0].options], ["staging"])

    def test_a_question_without_options_is_allowed_as_free_text(self):
        parsed = au.normalize_questions(question="What should the release be called?")
        self.assertEqual(parsed[0].options, [])

    def test_bad_input_is_rejected_with_a_message_the_model_can_act_on(self):
        for kwargs in ({}, {"questions": []}, {"questions": [{"question": ""}]}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    au.normalize_questions(**kwargs)
        with self.assertRaisesRegex(ValueError, "at most 4"):
            au.normalize_questions(questions=[{"question": f"q{i}"} for i in range(5)])

    def test_duplicates_are_dropped_and_options_are_capped(self):
        parsed = au.normalize_questions(questions=[{
            "question": "Pick one?",
            "options": ["a", "A", *[f"o{i}" for i in range(20)]],
        }])
        labels = [o.label for o in parsed[0].options]
        self.assertEqual(labels.count("a") + labels.count("A"), 1)
        self.assertLessEqual(len(labels), au.MAX_OPTIONS)


class SelectorStateTests(unittest.TestCase):
    def _state(self, **kw):
        return au.SelectorState(au.normalize_questions(questions=[QUESTIONS[kw.get("index", 0)]])[0])

    def test_enter_on_the_default_row_chooses_the_recommended_option(self):
        self.assertEqual(self._state().confirm(), ("done", ["Restart now (Recommended)"]))

    def test_moving_wraps_and_an_other_row_is_always_present(self):
        state = self._state()
        state.move(-1)  # wraps up to the last row
        self.assertEqual(state.cursor, state.other_index)
        self.assertEqual(state.confirm(), ("other", []))

    def test_jump_selects_a_numbered_row(self):
        state = self._state()
        self.assertTrue(state.jump(2))
        self.assertEqual(state.confirm(), ("done", ["Keep offline for now"]))
        self.assertFalse(state.jump(9))

    def test_multi_select_toggles_and_confirms_the_set(self):
        state = self._state(index=1)
        state.toggle()
        state.move(2)
        state.toggle()
        self.assertEqual(state.confirm(), ("done", ["seo script", "site list"]))

    def test_multi_select_with_nothing_toggled_takes_the_highlighted_row(self):
        state = self._state(index=1)
        state.move(1)
        self.assertEqual(state.confirm(), ("done", ["permissions script"]))

    def test_the_view_shows_the_recommendation_descriptions_and_the_hint(self):
        text = "".join(fragment for _style, fragment in self._state().render())
        self.assertIn("❯ 1. Restart now (Recommended)", text)
        self.assertIn("A few seconds of dropped connections", text)
        self.assertIn("Other…", text)
        self.assertIn("Esc skip", text)


class RealKeyPressTests(unittest.TestCase):
    """Drive the real prompt_toolkit application with scripted key presses."""

    def _q(self, index):
        return au.normalize_questions(questions=[QUESTIONS[index]])[0]

    def test_enter_selects_the_recommended_option(self):
        self.assertEqual(asyncio.run(_select(self._q(0), "\r")), ("done", ["Restart now (Recommended)"]))

    def test_arrow_down_then_enter_selects_the_next_option(self):
        self.assertEqual(asyncio.run(_select(self._q(0), "\x1b[B\r")), ("done", ["Keep offline for now"]))

    def test_a_digit_selects_immediately(self):
        self.assertEqual(asyncio.run(_select(self._q(0), "2")), ("done", ["Keep offline for now"]))

    def test_space_toggles_in_multi_select(self):
        result = asyncio.run(_select(self._q(1), " \x1b[B\x1b[B \r"))
        self.assertEqual(result, ("done", ["seo script", "site list"]))

    def test_ctrl_c_skips(self):
        self.assertEqual(asyncio.run(_select(self._q(0), "\x03"))[0], "skip")


class AskQuestionsTests(unittest.TestCase):
    def test_answers_come_back_in_order_and_other_prompts_for_free_text(self):
        parsed = au.normalize_questions(questions=QUESTIONS)
        script = iter([("done", ["Restart now (Recommended)"]), ("other", [])])

        async def selector(state, **_):
            return next(script)

        with patch.object(au, "_prompt_free_text", AsyncMock(return_value="all three")):
            answers = asyncio.run(au.ask_questions(Console(file=StringIO()), parsed, selector=selector))
        self.assertEqual(answers[0].text(), "Restart now (Recommended)")
        self.assertEqual(answers[1].text(), "all three")
        self.assertTrue(answers[1].other)

    def test_a_skipped_question_says_so(self):
        parsed = au.normalize_questions(questions=[QUESTIONS[0]])

        async def selector(state, **_):
            return ("skip", [])

        answers = asyncio.run(au.ask_questions(Console(file=StringIO()), parsed, selector=selector))
        self.assertEqual(answers[0].text(), "(no answer given)")

    def test_the_result_text_lists_every_question_and_answer(self):
        parsed = au.normalize_questions(questions=QUESTIONS)
        text = au.answers_to_text(parsed, [au.Answer(["Restart now (Recommended)"]), au.Answer(["a", "b"])])
        self.assertEqual(text.splitlines()[0], "User answered the agent's questions:")
        self.assertIn("→ Restart now (Recommended)", text)
        self.assertIn("→ a, b", text)


class _FakeConsole:
    """Just enough of a console for the numbered-prompt fallback."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.printed = []

    def print(self, *args, **kwargs):
        self.printed.append(" ".join(str(a) for a in args))

    def input(self, prompt=""):
        self.printed.append(prompt)
        return self.answers.pop(0)


class NumberedPromptFallbackTests(unittest.TestCase):
    """No real terminal (piped stdin, CI): the plain numbered prompt."""

    def _ask(self, question, answers):
        console = _FakeConsole(answers)
        parsed = au.normalize_questions(questions=[question])
        with patch.object(au, "_has_terminal", return_value=False):
            result = asyncio.run(au.ask_questions(console, parsed))
        return result[0], console

    def test_a_number_picks_the_option(self):
        answer, console = self._ask(QUESTIONS[0], ["2"])
        self.assertEqual(answer.text(), "Keep offline for now")  # option 2 after recommended-first sorting
        self.assertTrue(any("1. Restart now (Recommended)" in line for line in console.printed))

    def test_free_text_is_always_accepted(self):
        answer, _ = self._ask(QUESTIONS[0], ["restart tomorrow morning"])
        self.assertEqual(answer.text(), "restart tomorrow morning")
        self.assertTrue(answer.other)

    def test_multi_select_takes_comma_separated_numbers(self):
        answer, _ = self._ask(QUESTIONS[1], ["1, 3"])
        self.assertEqual(answer.text(), "seo script, site list")

    def test_an_empty_reply_skips_and_a_missing_option_number_is_free_text(self):
        answer, _ = self._ask(QUESTIONS[0], [""])
        self.assertTrue(answer.skipped)
        answer, _ = self._ask(QUESTIONS[0], ["9"])
        self.assertEqual(answer.text(), "9")

    def test_a_question_without_options_returns_the_typed_answer(self):
        answer, _ = self._ask({"question": "What should it be called?"}, ["release-x"])
        self.assertEqual(answer.text(), "release-x")

    def test_a_selector_that_cannot_start_falls_back_to_the_numbered_prompt(self):
        parsed = au.normalize_questions(questions=[QUESTIONS[0]])
        console = _FakeConsole(["1"])

        async def broken(state, **_):
            raise RuntimeError("no tty")

        answers = asyncio.run(au.ask_questions(console, parsed, selector=broken))
        self.assertEqual(answers[0].text(), "Restart now (Recommended)")


class ToolHandlerTests(unittest.TestCase):
    def _server(self, *, interactive):
        return MCPServer(workspace_root="/tmp", session_id=1, console=Console(file=StringIO()), interactive=interactive)

    def test_the_tool_advertises_structured_questions_and_recommended_first(self):
        tool = self._server(interactive=True).tools["ask_user_question"]
        self.assertIn("questions", tool.parameters["properties"])
        self.assertIn("(Recommended)", tool.description)
        self.assertIn("BEFORE any hard-to-reverse", tool.description)
        self.assertEqual(tool.parameters["required"], [])

    def test_non_interactive_says_so_and_tells_the_model_not_to_guess_on_risky_decisions(self):
        result = asyncio.run(self._server(interactive=False)._ask_user_question(questions=QUESTIONS))
        self.assertIn("unavailable", result)
        self.assertIn("reversible/safest", result)

    def test_invalid_arguments_are_reported_not_raised(self):
        result = asyncio.run(self._server(interactive=True)._ask_user_question())
        self.assertIn("could not be shown", result)

    def test_the_original_single_question_form_still_returns_just_the_answer(self):
        async def fake_ask(console, parsed, **_):
            return [au.Answer(["staging"])]

        with patch.object(au, "ask_questions", fake_ask):
            result = asyncio.run(self._server(interactive=True)._ask_user_question(question="Which env?", options=["staging", "prod"]))
        self.assertEqual(result, "staging")

    def test_interactive_runs_the_selector_and_returns_the_answers(self):
        async def fake_ask(console, parsed, **_):
            return [au.Answer(["Restart now (Recommended)"]), au.Answer(["seo script"])]

        with patch.object(au, "ask_questions", fake_ask):
            result = asyncio.run(self._server(interactive=True)._ask_user_question(questions=QUESTIONS))
        self.assertTrue(result.startswith("User answered the agent's questions:"))
        self.assertIn("→ Restart now (Recommended)", result)

    def test_live_input_is_suspended_around_the_question_and_always_resumed(self):
        events = []

        class _Renderer:
            async def suspend_live_async(self):
                events.append("suspended")

            def resume_live(self):
                events.append("resumed")

        server = self._server(interactive=True)
        server._renderer = _Renderer()

        async def failing(console, parsed, **_):
            events.append("asked")
            raise RuntimeError("terminal went away")

        with patch.object(au, "ask_questions", failing):
            with self.assertRaises(RuntimeError):
                asyncio.run(server._ask_user_question(question="Proceed?", options=["yes", "no"]))
        self.assertEqual(events, ["suspended", "asked", "resumed"])


class DisplayTests(unittest.TestCase):
    def test_the_header_names_the_question_or_the_count(self):
        self.assertEqual(display_target("ask_user_question", {"questions": QUESTIONS}), "2 questions")
        self.assertIn("Restart Caddy", display_target("ask_user_question", {"questions": QUESTIONS[:1]}))
        self.assertEqual(display_target("ask_user_question", {"question": "Ship it?"}), "Ship it?")

    def test_the_result_shows_every_answer_line(self):
        parsed = au.normalize_questions(questions=QUESTIONS)
        text = au.answers_to_text(parsed, [au.Answer(["Restart now (Recommended)"]), au.Answer(["a"])])
        lines, failed = summarize_result("ask_user_question", {"tool": "ask_user_question", "success": True, "result": text})
        self.assertFalse(failed)
        self.assertEqual(lines[0], "User answered the agent's questions:")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("· Restart Caddy now?"))


class SystemPromptGuidanceTests(unittest.TestCase):
    def test_the_prompt_tells_the_model_when_to_ask_and_when_not_to(self):
        import tempfile
        from pathlib import Path

        from tamfis_code import state as state_module
        from tamfis_code.workspace import build_system_prompt

        with tempfile.TemporaryDirectory() as ws:
            originals = (state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH)
            base = Path(ws) / "cfg"
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH = (
                base, base / "state.json", base / ".lock",
            )
            state_module._STATE_CACHE = None
            try:
                prompt = build_system_prompt(1, Path(ws))
            finally:
                state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH = originals
                state_module._STATE_CACHE = None
        self.assertIn("Ask before you act on a real decision", prompt)
        self.assertIn("ask_user_question", prompt)
        self.assertIn("(Recommended)", prompt)
        self.assertIn("never ask what a tool can tell you", prompt)


if __name__ == "__main__":
    unittest.main()
