"""The live "think card": reasoning streamed by reasoning-capable routes is
shown while the model thinks, summarised once, and never leaks between turns.

Covers the three layers end to end:
  * think_card.py pure layout (card shape, bounds, empty cases, summary line)
  * StreamRenderer event lifecycle (buffer builds, freezes at first answer
    delta, resets on task_started, exactly one durable summary line)
  * LiveInputListener._composer_message integration (card visible while
    thinking, gone once the answer starts)
"""
import shutil
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from prompt_toolkit.formatted_text import HTML as _PT_HTML  # noqa: F401  (parity with other composer tests)

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.live_input import LiveInputListener
from tamfis_code.render import StreamRenderer
from tamfis_code.think_card import think_card_html, thought_summary_line


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200, force_terminal=False)


def _config() -> Config:
    cfg = Config.__new__(Config)
    cfg.approval_policy = "ask"
    return cfg


def _plain(line: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", line)


class ThinkCardLayoutTests(unittest.TestCase):
    def test_card_renders_a_boxed_tail_of_the_reasoning(self):
        text = " ".join(f"word{i}" for i in range(60))
        lines = think_card_html(text, width=80)
        self.assertGreaterEqual(len(lines), 4)  # top + >=2 rows + bottom
        plain = [_plain(line) for line in lines]
        self.assertTrue(plain[0].startswith("╭"))
        self.assertIn("Thinking", plain[0])
        self.assertTrue(plain[-1].startswith("╰"))
        for row in plain[1:-1]:
            self.assertTrue(row.startswith("│"))
            self.assertTrue(row.rstrip().endswith("│"))

    def test_card_is_empty_below_the_signal_threshold(self):
        self.assertEqual(think_card_html("hmm", width=80), [])

    def test_card_is_empty_for_blank_text(self):
        self.assertEqual(think_card_html("", width=80), [])
        self.assertEqual(think_card_html("   \n  ", width=80), [])

    def test_card_keeps_the_latest_reasoning_within_bounds(self):
        text = " ".join(f"word{i}" for i in range(500))
        lines = think_card_html(text, width=80)
        plain = [_plain(line) for line in lines]
        # Bounded card: never more than the max row count plus the borders.
        self.assertLessEqual(len(plain), 8)
        # The newest words survive; the oldest are the ones dropped.
        self.assertIn("word499", " ".join(plain))
        self.assertNotIn("word0", " ".join(plain))

    def test_card_terminates_markup_and_escapes_angle_brackets(self):
        lines = think_card_html("checking <SuspiciousTag> and things " * 4, width=80)
        joined = "".join(lines)
        self.assertTrue(joined.endswith("</ansicyan>"))
        self.assertNotIn("<SuspiciousTag>", joined)

    def test_summary_line_formats_seconds_and_minutes(self):
        self.assertEqual(thought_summary_line(7.4), "✻ Thought for 7s")
        self.assertEqual(thought_summary_line(75), "✻ Thought for 1m 15s")
        self.assertEqual(thought_summary_line(0), "✻ Thought for 0s")


class _RendererStateMixin(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


class RendererThinkCardLifecycleTests(_RendererStateMixin, unittest.TestCase):
    def _renderer(self) -> StreamRenderer:
        return StreamRenderer(_console())

    def test_reasoning_deltas_build_a_visible_card(self):
        renderer = self._renderer()
        renderer._model = "deepresearch-pro"
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        for chunk in ("Reading ", "resume.py ", "to decide ", "what to fix ", "next"):
            renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": chunk}})
        lines = renderer._think_card_lines(80)
        self.assertTrue(lines)
        plain = " ".join(_plain(line) for line in lines)
        self.assertIn("resume.py", plain)

    def test_first_answer_delta_freezes_thinking_and_prints_one_summary_line(self):
        renderer = self._renderer()
        out = StringIO()
        renderer.console = Console(file=out, no_color=True, width=120)
        renderer._model = "deepresearch-pro"
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "analysis " * 40}})
        # No durable line while still thinking.
        self.assertEqual(out.getvalue(), "")
        # Simulate a real thinking pause (instant test events would measure 0s,
        # which the 1s threshold correctly refuses to summarise).
        import time as _time

        renderer._reasoning_start = _time.monotonic() - 12.0
        # The answer starts: thinking freezes, card closes.
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Here is the fix."}})
        self.assertIsNotNone(renderer._thought_seconds)
        rendered = out.getvalue()
        self.assertIn("Thought for", rendered)
        # A second answer delta (and a duplicate completion notice) must not
        # print the summary again.
        before = out.getvalue().count("Thought for")
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": " More."}})
        renderer.conclude("completed")
        renderer.conclude("completed")
        self.assertEqual(out.getvalue().count("Thought for"), before)

    def test_card_is_gone_once_the_answer_starts(self):
        renderer = self._renderer()
        renderer._model = "deepresearch-pro"
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "analysis " * 40}})
        self.assertTrue(renderer._think_card_lines(80))
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Answer."}})
        self.assertEqual(renderer._think_card_lines(80), [])

    def test_new_turn_resets_the_episode(self):
        renderer = self._renderer()
        renderer._model = "deepresearch-pro"
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "old turn thinking " * 20}})
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Answer."}})
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        self.assertEqual(renderer._reasoning_buffer, "")
        self.assertIsNone(renderer._reasoning_start)
        self.assertIsNone(renderer._thought_seconds)
        self.assertFalse(renderer._thought_line_printed)
        self.assertEqual(renderer._think_card_lines(80), [])

    def test_conclude_mid_thinking_closes_the_card_without_a_summary(self):
        renderer = self._renderer()
        out = StringIO()
        renderer.console = Console(file=out, no_color=True, width=120)
        renderer._model = "deepresearch-pro"
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "thinking " * 40}})
        renderer.conclude("cancelled")
        # No frozen duration existed (the turn never produced answer text),
        # so no durable summary line is printed -- but the buffer is cleared.
        self.assertEqual(renderer._reasoning_buffer, "")
        self.assertNotIn("Thought for", out.getvalue())


class ComposerThinkCardTests(_RendererStateMixin, unittest.TestCase):
    def _message_text(self, listener) -> str:
        return "".join(
            fragment[1] for fragment in listener._composer_message().__pt_formatted_text__()
            if len(fragment) >= 2 and isinstance(fragment[1], str)
        )

    def test_card_appears_in_the_composer_while_thinking(self):
        renderer = StreamRenderer(_console())
        renderer._model = "deepresearch-pro"
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config())
        listener._active = True
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "deciding between two approaches to the resume logic " * 3}})
        message = self._message_text(listener)
        self.assertIn("Thinking", message)
        self.assertIn("resume logic", message)

    def test_card_yields_to_the_message_viewer(self):
        from tamfis_code.message_viewer import VIEWER

        renderer = StreamRenderer(_console())
        renderer._model = "deepresearch-pro"
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config())
        listener._active = True
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "deciding between two approaches " * 6}})
        previous = VIEWER.is_open
        VIEWER.is_open = True
        try:
            message = self._message_text(listener)
        finally:
            VIEWER.is_open = previous
        # The headline may still say "Thinking"; the boxed card itself is
        # what must yield to the viewer.
        self.assertNotIn("╭", message)
        self.assertNotIn("│", message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
