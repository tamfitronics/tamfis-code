"""Behavioural regressions for the 2026-09-24 session failures ("Fix
Workspace Canvas Streaming", session 1380884522) and the think-card toggle:

A. Assistant prose must not become/recover as the user objective.
C. "The next step would be to inspect..." is future work, not completion.
E. Farewell + handoff cancels without a confirmation question; quoted/code
   occurrences of the same words do not.
F. Route note and "ready" never concatenate into "failoversready".
Plus: degenerate repeated flags collapse; show_think_card=false silences
the card and the durable thought line.
"""
import re
import shutil
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.live_input import LiveInputListener, classify_disengagement, idle_bottom_toolbar
from tamfis_code.render import StreamRenderer
from tamfis_code.runner_local import (
    _checkpoint_resume_objective,
    _collapse_repeated_flags,
    _looks_like_narrated_tool_intent,
)


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200, force_terminal=False)


def _config() -> Config:
    cfg = Config.__new__(Config)
    cfg.approval_policy = "ask"
    return cfg


class _StatePatchMixin(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


class ObjectiveProvenanceTests(unittest.TestCase):
    """Failure A: the stored objective was an assistant progress statement."""

    CORRUPT_CHECKPOINT = {
        "objective": (
            "The existing code confirms the architecture is split across "
            "three intent systems and the streaming path is coherent."
        ),
        "messages": [
            {"role": "user", "content": (
                "Fix Tamfis-Code's orchestration and execution codebase so it "
                "behaves as a dependable autonomous coding agent."
            )},
            {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "list_directory", "arguments": "{\"path\": \"/home\"}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "[]"},
            {"role": "assistant", "content": (
                "The existing code confirms the architecture is split across "
                "three intent systems and the streaming path is coherent."
            )},
        ],
    }

    def test_assistant_progress_statement_never_recovers_as_objective(self):
        recovered = _checkpoint_resume_objective(self.CORRUPT_CHECKPOINT)
        self.assertIn("Fix Tamfis-Code", recovered)
        self.assertNotIn("existing code confirms", recovered)

    def test_corrupted_objective_without_user_evidence_recovers_empty(self):
        checkpoint = {
            "objective": "assistant prose only",
            "messages": [{"role": "assistant", "content": "assistant prose only"}],
        }
        self.assertEqual(_checkpoint_resume_objective(checkpoint), "")

    def test_legitimate_user_chain_still_recovers_in_full(self):
        checkpoint = {
            "objective": "Fix the streaming bug",
            "messages": [
                {"role": "user", "content": "Fix the streaming bug"},
                {"role": "assistant", "content": "On it", "tool_calls": [{"id": "t1"}]},
                {"role": "user", "content": "also check the renderer"},
            ],
        }
        recovered = _checkpoint_resume_objective(checkpoint)
        self.assertIn("Fix the streaming bug", recovered)
        self.assertIn("also check the renderer", recovered)

    def test_stored_objective_matching_a_user_message_still_anchors(self):
        # The stored text IS a user message: the normal anchored path runs.
        checkpoint = {
            "objective": "please continue from the saved checkpoint",
            "messages": [
                {"role": "user", "content": "Fix the login flow"},
                {"role": "assistant", "content": "still working", "tool_calls": [{"id": "t1"}]},
                {"role": "user", "content": "please continue from the saved checkpoint"},
            ],
        }
        recovered = _checkpoint_resume_objective(checkpoint)
        self.assertIn("Fix the login flow", recovered)


class PrematureTerminationTests(unittest.TestCase):
    """Failure C: the agent listed a directory, said "The next step would be
    to inspect the context..." and stopped despite an unfinished plan."""

    OBSERVED_ROADMAP = (
        "I've listed the directory and searched for 'dump'. Here are the files: "
        "a.py, b.py. The next step would be to inspect the context assembly path."
    )

    def test_future_step_roadmap_is_a_narrated_intent(self):
        self.assertTrue(_looks_like_narrated_tool_intent(self.OBSERVED_ROADMAP))

    def test_completed_report_is_not_misflagged(self):
        report = (
            "The fix is complete and tests pass. The next step in the original "
            "plan was completed earlier."
        )
        self.assertFalse(_looks_like_narrated_tool_intent(report))

    def test_first_person_promise_still_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent("Let me inspect the checkpoint next."))


class RepeatedFlagCollapseTests(unittest.TestCase):
    """The `wp --no-plugins` x15 rendering garbage."""

    def test_degenerate_flag_repetition_collapses(self):
        command = (
            "wp --allow-root --path=/home/finima/www --debug --quiet "
            + " ".join(["--no-plugins"] * 15)
        )
        collapsed = _collapse_repeated_flags(command)
        self.assertEqual(collapsed.count("--no-plugins"), 1)
        self.assertIn("--allow-root", collapsed)
        self.assertIn("--path=/home/finima/www", collapsed)

    def test_repeated_value_options_survive(self):
        command = "git log --exclude=a --exclude=b --grep=x --grep=y --grep=z"
        self.assertEqual(_collapse_repeated_flags(command), command)

    def test_normal_commands_untouched(self):
        command = "rg -n 'pattern' src/ --type py"
        self.assertEqual(_collapse_repeated_flags(command), command)


class DisengagementTests(unittest.TestCase):
    """Failure E: farewell + handoff cancelled cleanly, no confirmation."""

    OBSERVED_GOODBYE = (
        "Since you are being cocky, I have delegated the task to Codex to fix "
        "it for me. Bye."
    )

    def test_observed_goodbye_is_disengagement(self):
        self.assertTrue(classify_disengagement(self.OBSERVED_GOODBYE))

    def test_bare_farewell_is_disengagement(self):
        self.assertTrue(classify_disengagement("bye"))
        self.assertTrue(classify_disengagement("Goodbye."))

    def test_quoted_or_code_occurrences_do_not_trigger(self):
        self.assertFalse(classify_disengagement('print("bye")'))
        self.assertFalse(classify_disengagement("```python\n# bye\n```"))

    def test_questions_and_imperative_requests_are_real_work(self):
        self.assertFalse(classify_disengagement("Should I delegate this to Codex?"))
        self.assertFalse(classify_disengagement("Please delegate this task to Codex for me"))
        self.assertFalse(classify_disengagement("I will delegate this to Codex, bye"))

    def test_past_handoff_without_named_agent_counts(self):
        self.assertTrue(classify_disengagement("I have handed it over to another tool. Goodbye"))

    def test_live_listener_cancels_without_confirmation_question(self):
        renderer = StreamRenderer(_console())
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config())
        listener._active = True
        interrupt_calls = []
        listener._interrupt_callback = lambda classification: interrupt_calls.append(classification)
        listener._enqueue(self.OBSERVED_GOODBYE)
        self.assertEqual(interrupt_calls, ["cancel"])
        self.assertIsNotNone(listener._interrupt_classification)

    def test_listener_emits_stop_acknowledgement_only(self):
        renderer = StreamRenderer(_console())
        out = StringIO()
        renderer.console = Console(file=out, no_color=True, width=120)
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config())
        listener._active = True
        listener._enqueue(self.OBSERVED_GOODBYE)
        rendered = out.getvalue()
        self.assertNotIn("Are you sure", rendered)
        self.assertNotIn("delegat", rendered.lower())
        self.assertIn("checkpointed", rendered)


class FooterRouteNoteTests(unittest.TestCase):
    """Failure F: 'rerouted · 2 failoversready' concatenation."""

    def test_route_note_and_ready_never_concatenate(self):
        session_id = 424242
        from tamfis_code import state as local_state

        local_state.record_route_event(
            session_id,
            provider="tamfisgpt",
            model="auto",
            previous_provider="tamfisgpt",
            previous_model="auto",
            reason="credits",
            kind="failover",
        )
        toolbar = idle_bottom_toolbar(_config(), session_id, model="auto")
        plain = "".join(
            text for _style, text in toolbar.__pt_formatted_text__() if isinstance(text, str)
        )
        stripped = re.sub(r"<[^>]+>", "", plain)
        self.assertNotIn("failoversready", stripped)
        self.assertRegex(stripped, r"failovers?\b")
        self.assertIn("ready", stripped)
        # The separator between the route note and the status survived.
        self.assertRegex(stripped, r"failovers?\s*·\s*ready")


class ThinkCardToggleTests(_StatePatchMixin, unittest.TestCase):
    def test_disabled_card_hides_live_card_and_summary(self):
        from tamfis_code.config import load_config

        with patch("tamfis_code.render.load_config") as loader:
            cfg = Config()
            cfg.show_think_card = False
            loader.return_value = cfg
            renderer = StreamRenderer(_console())
        out = StringIO()
        renderer.console = Console(file=out, no_color=True, width=120)
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "analysis " * 40}})
        self.assertEqual(renderer._think_card_lines(80), [])
        renderer._reasoning_start = renderer._reasoning_start - 12.0
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Answer."}})
        self.assertNotIn("Thought for", out.getvalue())

    def test_default_config_shows_the_card(self):
        renderer = StreamRenderer(_console())
        self.assertTrue(renderer._show_think_card)
        renderer.handle_event({"event_type": "task_started", "payload": {}})
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "analysis " * 40}})
        self.assertTrue(renderer._think_card_lines(80))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
