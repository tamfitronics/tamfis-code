"""The running status names what the agent is ACTUALLY doing, not a decorative word.

Owner request 2026-09-20: "the Execution Card feedback of current AI Assistant Activity
should be real and not generic". The headline used to rotate three invented verbs per
phase every 2 seconds ("Razzmatazzing", "Rummaging", "Wiring", "Untangling"), while the
real detail (_status_detail) was computed and never shown.
"""
import unittest
from unittest.mock import patch

from tamfis_code.render import StreamRenderer, _PHASE_ACTIVITY, _shorten_middle

from test_live_input import _console

DECORATIVE = ("Razzmatazzing", "Smoothing", "Rummaging", "Wiring", "Polishing", "Untangling",
              "Mending", "Sequencing", "Plotting", "Tracing", "Calibrating", "Orienting", "Holding")

PLAN = [
    {"step": "List directory contents", "status": "completed"},
    {"step": "Read package.json manifest", "status": "in_progress"},
    {"step": "Read setup.cfg and setup.py manifests", "status": "pending"},
]


def _renderer():
    renderer = StreamRenderer(_console())
    renderer._model = "TamfisGPT-Ultra"
    return renderer


def _tool(renderer, name, **arguments):
    renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": name, "arguments": arguments}})


class RealActivityTests(unittest.TestCase):
    def test_a_tool_in_flight_is_named_with_its_target(self):
        renderer = _renderer()
        _tool(renderer, "read_file", path="/home/betpredict/package.json")
        activity = renderer.current_activity()
        self.assertIn("Reading", activity)
        self.assertIn("/home/betpredict/package.json", activity)
        headline = renderer.live_input_headline("⠋")
        self.assertIn("Reading", headline)
        self.assertIn("package.json", headline)

    def test_different_tools_give_different_activities(self):
        renderer = _renderer()
        _tool(renderer, "search_code", pattern="RouteLabeler")
        self.assertIn("Searching", renderer.current_activity())
        self.assertIn("RouteLabeler", renderer.current_activity())
        _tool(renderer, "edit_file", path="tamfis_code/render.py")
        self.assertIn("Editing", renderer.current_activity())
        self.assertIn("render.py", renderer.current_activity())

    def test_a_running_command_is_shown(self):
        renderer = _renderer()
        _tool(renderer, "execute_command", command="python3 -m pytest tests/test_routing.py -q")
        self.assertEqual(renderer.current_activity(), "Running python3 -m pytest tests/test_routing.py -q")
        # The composer shows the command on its own line, so its headline does not repeat it.
        self.assertEqual(renderer.current_activity(include_command=False), "Running command")

    def test_gateway_activity_detail_survives_remote_event_normalization(self):
        renderer = _renderer()
        renderer.handle_event({
            "event_type": "tool_call_requested",
            "payload": {
                "name": "extract_document",
                "sub_status": "Extracting text from: quarterly-report.pdf",
            },
        })
        self.assertEqual(renderer.current_activity(), "Extracting text from: quarterly-report.pdf")
        renderer.handle_event({
            "event_type": "tool_output",
            "payload": {
                "tool": "extract_document",
                "sub_status": "Extracted text from: quarterly-report.pdf",
                "content": "ok",
            },
        })
        self.assertEqual(renderer.current_activity(), "Extracted text from: quarterly-report.pdf")

    def test_gateway_activity_detail_is_bounded_and_strips_controls(self):
        renderer = _renderer()
        renderer.handle_event({
            "event_type": "tool_call_requested",
            "payload": {"name": "read_file", "sub_status": "Reading\n" + "x" * 200},
        })
        activity = renderer.current_activity()
        self.assertNotIn("\n", activity)
        self.assertLessEqual(len(activity), 120)

    def test_the_headline_fits_a_narrow_terminal_without_cutting_the_timing(self):
        renderer = _renderer()
        _tool(renderer, "read_file", path="/home/user/projects/very/deep/nested/structure/of/folders/package.json")
        headline = renderer.live_input_headline("⠋", width=50)
        self.assertLessEqual(len(headline), 50, headline)
        self.assertRegex(headline, r"\(\d+s(?: · [^)]+)?\)$")
        self.assertIn("Reading", headline)

    def test_when_only_waiting_on_the_model_the_plan_step_is_shown(self):
        renderer = _renderer()
        renderer._plan_steps = PLAN
        renderer.handle_event({"event_type": "provider_request_started", "payload": {}})
        self.assertEqual(renderer.current_activity(), "Step 2/3 · Read package.json manifest")

    def test_a_specific_tool_beats_the_plan_step(self):
        renderer = _renderer()
        renderer._plan_steps = PLAN
        _tool(renderer, "read_file", path="package.json")
        self.assertIn("Reading", renderer.current_activity())

    def test_approval_is_stated(self):
        renderer = _renderer()
        renderer.handle_event({"event_type": "approval_required", "payload": {}})
        self.assertEqual(renderer.current_activity(), "Waiting for your approval")

    def test_a_file_change_names_the_file(self):
        renderer = _renderer()
        renderer.handle_event({"event_type": "file_mutation", "payload": {"path": "src/app.py"}})
        self.assertIn("src/app.py", renderer.current_activity())

    def test_the_headline_does_not_rotate_through_decorative_words(self):
        renderer = _renderer()
        renderer._phase = "plan"
        seen = set()
        for offset in range(0, 40, 2):  # every 2s -- the old rotation period
            renderer._task_start -= 2
            seen.add(renderer.live_input_headline("⠋").split("(")[0])
        self.assertEqual(len(seen), 1, seen)
        self.assertFalse(any(word in "".join(seen) for word in DECORATIVE))

    def test_no_phase_label_is_decorative_or_generic_filler(self):
        for phase, label in _PHASE_ACTIVITY.items():
            self.assertFalse(any(word in label for word in DECORATIVE), (phase, label))
            self.assertNotEqual(label.strip().lower(), "working", phase)

    def test_every_phase_has_a_specific_label(self):
        for phase in ("understand", "inspect", "route", "reasoning", "respond", "plan", "execute",
                      "observe", "repair", "waiting_for_approval", "validate", "report"):
            renderer = _renderer()
            renderer._phase = phase
            renderer._status_detail = "Preparing the task"
            self.assertEqual(renderer.current_activity(), _PHASE_ACTIVITY[phase])

    def test_a_long_path_is_cut_in_the_middle_keeping_the_file_name(self):
        renderer = _renderer()
        _tool(renderer, "read_file", path="/home/user/projects/very/deep/nested/structure/of/folders/package.json")
        activity = renderer.current_activity()
        self.assertLessEqual(len(activity), 73)
        self.assertTrue(activity.startswith("Reading"))
        self.assertTrue(activity.endswith("package.json"), activity)
        self.assertEqual(_shorten_middle("short", 20), "short")

    def test_the_footer_status_uses_the_same_real_activity(self):
        renderer = _renderer()
        _tool(renderer, "read_file", path="package.json")
        self.assertIn("Reading", renderer.live_input_status("⠋"))

    def test_the_rich_spinner_path_uses_it_too(self):
        renderer = _renderer()
        _tool(renderer, "read_file", path="package.json")
        renderer._build_status()  # updates the Rich spinner's text in place
        self.assertIn("Reading", str(renderer._spinner.text))

    def test_the_tool_result_returns_to_the_plan_step_not_a_stale_tool(self):
        renderer = _renderer()
        renderer._plan_steps = PLAN
        _tool(renderer, "read_file", path="package.json")
        renderer.handle_event({"event_type": "tool_output", "payload": {"content": "ok"}})
        self.assertEqual(renderer.current_activity(), "Step 2/3 · Read package.json manifest")


if __name__ == "__main__":
    unittest.main()
