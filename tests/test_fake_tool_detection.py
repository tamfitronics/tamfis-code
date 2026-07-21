"""Direct unit coverage for runner_local.py's three prose-vs-real-action
detectors.

Before this file, all three had only end-to-end/integration-style coverage
(if any) -- test_runner_local.py's test_fake_tool_call_in_text_gets_a_caveat
exercises _looks_like_fake_tool_call indirectly through a full streamed
turn, and _looks_like_narrated_tool_intent / _looks_like_capitulation had
no test anywhere naming them, despite being the guards that stop a weak
model from claiming to have done repository work it never actually did.
A future refactor could silently break any of these three regexes and
nothing would fail. This file tests each function directly and fast,
without needing to drive a full agent-loop round.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamfis_code.runner_local import (
    _looks_like_capitulation,
    _looks_like_fake_tool_call,
    _looks_like_narrated_tool_intent,
    _requests_autonomous_execution,
    _requests_no_confirmation,
)


class FakeToolCallDetectionTests(unittest.TestCase):
    def test_paren_style_fake_call_is_detected(self):
        self.assertTrue(_looks_like_fake_tool_call("Let me run this: read_file('src/app.py')"))

    def test_json_style_fake_call_is_detected(self):
        self.assertTrue(_looks_like_fake_tool_call('{"tool": "list_directory", "argument": {"path": "."}}'))

    def test_json_style_fake_call_with_name_key_is_detected(self):
        self.assertTrue(_looks_like_fake_tool_call('{"name": "search_code", "arguments": {"query": "TODO"}}'))

    def test_all_registered_tool_names_are_covered(self):
        for name in (
            "read_file", "write_file", "edit_file", "extract_archive", "repackage_archive",
            "list_directory", "search_code", "execute_command", "get_git_info", "browser", "web_search",
        ):
            with self.subTest(name=name):
                self.assertTrue(_looks_like_fake_tool_call(f"{name}(path='.')"))

    def test_plain_prose_mentioning_a_tool_name_without_parens_is_not_flagged(self):
        self.assertFalse(_looks_like_fake_tool_call("I already used read_file to inspect this earlier."))

    def test_empty_text_is_not_flagged(self):
        self.assertFalse(_looks_like_fake_tool_call(""))

    def test_unregistered_function_looking_text_is_not_flagged(self):
        self.assertFalse(_looks_like_fake_tool_call("def calculate_total(items): return sum(items)"))


class NarratedToolIntentDetectionTests(unittest.TestCase):
    def test_let_me_check_is_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent("Let me check the config file for the setting."))

    def test_now_let_me_examine_is_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent("Now let me examine the failing test."))

    def test_i_will_run_is_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent("I will run the test suite to confirm."))

    def test_ill_look_through_is_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent("I'll look through the repository for the bug."))

    def test_i_am_going_to_inspect_is_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent("I am going to inspect the router module."))

    def test_future_registered_tool_dispatch_is_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent(
            "I will call the appropriate registered tool to inspect the full stack."
        ))

    def test_first_i_will_search_is_detected(self):
        self.assertTrue(_looks_like_narrated_tool_intent("First, I'll search for the failing import."))

    def test_past_tense_report_is_not_flagged(self):
        self.assertFalse(_looks_like_narrated_tool_intent("I checked the config file and found the bug on line 12."))

    def test_plain_answer_with_no_promise_is_not_flagged(self):
        self.assertFalse(_looks_like_narrated_tool_intent("The bug was a missing null check in validate()."))

    def test_try_to_install_is_detected(self):
        # Live-reproduced: meta/llama-3.1-70b-instruct on NVIDIA NIM said
        # exactly this, with zero tool calls, and it matched neither the
        # original (inspection-only) verb list nor _looks_like_capitulation
        # -- fell straight through to a "completed" answer with validator
        # caveats. "install" plus the "try to" prefix are the two gaps that
        # let it slip past.
        self.assertTrue(_looks_like_narrated_tool_intent("I will try to install the Python debugger pdb."))

    def test_mutation_verbs_are_detected(self):
        for phrase in (
            "I'll fix the null check in validate().",
            "I will create a new config file for this.",
            "I'll add the missing import at the top.",
            "I will update the README with the new flag.",
            "I'll implement the retry logic now.",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(_looks_like_narrated_tool_intent(phrase))

    def test_past_tense_fix_report_is_not_flagged(self):
        self.assertFalse(_looks_like_narrated_tool_intent("I fixed the bug on line 12 by adding a null check."))

    def test_empty_text_is_not_flagged(self):
        self.assertFalse(_looks_like_narrated_tool_intent(""))


class CapitulationDetectionTests(unittest.TestCase):
    def test_lack_of_clear_next_step_is_detected(self):
        self.assertTrue(_looks_like_capitulation("The task is stuck due to the lack of a clear next step."))

    def test_not_sure_what_to_fix_is_detected(self):
        self.assertTrue(_looks_like_capitulation("I'm not sure what to fix without more context."))

    def test_please_clarify_is_detected(self):
        self.assertTrue(_looks_like_capitulation("Could you clarify which file has the issue?"))

    def test_need_more_information_is_detected(self):
        self.assertTrue(_looks_like_capitulation("I need more information before I can proceed."))

    def test_what_would_you_like_me_to_fix_is_detected(self):
        self.assertTrue(_looks_like_capitulation("What would you like me to fix specifically?"))

    def test_normal_completed_answer_is_not_flagged(self):
        self.assertFalse(_looks_like_capitulation("Fixed the off-by-one error in the loop on line 42."))

    def test_empty_text_is_not_flagged(self):
        self.assertFalse(_looks_like_capitulation(""))


if __name__ == "__main__":
    unittest.main()


class AutonomousResumeDirectiveTests(unittest.TestCase):
    def test_continue_until_fixed_is_authoritative(self):
        self.assertTrue(_requests_autonomous_execution(
            "continue until you fix everything dont ask me for confirmation just go ahead"
        ))

    def test_plain_continue_does_not_overclaim_fix_everything(self):
        self.assertFalse(_requests_autonomous_execution("continue from where you stopped"))

    def test_no_confirmation_wording_is_detected(self):
        for text in (
            "don't ask me for confirmation",
            "dont ask for approval",
            "continue without asking permission",
            "just go ahead",
        ):
            with self.subTest(text=text):
                self.assertTrue(_requests_no_confirmation(text))
