"""Regression tests for a session's stable display title.

The mechanical (first-N-words) title generator was REMOVED entirely: the
live-reported symptom was sessions titled like "Fist you need to fix" --
an LLM title attempt failing silently and a truncated-objective label
standing in its place, indistinguishable from "first few words".

The contract now: titles come ONLY from the LLM
(state.upgrade_session_title_with_ai via the providers system). A failed
attempt leaves NO persisted title -- display falls back to a transient
activity snapshot -- and is retried on the next turn. Provenance
(title_source/title_model/title_generated_at/title_fallback_reason) is
persisted and inspectable via session_title_diagnostics(). The user can
rename (title_source="user", never auto-overwritten) or ask for a fresh
LLM title at any time (/regenerate-title).
"""
import asyncio
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tamfis_code import state as state_module


class _StateDirFixture:
    def setUp(self):
        self._originals = (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
        )
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        # _LOCK_PATH is bound from CONFIG_DIR at import time, so overriding
        # CONFIG_DIR alone leaves the lock pointing at the real user config
        # dir -- under a read-only sandbox every test then emits the
        # "could not acquire the session-state lock" warning.
        state_module._LOCK_PATH = base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None

    def tearDown(self):
        (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
        ) = self._originals
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self._tmp.cleanup()


def _fake_providers(content_or_exc, *, provider_name="nvidia"):
    """Stand in for tamfis_code.providers.ProviderManager inside
    _generate_session_title: the title path now goes through the manager's
    own chat_completion(AUTO) machinery, so the fake replaces that method
    (an async generator yielding `content_or_exc`, or raising it).
    Returns (patcher, create_calls) -- the caller must invoke the patcher's
    stop(); create_calls records each chat_completion kwargs dict."""
    create_calls = []

    async def fake_chat_completion(provider, messages, **kwargs):
        create_calls.append({"provider": provider, **kwargs})
        if isinstance(content_or_exc, Exception):
            raise content_or_exc
        for piece in (content_or_exc[i:i + 12] for i in range(0, len(content_or_exc), 12)):
            yield piece

    manager = MagicMock()
    manager.chat_completion = fake_chat_completion

    patcher = patch("tamfis_code.providers.ProviderManager", return_value=manager)
    return patcher, create_calls


class MechanicalTitleGeneratorIsGoneTests(unittest.TestCase):
    def test_the_first_words_generator_no_longer_exists(self):
        """The root cause of "still just taking the first few words": a
        synchronous mechanical generator stood behind every LLM failure.
        It must not exist at all anymore."""
        self.assertFalse(hasattr(state_module, "_derive_session_title"))
        self.assertFalse(hasattr(state_module, "_TITLE_FILLER_WORDS"))

    def test_ensure_session_title_never_persists_a_mechanical_title(self):
        state = state_module.SessionState(session_id=1)
        state_module.put_session_state(state)


class EnsureSessionTitleTests(_StateDirFixture, unittest.TestCase):
    def test_no_objective_no_title(self):
        """A first-N-words title must NEVER be persisted -- with the
        mechanical generator removed, ensure_session_title is a no-op title
        writer (the LLM path is the only title source)."""
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        self.assertEqual(state_module.get_session_state(1).session_title, "")

    def test_blank_objective_is_still_a_no_op(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "   ")
        self.assertEqual(state_module.get_session_state(1).session_title, "")


class SubstantiveObjectiveTests(unittest.TestCase):
    def test_a_bare_continue_is_not_substantive(self):
        self.assertFalse(state_module._is_substantive_objective("Continue"))

    def test_short_but_real_task_words_are_substantive(self):
        """Semantic content, not character count: 'fix login bug' is short
        but titles a session fine."""
        self.assertTrue(state_module._is_substantive_objective("Fix login bug"))

    def test_pure_filler_is_not_substantive_no_matter_the_length(self):
        self.assertFalse(
            state_module._is_substantive_objective("please please please help me please help")
        )


class UpgradeSessionTitleWithAiTests(_StateDirFixture, unittest.TestCase):
    """Titles come ONLY from the LLM (providers system, main + fallback).
    Never raises, never blocks completion; failures leave NO title (not a
    mechanical stand-in) and are retried on a later turn."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _upgrade(self, content_or_exc, objective="Fix the flaky auth test", session_id=1):
        patcher, calls = _fake_providers(content_or_exc)
        with patcher:
            self._run(state_module.upgrade_session_title_with_ai(session_id, objective))
        patcher.stop()
        return calls

    def test_semantic_llm_title_replaces_nothing_else(self):
        """Acceptance TEST 1: an investigation-style objective gets a
        semantic title, never a first-N-words truncation."""
        state_module.save_session_state(1, workspace_root="/a")
        objective = (
            "Please investigate why image and video workspace generation keeps "
            "failing after routing changes."
        )
        self._upgrade("Fix Image & Video Workspace", objective=objective)
        title = state_module.get_session_state(1).session_title
        self.assertEqual(title, "Fix Image & Video Workspace")
        self.assertNotEqual(title, objective[:len(title)])  # not a truncation

    def test_provenance_is_persisted_on_success(self):
        state_module.save_session_state(1, workspace_root="/a")
        self._upgrade("Fix flaky auth test")
        state = state_module.get_session_state(1)
        self.assertEqual(state.title_source, "llm")
        # model attribution names the route that ACTUALLY answered (the first
        # preferred route here), not a generic "auto": "which model named
        # this session" is the first question a bad title raises.
        self.assertEqual(state.title_model, state_module.title_route_preference()[0])
        self.assertTrue(state.title_generated_at)
        self.assertIsNone(state.title_fallback_reason)
        self.assertTrue(state.ai_title_attempted)

    def test_llm_call_uses_a_generous_token_budget_not_a_starving_20(self):
        state_module.save_session_state(1, workspace_root="/a")
        calls = self._upgrade("Fix flaky auth test")
        self.assertTrue(calls)
        self.assertGreaterEqual(calls[0].get("max_tokens", 0), 100)
    def test_rejects_a_response_that_answers_the_objective_instead_of_titling_it(self):
        """An answer-shaped response must be rejected -- and with the
        mechanical generator gone, rejection now means NO title, never a
        truncated-objective stand-in."""
        state_module.save_session_state(1, workspace_root="/a")
        objective = "What does a semicolon do in Python?"
        self._upgrade(
            "In Python, a semicolon is used to separate multiple statements "
            "written on a single line, though it is rarely used because "
            "newlines already terminate statements.",
            objective=objective,
        )
        self.assertEqual(state_module.get_session_state(1).session_title, "")
        self.assertEqual(
            state_module.get_session_state(1).title_fallback_reason, "invalid_response",
        )

    def test_rejects_a_response_ending_in_terminal_punctuation_even_if_short(self):
        state_module.save_session_state(1, workspace_root="/a")
        self._upgrade("It fixes the test.")
        self.assertEqual(state_module.get_session_state(1).session_title, "")

    def test_accepts_a_genuinely_short_compliant_title(self):
        state_module.save_session_state(1, workspace_root="/a")
        self._upgrade("Investigate retry loop timeout",
                      objective="Investigate the timeout bug in the retry loop")
        self.assertEqual(
            state_module.get_session_state(1).session_title, "Investigate Retry Loop Timeout",
        )

    def test_provider_failure_leaves_no_title_and_frees_the_budget(self):
        """Acceptance TEST 5: a forced LLM failure leaves the session
        working, records WHY, and retries on the next turn."""
        state_module.save_session_state(1, workspace_root="/a")
        self._upgrade(ConnectionError("boom"))
        state = state_module.get_session_state(1)
        self.assertEqual(state.session_title, "")
        self.assertEqual(state.title_fallback_reason, "provider_error")
        self.assertFalse(state.ai_title_attempted)  # next turn retries

        self._upgrade("Fix flaky auth test")
        # The accepted title is tightened (Title Case) before it is persisted.
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix Flaky Auth Test")

    def test_blank_objective_is_a_no_op(self):
        state_module.save_session_state(1, workspace_root="/a")
        self._run(state_module.upgrade_session_title_with_ai(1, "   "))
        self.assertEqual(state_module.get_session_state(1).session_title, "")

    def test_a_successful_upgrade_is_never_retried_on_a_later_turn(self):
        state_module.save_session_state(1, workspace_root="/a")
        self._upgrade("Fix flaky auth test")
        patcher, _calls = _fake_providers("Unrelated later message")
        with patcher as factory:
            self._run(state_module.upgrade_session_title_with_ai(1, "what's the weather like"))
            factory.assert_not_called()
        patcher.stop()
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix Flaky Auth Test")

    def test_a_nonsubstantive_later_turn_does_not_retitle_the_session(self):
        """Acceptance TEST 9: 'Continue' must not produce a useless title,
        and must not retitle an already-titled session."""
        state_module.save_session_state(1, workspace_root="/a")
        self._upgrade("Fix flaky auth test")
        self._upgrade("Fix the auth retry storm", objective="Continue")
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix Flaky Auth Test")

    def test_user_rename_is_never_overwritten_by_the_llm(self):
        """Acceptance TEST 8: an explicit user title survives every later
        automatic generation attempt."""
        state_module.save_session_state(1, workspace_root="/a")
        self.assertTrue(state_module.rename_session_title(1, "My Own Name"))
        self._upgrade("LLM tries to retitle")
        state = state_module.get_session_state(1)
        self.assertEqual(state.session_title, "My Own Name")
        self.assertEqual(state.title_source, "user")

    def test_provider_failure_is_diagnosed_not_silently_swallowed(self):
        import contextlib
        import io
        state_module.save_session_state(1, workspace_root="/a")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            calls = self._upgrade(ConnectionError("boom"))
        self.assertIn("[title]", err.getvalue())
        # One route raising must not abort the chain: every preferred route
        # is still attempted, each attempt named, and the last reason kept.
        self.assertIn("failed (ConnectionError: boom)", err.getvalue())
        self.assertIn("trying next preferred route", err.getvalue())
        self.assertEqual(len(calls), state_module._TITLE_MAX_MACHINERY_ATTEMPTS)
        self.assertEqual(
            state_module.get_session_state(1).title_fallback_reason, "provider_error",
        )


class DisplayFallbackTests(_StateDirFixture, unittest.TestCase):
    def test_display_falls_back_to_a_transient_activity_label_not_a_title(self):
        """Before the LLM title lands, the resume picker shows a live
        activity snapshot -- never persisted as session_title."""
        state_module.save_session_state(
            1, workspace_root="/a",
            active_task={"objective": "Fix the flaky auth test", "task_id": "t1"},
        )
        self.assertEqual(state_module.get_session_state(1).session_title, "")
        self.assertEqual(
            state_module.session_display_title(1), "Fix the flaky auth test",
        )
        # ...and the fallback is transient: still no persisted title.
        self.assertEqual(state_module.get_session_state(1).session_title, "")

    def test_unknown_session_gets_the_generic_label(self):
        self.assertEqual(state_module.session_display_title(999), "Session 999")


class RenameAndRegenerateTests(_StateDirFixture, unittest.TestCase):
    def test_rename_sets_a_user_sourced_title(self):
        state_module.save_session_state(1, workspace_root="/a")
        self.assertTrue(state_module.rename_session_title(1, "Spreadsheet Engineering Skill"))
        state = state_module.get_session_state(1)
        self.assertEqual(state.session_title, "Spreadsheet Engineering Skill")
        self.assertEqual(state.title_source, "user")

    def test_regeneration_clears_the_title_for_a_fresh_llm_pass(self):
        """Acceptance TEST 10's core: /regenerate-title clears a stale
        title so the next upgrade generates a fresh one."""
        state_module.save_session_state(1, workspace_root="/a")
        patcher, _ = _fake_providers("Fix Flaky Auth Test")
        with patcher:
            asyncio.run(state_module.upgrade_session_title_with_ai(1, "Fix the flaky auth test"))
        patcher.stop()
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix Flaky Auth Test")

        self.assertTrue(state_module.request_session_title_regeneration(1))
        state = state_module.get_session_state(1)
        self.assertEqual(state.session_title, "")
        self.assertFalse(state.ai_title_attempted)

        patcher, _ = _fake_providers("Add Streaming Retry Budget")
        with patcher:
            asyncio.run(state_module.upgrade_session_title_with_ai(1, "Add a streaming retry budget"))
        patcher.stop()
        self.assertEqual(
            state_module.get_session_state(1).session_title, "Add Streaming Retry Budget",
        )

    def test_regeneration_refuses_to_touch_a_user_title(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.rename_session_title(1, "User's Own Name")
        self.assertFalse(state_module.request_session_title_regeneration(1))
        self.assertEqual(state_module.get_session_state(1).session_title, "User's Own Name")

    def test_diagnostics_expose_the_full_provenance_record(self):
        state_module.save_session_state(1, workspace_root="/a")
        patcher, _ = _fake_providers("Fix flaky auth test")
        with patcher:
            asyncio.run(state_module.upgrade_session_title_with_ai(1, "Fix the flaky auth test"))
        patcher.stop()
        diag = state_module.session_title_diagnostics(1)
        self.assertEqual(diag["title"], "Fix Flaky Auth Test")
        self.assertEqual(diag["title_source"], "llm")
        self.assertEqual(diag["title_model"], state_module.title_route_preference()[0])
        self.assertTrue(diag["title_generated_at"])
        self.assertIsNone(diag["fallback_reason"])


# --------------------------------------------------------------------------
# Tight titling: what a title is allowed to be (live-reported: /regenerate
# "did nothing", and titles that were still echoes of the request)
# --------------------------------------------------------------------------


class TitleValidationTests(unittest.TestCase):
    """The acceptance table from the titling contract: the GOOD examples must
    be accepted, the BAD (first-words / truncation / filler) must not."""

    CASES = (
        (
            "Please investigate why image and video workspace generation keeps "
            "throwing errors after the routing changes",
            ("Fix Image & Video Workspace", "Image & Video Workspace Generation"),
            ("Please investigate why image and", "Please"),
        ),
        (
            "Fist you need to fix tamfis-code to properly keep track of sessions "
            "so users can use multiple sessions",
            ("Fix Tamfis-Code Sessions", "Fix Session Tracking"),
            ("Fist you need to fix", "Fist"),
        ),
        (
            "Build a production-grade spreadsheet engineering skill with workbook "
            "reconciliation and validation",
            ("Spreadsheet Engineering Skill", "Build Spreadsheet Capability"),
            ("Build a production-grade spreadsheet", "Build a"),
        ),
        (
            "Fix tamfis-code so multiple sessions don't leak into one another and "
            "resume doesn't show duplicates",
            ("Fix Session Isolation", "Fix Session Management"),
            # "Fix tamfis-code" alone is a legitimate verb+system title (the
            # contract's own "Train TamGPT-3.0" is that shape); what must never
            # be accepted is the mid-sentence truncation.
            ("Fix tamfis-code so multiple",),
        ),
        (
            "Please please can you kindly help me to fix streaming because responses "
            "keep repeating",
            ("Fix Streaming Repetition", "Fix Streaming"),
            ("Please please can you", "Please"),
        ),
        (
            "Train TamGPT-3.0 with the new corpus and report loss curves",
            ("Train TamGPT-3.0", "Train TamGPT-3.0 Corpus"),
            (),
        ),
        (
            "Investigate and repair provider weighting so AUTO stops churning routes",
            ("Repair Provider Weighting", "Fix Route Churn"),
            (),
        ),
    )

    def test_good_titles_are_accepted(self):
        for objective, good, _bad in self.CASES:
            for candidate in good:
                accepted, reason = state_module.validate_session_title(candidate, objective=objective)
                self.assertTrue(accepted, f"{candidate!r} rejected for {objective!r}: {reason}")
                self.assertEqual(reason, "")

    def test_first_words_and_truncations_are_rejected(self):
        for objective, _good, bad in self.CASES:
            for candidate in bad:
                accepted, reason = state_module.validate_session_title(candidate, objective=objective)
                self.assertFalse(accepted, f"{candidate!r} was accepted for {objective!r}")
                self.assertTrue(reason, "a rejection must carry a reason for the corrective retry")

    def test_structural_rejections(self):
        objective = "Fix the flaky auth test in the login flow"
        for candidate, fragment in (
            ("", "empty"),
            ("Fix", "single word"),
            ("Fix the flaky auth test in the login flow now please", "sentence"),
            ("Session Task Request", "filler/meta"),
            ("Totally Unrelated Words", "shares no words"),
        ):
            accepted, reason = state_module.validate_session_title(candidate, objective=objective)
            self.assertFalse(accepted, f"{candidate!r} was accepted")
            self.assertIn(fragment, reason)

    def test_a_title_identical_to_the_existing_one_is_rejected(self):
        accepted, reason = state_module.validate_session_title(
            "Fix Flaky Auth Test",
            objective="Fix the flaky auth test in the login flow",
            previous_title="Fix Flaky Auth Test",
        )
        self.assertFalse(accepted)
        self.assertIn("identical", reason)


class TitleCommandAliasTests(unittest.TestCase):
    """Live-reported: `/regenerate title` did nothing. Every natural spelling
    must resolve to the command, and a non-command must not."""

    def _argument(self, text: str):
        from tamfis_code.interactive import _title_command_argument

        return _title_command_argument(text)

    def test_every_alias_resolves_to_regeneration(self):
        for text in (
            "/regenerate-title", "/regenerate-title title", "/regenerate", "/regenerate title",
            "/REGENERATE", "/retitle", "/title", "/title session title", "/regenerate-title  ",
        ):
            self.assertEqual(self._argument(text), "", f"{text!r} did not resolve to regeneration")

    def test_an_argument_is_a_rename_target(self):
        self.assertEqual(self._argument("/regenerate-title Fix Session Isolation"), "Fix Session Isolation")
        self.assertEqual(self._argument('/retitle "My Session"'), "My Session")
        self.assertEqual(self._argument("/title Spreadsheet Skill"), "Spreadsheet Skill")

    def test_non_title_commands_are_not_claimed(self):
        for text in ("/status", "/titlecase", "/regeneration", "regenerate title", "/help"):
            self.assertIsNone(self._argument(text), f"{text!r} was wrongly treated as a title command")


class UnknownSlashCommandGuardTests(unittest.TestCase):
    def test_a_mistyped_command_is_shaped_as_a_command(self):
        from tamfis_code.interactive import _looks_like_unknown_slash_command

        self.assertEqual(_looks_like_unknown_slash_command("/regenrate"), "/regenrate")
        self.assertEqual(
            _looks_like_unknown_slash_command("/regenerat title now"), "/regenerat",
        )

    def test_paths_and_prose_are_never_treated_as_commands(self):
        from tamfis_code.interactive import _looks_like_unknown_slash_command

        for text in (
            "/home/tamfiscode/file.py",
            "/tmp/website-redesign-prompt.docx",
            "no slash here",
            "/2 split ratios",
            "/",
        ):
            self.assertIsNone(_looks_like_unknown_slash_command(text), f"{text!r} was treated as a command")


class TitleSeedTests(unittest.TestCase):
    """The title model must be given what the session is ABOUT, not just the
    latest (possibly trivial) message."""

    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._LOCK_PATH = base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH = self._originals
        state_module._STATE_CACHE = None
        self._tmp.cleanup()

    def test_active_task_objective_wins(self):
        state_module.save_session_state(
            1, workspace_root="/a",
            active_task={"objective": "Fix the image workspace generation"},
        )
        state = state_module.get_session_state(1)
        from tamfis_code.interactive import _title_seed_objective

        self.assertEqual(_title_seed_objective(state), "Fix the image workspace generation")

    def test_falls_back_to_the_last_substantive_turn_then_summary(self):
        from tamfis_code.interactive import _title_seed_objective

        state_module.save_session_state(1, workspace_root="/a")
        state_module.remember_conversation_turn(
            1, objective="Refactor the provider router", answer="Done.",
        )
        self.assertEqual(_title_seed_objective(state_module.get_session_state(1)), "Refactor the provider router")

        state_module.save_session_state(2, workspace_root="/a", conversation_summary="compacted recap text")
        self.assertEqual(_title_seed_objective(state_module.get_session_state(2)), "compacted recap text")

    def test_the_title_prompt_carries_session_context_and_strict_rules(self):
        state_module.save_session_state(1, workspace_root="/a/workspace")
        state_module.remember_conversation_turn(1, objective="Fix image generation", answer="ok")
        messages = state_module.build_session_title_messages(1, "Fix image generation")
        self.assertIn("3 to 6 words", messages[0]["content"])
        self.assertIn("NEVER repeat the opening words", messages[0]["content"])
        self.assertIn("Fix Image & Video Workspace", messages[0]["content"])
        self.assertIn("Primary request: Fix image generation", messages[1]["content"])
        self.assertIn("Workspace: /a/workspace", messages[1]["content"])


class CorrectiveRetryTests(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._LOCK_PATH = base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH = self._originals
        state_module._STATE_CACHE = None
        self._tmp.cleanup()

    def test_a_rejected_title_triggers_one_corrective_retry(self):
        objective = "Please investigate why image and video workspace generation keeps failing"
        candidates = ["Please investigate why image and", "Fix Image Video Workspace"]
        seen: list[list[dict]] = []

        offsets: list[int] = []

        async def fake_generate(messages, route_offset=0):
            seen.append(list(messages))
            offsets.append(route_offset)
            return candidates[len(seen) - 1], "nvidia", ""

        with patch.object(state_module, "_generate_session_title", fake_generate):
            title, model, reason = asyncio.run(
                state_module._generate_and_validate_title(1, objective)
            )
        self.assertEqual(title, "Fix Image Video Workspace")
        self.assertEqual(model, "nvidia")
        self.assertEqual(reason, "")
        self.assertEqual(len(seen), 2, "one retry, not an unbounded negotiation")
        # The retry carries the rejection reason so the model can correct it.
        self.assertIn("rejected", seen[1][-1]["content"])
        # ...and rotates the preferred route, so one weak model that cannot
        # follow the contract isn't the session's only chance at a title.
        self.assertEqual(offsets, [0, 1])

    def test_two_rejections_yield_no_title_rather_than_a_bad_one(self):
        async def fake_generate(messages, route_offset=0):
            return "Please investigate why image and", "nvidia", ""

        with patch.object(state_module, "_generate_session_title", fake_generate):
            title, _model, reason = asyncio.run(
                state_module._generate_and_validate_title(
                    1, "Please investigate why image and video workspace generation keeps failing",
                )
            )
        self.assertEqual(title, "")
        self.assertEqual(reason, "invalid_response")

    def test_a_provider_failure_is_reported_not_masked(self):
        async def fake_generate(messages, route_offset=0):
            return "", "", "provider_timeout"

        with patch.object(state_module, "_generate_session_title", fake_generate):
            title, _model, reason = asyncio.run(
                state_module._generate_and_validate_title(1, "Fix the flaky auth test")
            )
        self.assertEqual(title, "")
        self.assertEqual(reason, "provider_timeout")


class AcceptedTitlesAreTightened(unittest.TestCase):
    """An accepted LLM title gets a deterministic tightening pass, so every
    title reads like a task name: imperative opener, Title Case, "&" between
    the halves of a compound task, product/API names left exactly as written.
    """

    def _title(self, candidate, objective):
        title, reason = state_module.validate_session_title(
            candidate, objective=objective,
        )
        self.assertEqual(reason, "")
        return title

    def test_a_gerund_opener_becomes_the_imperative(self):
        self.assertEqual(
            self._title(
                "Investigating image video workspace errors",
                "Please investigate why image and video workspace generation keeps throwing errors",
            ),
            "Investigate Image Video Workspace Errors",
        )

    def test_a_compound_subject_uses_the_ampersand_form(self):
        # The reference title for this exact request is "Fix Image & Video
        # Workspace" -- the "and" form is the loose version of it.
        self.assertEqual(
            self._title(
                "Investigating image and video workspace errors",
                "Please investigate why image and video workspace generation keeps throwing errors",
            ),
            "Investigate Image & Video Workspace Errors",
        )

    def test_a_lowercase_candidate_is_title_cased(self):
        self.assertEqual(
            self._title(
                "fix streaming repetition",
                "Please help me fix streaming because responses keep repeating",
            ),
            "Fix Streaming Repetition",
        )

    def test_product_and_version_names_keep_their_exact_form(self):
        self.assertEqual(
            self._title("Train TamGPT-3.0", "Train TamGPT-3.0 on the new corpus"),
            "Train TamGPT-3.0",
        )
        self.assertEqual(
            self._title(
                "Reconcile MSC-2 revenue",
                "Reconcile the MSC-2 revenue recognition and validation",
            ),
            "Reconcile MSC-2 Revenue",
        )

    def test_a_hyphenated_product_name_is_a_name_not_prose(self):
        self.assertEqual(
            self._title(
                "Fix tamfis-code sessions",
                "First you need to fix tamfis-code to properly keep track of sessions",
            ),
            "Fix Tamfis-Code Sessions",
        )

    def test_tightening_is_not_a_generator(self):
        """It may only re-shape the model's own words -- a title the validator
        rejected must still be rejected, and tightening must never invent one."""
        title, reason = state_module.validate_session_title(
            "Please investigate why image and",
            objective="Please investigate why image and video workspace generation keeps failing",
        )
        self.assertEqual(title, "")
        self.assertTrue(reason)
        self.assertEqual(state_module._tighten_llm_title(""), "")


class TitleCommandReportSurvivesRichMarkup(unittest.TestCase):
    """Live-reproduced 2026-09-19: /regenerate-title generated the title and
    then crashed the whole REPL rendering its own report -- one opening green
    tag, two closing ones. From the user's side, "the command did nothing".
    """

    def _render(self, markup):
        from rich.console import Console
        import io
        buffer = io.StringIO()
        Console(file=buffer, width=400).print(markup)
        return buffer.getvalue()

    def test_the_report_has_balanced_markup(self):
        from tamfis_code.interactive import session_title_report

        report = session_title_report(
            "Add Hello Output", "Send Hello Message", "openrouter",
        )
        self.assertEqual(report.count("["), report.count("]"))
        # One opening green tag, one closing -- the old line had TWO closes and
        # rich raised MarkupError, killing the REPL.
        self.assertEqual(report.count("[green]"), 1)
        self.assertEqual(report.count("[/green]"), 1)
        self.assertTrue(report.startswith("[green]"))
        rendered = self._render(report)
        self.assertIn("Send Hello Message", rendered)
        self.assertIn("Add Hello Output", rendered)  # "was:" line

    def test_a_bracket_laden_title_renders_instead_of_raising(self):
        from rich.errors import MarkupError

        from tamfis_code.interactive import session_title_report

        report = session_title_report("Build A", "Fix [beta] Routing [/green]", "auto")
        try:
            rendered = self._render(report)
        except MarkupError as exc:  # pragma: no cover - the regression
            self.fail(f"title report raised MarkupError: {exc}")
        self.assertIn("Fix [beta] Routing [/green]", rendered)

    def test_an_unchanged_title_says_so(self):
        from tamfis_code.interactive import session_title_report

        report = session_title_report("Fix Sessions", "Fix Sessions", "auto")
        self.assertIn("unchanged", report)
        self.assertIn("Fix Sessions", self._render(report))


if __name__ == "__main__":
    unittest.main()
