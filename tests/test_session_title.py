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
        # model attribution reports the AUTO routing directive; the
        # concrete underlying model lives in the router's own telemetry.
        self.assertEqual(state.title_model, "auto")
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
            state_module.get_session_state(1).session_title, "Investigate retry loop timeout",
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
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix flaky auth test")

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
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix flaky auth test")

    def test_a_nonsubstantive_later_turn_does_not_retitle_the_session(self):
        """Acceptance TEST 9: 'Continue' must not produce a useless title,
        and must not retitle an already-titled session."""
        state_module.save_session_state(1, workspace_root="/a")
        self._upgrade("Fix flaky auth test")
        self._upgrade("Fix the auth retry storm", objective="Continue")
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix flaky auth test")

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
            self._upgrade(ConnectionError("boom"))
        self.assertIn("[title]", err.getvalue())
        self.assertIn("provider attempt failed", err.getvalue())


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
        patcher, _ = _fake_providers("Stale title")
        with patcher:
            asyncio.run(state_module.upgrade_session_title_with_ai(1, "objective"))
        patcher.stop()
        self.assertEqual(state_module.get_session_state(1).session_title, "Stale title")

        self.assertTrue(state_module.request_session_title_regeneration(1))
        state = state_module.get_session_state(1)
        self.assertEqual(state.session_title, "")
        self.assertFalse(state.ai_title_attempted)

        patcher, _ = _fake_providers("Fresh Semantic Title")
        with patcher:
            asyncio.run(state_module.upgrade_session_title_with_ai(1, "new objective"))
        patcher.stop()
        self.assertEqual(state_module.get_session_state(1).session_title, "Fresh Semantic Title")

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
        self.assertEqual(diag["title"], "Fix flaky auth test")
        self.assertEqual(diag["title_source"], "llm")
        self.assertEqual(diag["title_model"], "auto")
        self.assertTrue(diag["title_generated_at"])
        self.assertIsNone(diag["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
