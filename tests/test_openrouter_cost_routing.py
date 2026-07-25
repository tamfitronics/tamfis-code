"""OpenRouter free-vs-paid model selection (credit-saving routing).

OpenRouter was previously always called with its paid default_model
(google/gemini-2.5-flash) regardless of how trivial the task was, even
though the config already listed free (`:free`-suffixed) models nobody
ever selected. ProviderManager.select_model now uses a provider's
free_model for routine tasks and only escalates to default_model (the
paid tier) for deep research or genuinely demanding coding/analysis --
see providers.py's _task_needs_paid_tier. Other providers (no free_model
configured) are unaffected and always return default_model.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.routing import TaskProfile, TaskType, classify_task


class OpenRouterFreeTierSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manager = ProviderManager()
        self.config = self.manager.PROVIDERS[ProviderType.OPENROUTER]

    def test_openrouter_has_a_free_model_configured(self):
        self.assertTrue(self.config.free_model)
        self.assertIn(self.config.free_model, self.config.models)

    def test_no_task_profile_uses_the_free_model(self):
        self.assertEqual(self.manager.select_model(self.config, None), self.config.free_model)

    def test_plain_conversation_uses_the_free_model(self):
        profile = classify_task("hello")
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.free_model)

    def test_plain_question_uses_the_free_model(self):
        profile = classify_task("what does this function do?")
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.free_model)

    def test_simple_inspection_uses_the_free_model(self):
        profile = classify_task("search the codebase for TODO comments")
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.free_model)

    def test_research_escalates_to_the_paid_model(self):
        profile = classify_task("search the web for the latest news on this library")
        self.assertEqual(profile.task_type, TaskType.RESEARCH)
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.default_model)

    def test_debug_escalates_to_the_paid_model(self):
        profile = classify_task("fix the bug causing the crash on startup")
        self.assertEqual(profile.task_type, TaskType.DEBUG)
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.default_model)

    def test_edit_escalates_to_the_paid_model(self):
        profile = classify_task("implement a new caching layer for this module")
        self.assertEqual(profile.task_type, TaskType.EDIT)
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.default_model)

    def test_audit_escalates_to_the_paid_model(self):
        profile = classify_task("audit the entire repository end-to-end")
        self.assertEqual(profile.task_type, TaskType.AUDIT)
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.default_model)

    def test_any_high_complexity_profile_escalates_even_without_a_named_task_type(self):
        profile = TaskProfile(TaskType.MIXED, "high", True, True, True, True, "frontier")
        self.assertEqual(self.manager.select_model(self.config, profile), self.config.default_model)


class OtherProvidersUnaffectedTests(unittest.TestCase):
    """Providers with no free_model must always return default_model,
    regardless of task complexity -- this feature is OpenRouter-only."""

    def setUp(self):
        self.manager = ProviderManager()

    def test_nvidia_always_uses_default_model(self):
        config = self.manager.PROVIDERS[ProviderType.NVIDIA]
        self.assertIsNone(config.free_model)
        for text in ("hello", "fix this bug", "search the web for news"):
            profile = classify_task(text)
            with self.subTest(text=text):
                self.assertEqual(self.manager.select_model(config, profile), config.default_model)

    def test_hf_always_uses_default_model(self):
        config = self.manager.PROVIDERS[ProviderType.HF]
        self.assertIsNone(config.free_model)
        self.assertEqual(self.manager.select_model(config, None), config.default_model)


if __name__ == "__main__":
    unittest.main()
