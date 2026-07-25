"""reasoning_effort must be gated per-model, not just per-provider.

Confirmed live: a real streaming NVIDIA NIM call to meta/llama-3.1-70b-
instruct (NVIDIA's own current default_model) with reasoning_effort="high"
produced zero chunks in 40+ seconds -- a real hang, not a clean error --
while the identical call without reasoning_effort returned in well under a
second, and nvidia/nemotron-3-super-120b-a12b handled reasoning_effort
fine. REASONING_EFFORT_CAPABLE_PROVIDERS alone (provider-only gating) was
correct when NVIDIA's default_model was a nemotron variant, but silently
became a hang bug the moment the default moved to a plain instruct model
(the v0.4.39 kimi-k2.6-404 fix) -- nobody had re-verified reasoning_effort
against the new default. reasoning_effort_capable() closes that gap.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamfis_code.providers import ProviderType, reasoning_effort_capable


class ReasoningEffortCapabilityTests(unittest.TestCase):
    def test_nemotron_on_nvidia_is_capable(self):
        self.assertTrue(reasoning_effort_capable(ProviderType.NVIDIA, "nvidia/nemotron-3-super-120b-a12b"))
        self.assertTrue(reasoning_effort_capable(ProviderType.NVIDIA, "nvidia/nemotron-3-ultra-550b-a55b"))

    def test_llama_on_nvidia_is_not_capable(self):
        # This is the exact model+provider combination confirmed live to hang.
        self.assertFalse(reasoning_effort_capable(ProviderType.NVIDIA, "meta/llama-3.1-70b-instruct"))
        self.assertFalse(reasoning_effort_capable(ProviderType.NVIDIA, "meta/llama-3.1-405b-instruct"))

    def test_other_instruct_models_on_nvidia_are_not_capable(self):
        for model in (
            "moonshotai/kimi-k2.6",
            "mistralai/mistral-large-2-123b",
            "google/gemma-2-27b-it",
            "microsoft/phi-3-medium-128k-instruct",
        ):
            with self.subTest(model=model):
                self.assertFalse(reasoning_effort_capable(ProviderType.NVIDIA, model))

    def test_case_insensitive_nemotron_match(self):
        self.assertTrue(reasoning_effort_capable(ProviderType.NVIDIA, "NVIDIA/Nemotron-3-Super"))

    def test_openrouter_and_tier_iv_remain_provider_level(self):
        self.assertTrue(reasoning_effort_capable(ProviderType.OPENROUTER, "anything/at-all"))
        self.assertTrue(reasoning_effort_capable(ProviderType.TIER_IV, ""))

    def test_providers_outside_the_capable_set_are_never_capable(self):
        self.assertFalse(reasoning_effort_capable(ProviderType.HF, "Qwen/Qwen2.5-Coder-32B-Instruct"))
        self.assertFalse(reasoning_effort_capable(ProviderType.LOCAL, "nemotron-lookalike:latest"))


if __name__ == "__main__":
    unittest.main()
