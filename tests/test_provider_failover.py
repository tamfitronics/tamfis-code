"""A provider that is out of credits must never end the task.

Live-reported 2026-09-19: a 35-minute TamfisGPT-Ultra run ended with

    Error: TamfisGPT-Ultra streaming failed: Error code: 402 - {'error': 'You
    have depleted your monthly included credits. Purchase pre-paid credits to
    continue using Inference Providers.'} ... type `continue` to resume

...with NVIDIA NIM configured and unused.

Two independent gaps made that stop possible, and both are pinned here:

1. The 402 arrived as a wrapper that preserved only the SDK's str(), so
   `provider_error_status` read None and no retryable marker matched the
   credit wording -> the error was classified NON-retryable, so the runner's
   fallback gate (`can_fallback`) was False and no other route was ever tried.
2. Even when an error IS retryable, `fallback_candidates` skips routes whose
   health circuit is open (recently failed). If every alternative is cooling,
   the candidate list came back empty and the task stopped anyway -- even
   though a cooling route is a latency heuristic, not a dead provider.
"""
from __future__ import annotations

import unittest

from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.runner_local import _fallback_candidates_for_turn

LIVE_402 = (
    "Error code: 402 - {'error': 'You have depleted your monthly included "
    "credits. Purchase pre-paid credits to continue using Inference Providers.'}"
)


class ExhaustedProviderIsRetryableTests(unittest.TestCase):
    def test_the_live_wrapped_402_is_classified_retryable(self):
        """The exact live message, with the SDK attributes lost."""
        exc = Exception(LIVE_402)
        self.assertEqual(ProviderManager.provider_error_status(exc), 402)
        self.assertTrue(ProviderManager.is_retryable_provider_error(exc))

    def test_credit_wording_alone_is_enough(self):
        for message in (
            "You have depleted your monthly included credits.",
            "Insufficient credits to continue.",
            "Out of credits for this account.",
            "Please purchase pre-paid credits.",
            "Billing issue on this account.",
        ):
            with self.subTest(message=message):
                self.assertTrue(
                    ProviderManager.is_retryable_provider_error(Exception(message))
                )

    def test_an_http_code_in_the_text_restores_the_routing_decision(self):
        self.assertEqual(
            ProviderManager.provider_error_status(Exception("HTTP 503 Service Unavailable")),
            503,
        )
        self.assertEqual(
            ProviderManager.provider_error_status(Exception("status code: 429")),
            429,
        )

    def test_a_genuinely_non_retryable_error_stays_non_retryable(self):
        """The failover path must not swallow every error: an unrelated,
        unclassifiable failure still stops (that is what `continue` is for)."""
        self.assertFalse(
            ProviderManager.is_retryable_provider_error(Exception("something odd happened"))
        )


class _RoutingDouble(ProviderManager):
    """A manager whose routes are all configured but none are currently healthy."""

    def __init__(self, *, healthy: bool):
        self._healthy = healthy
        self.clients = {ProviderType.NVIDIA: object(), ProviderType.OPENROUTER: object()}
        self.PROVIDERS = {
            provider: type("C", (), {"tool_calling": True, "long_context": True})()
            for provider in (ProviderType.NVIDIA, ProviderType.OPENROUTER)
        }

    @property
    def routing_order(self):
        return [ProviderType.NVIDIA, ProviderType.OPENROUTER]

    def _has_valid_api_key(self, provider):
        return True

    def route_is_healthy(self, provider, model):
        return self._healthy

    def _fallback_provider_allowed(self, provider):
        return True

    def ollama_cloud_is_premium_primary(self):
        return False


class CoolingRoutesAreStillTriedTests(unittest.TestCase):
    def test_cooling_routes_are_excluded_by_default_and_available_on_request(self):
        manager = _RoutingDouble(healthy=False)
        self.assertEqual(
            manager.fallback_candidates(ProviderType.TAMFIS), [],
            "a cooling route is skipped by default (latency heuristic)",
        )
        self.assertEqual(
            [provider.value for provider in manager.fallback_candidates(
                ProviderType.TAMFIS, include_cooling=True,
            )],
            ["nvidia", "openrouter"],
            "an exhausted provider must be able to fall back to a cooling route",
        )

    def test_healthy_routes_are_unchanged(self):
        manager = _RoutingDouble(healthy=True)
        self.assertEqual(
            [provider.value for provider in manager.fallback_candidates(ProviderType.TAMFIS)],
            ["nvidia", "openrouter"],
        )

    def test_the_runner_helper_passes_include_cooling_through(self):
        manager = _RoutingDouble(healthy=False)
        self.assertEqual(
            _fallback_candidates_for_turn(manager, ProviderType.TAMFIS, None),
            [],
        )
        self.assertEqual(
            [p.value for p in _fallback_candidates_for_turn(
                manager, ProviderType.TAMFIS, None, include_cooling=True,
            )],
            ["nvidia", "openrouter"],
        )

    def test_a_lightweight_provider_double_without_the_keyword_still_works(self):
        """Older test/integration doubles accept neither extra keyword -- the
        helper must degrade to the old call instead of raising."""
        class LegacyDouble:
            def fallback_candidates(self, current, task_profile):
                return [ProviderType.NVIDIA]

        self.assertEqual(
            _fallback_candidates_for_turn(
                LegacyDouble(), ProviderType.TAMFIS, None, include_cooling=True,
            ),
            [ProviderType.NVIDIA],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
