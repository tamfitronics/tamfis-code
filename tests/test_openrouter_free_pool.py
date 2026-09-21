"""OpenRouter free-model pool: a 429/403 on one model rotates, it does not park OpenRouter."""
import time
import unittest
from unittest.mock import patch

from tamfis_code import providers, route_stats
from tamfis_code.providers import ProviderManager, ProviderType, record_route_failure_for


class _HttpError(Exception):
    def __init__(self, status, message=""):
        super().__init__(message or f"HTTP {status}")
        self.status_code = status


DEAD_IDS = ("qwen/qwen3-coder:free", "meta-llama/llama-3.3-70b-instruct:free")


class FreePoolTests(unittest.TestCase):
    def setUp(self):
        with providers._HEALTH_LOCK:
            self._saved = dict(providers._ROUTE_HEALTH)
            providers._ROUTE_HEALTH.clear()
        # route_stats persists to the real config dir: rank without any stored data.
        self._rank = patch.object(route_stats, "rank", side_effect=lambda models, **_: list(dict.fromkeys(models)))
        self._rank.start()
        self.config = ProviderManager.PROVIDERS[ProviderType.OPENROUTER]
        self.pool = list(self.config.free_models)

    def tearDown(self):
        self._rank.stop()
        with providers._HEALTH_LOCK:
            providers._ROUTE_HEALTH.clear()
            providers._ROUTE_HEALTH.update(self._saved)

    def test_pool_is_the_verified_set_with_no_dead_ids(self):
        self.assertEqual(self.config.free_model, self.pool[0])
        self.assertEqual(self.pool[-1], "openrouter/free")  # OpenRouter's own juggler is the last resort
        for model in self.pool:
            self.assertIn(model, self.config.models)
            self.assertTrue(model.endswith(":free") or model == "openrouter/free", model)
        for dead in DEAD_IDS:
            self.assertNotIn(dead, self.config.models)
            self.assertNotIn(dead, self.pool)
        for expected in ("nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-ultra-550b-a55b:free",
                         "poolside/laguna-s-2.1:free", "cohere/north-mini-code:free", "nex-agi/nex-n2.5-pro:free"):
            self.assertIn(expected, self.pool)

    def test_healthy_pool_selects_the_head_without_delay(self):
        started = time.perf_counter()
        for _ in range(1000):
            self.assertEqual(ProviderManager.select_free_model(self.config), self.pool[0])
        self.assertLess(time.perf_counter() - started, 0.5)  # pure in-memory: no probe, no sleep

    def test_429_on_one_model_rotates_to_the_next_and_keeps_openrouter_available(self):
        record_route_failure_for(ProviderType.OPENROUTER, self.pool[0], _HttpError(429, "upstream rate-limited"))
        self.assertTrue(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, ""), "provider must not be parked")
        self.assertFalse(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, self.pool[0]))
        self.assertEqual(ProviderManager.select_free_model(self.config), self.pool[1])

    def test_403_on_one_model_rotates_too(self):
        record_route_failure_for(ProviderType.OPENROUTER, self.pool[0], _HttpError(403, "Forbidden"))
        self.assertTrue(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, ""))
        self.assertEqual(ProviderManager.select_free_model(self.config), self.pool[1])

    def test_consecutive_failures_walk_the_pool_and_never_repeat_a_failed_model(self):
        seen = []
        for _ in range(len(self.pool)):
            choice = ProviderManager.select_free_model(self.config)
            self.assertNotIn(choice, seen)
            seen.append(choice)
            record_route_failure_for(ProviderType.OPENROUTER, choice, _HttpError(429))
        self.assertEqual(seen, self.pool)

    def test_provider_is_parked_only_when_every_free_model_is_cooling(self):
        for model in self.pool:
            record_route_failure_for(ProviderType.OPENROUTER, model, _HttpError(429))
        self.assertFalse(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, ""))

    def test_account_wide_daily_free_cap_parks_the_provider_at_once(self):
        record_route_failure_for(
            ProviderType.OPENROUTER, self.pool[0],
            _HttpError(429, "Rate limit exceeded: free-models-per-day. Add credits to unlock more"),
        )
        self.assertFalse(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, ""))

    def test_out_of_credit_402_parks_the_provider(self):
        record_route_failure_for(ProviderType.OPENROUTER, self.pool[0], _HttpError(402, "insufficient credits"))
        self.assertFalse(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, ""))

    def test_a_paid_model_failure_still_parks_the_provider(self):
        record_route_failure_for(ProviderType.OPENROUTER, "qwen/qwen3-coder", _HttpError(429))
        self.assertFalse(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, ""))

    def test_select_model_uses_the_pool_for_routine_work(self):
        manager = ProviderManager.__new__(ProviderManager)
        record_route_failure_for(ProviderType.OPENROUTER, self.pool[0], _HttpError(429))
        with patch.object(providers, "_paid_model_allowed", return_value=False):
            self.assertEqual(ProviderManager.select_model(manager, self.config, None), self.pool[1])

    def test_nim_rotation_is_unchanged(self):
        nim = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
        record_route_failure_for(ProviderType.NVIDIA, nim.default_model, _HttpError(429), provider_config=nim)
        self.assertTrue(ProviderManager.route_is_healthy(ProviderType.NVIDIA, ""))


if __name__ == "__main__":
    unittest.main()
