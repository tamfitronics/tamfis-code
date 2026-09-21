"""Ollama Cloud's weekly allowance is paced, and a stated reset time is honoured."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tamfis_code import config as config_module
from tamfis_code import ollama_pacing, providers
from tamfis_code.providers import ProviderManager, ProviderType, record_route_failure_for


class _HttpError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status


class ParseResetTests(unittest.TestCase):
    def test_reset_phrases(self):
        cases = {
            "You have reached your weekly usage limit. Resets in 25 minutes.": 25 * 60,
            "sessions resume in 2 hours": 2 * 3600,
            "limit resets in 3 days": 3 * 86400,
            "Resets in 90 seconds": 90,
            "resets in about 1.5 hours": 5400,
        }
        for text, seconds in cases.items():
            self.assertEqual(ollama_pacing.parse_reset_delay(text), seconds, text)

    def test_no_reset_text_means_unknown(self):
        self.assertIsNone(ollama_pacing.parse_reset_delay("weekly usage limit reached"))
        self.assertIsNone(ollama_pacing.parse_reset_delay("resets in 2 months"))
        self.assertIsNone(ollama_pacing.parse_reset_delay(""))


class PacingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._config_dir = config_module.CONFIG_DIR
        config_module.CONFIG_DIR = Path(self.tmp.name)
        # record_request() deliberately no-ops under pytest so test traffic never counts against the
        # real allowance; these tests point it at a temp dir and switch that guard off.
        self._env = patch.dict("os.environ", {"TAMFIS_CODE_OLLAMA_WEEKLY_BUDGET": "70", "TAMFIS_CODE_OLLAMA_DAILY_PACE": "1.5"})
        self._env.start()
        import os
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        ollama_pacing._cache = (0.0, 0, 0)
        ollama_pacing._paced_logged = False
        with providers._HEALTH_LOCK:
            self._health = dict(providers._ROUTE_HEALTH)
            providers._ROUTE_HEALTH.clear()

    def tearDown(self):
        config_module.CONFIG_DIR = self._config_dir
        self._env.stop()
        ollama_pacing._cache = (0.0, 0, 0)
        with providers._HEALTH_LOCK:
            providers._ROUTE_HEALTH.clear()
            providers._ROUTE_HEALTH.update(self._health)
        self.tmp.cleanup()

    def test_daily_pace_marks_ollama_unhealthy_and_only_ollama(self):
        for _ in range(14):
            ollama_pacing.record_request()
        self.assertTrue(ProviderManager.route_is_healthy(ProviderType.OLLAMA_CLOUD, "*"))
        ollama_pacing.record_request()  # 15th = budget/7*1.5
        self.assertFalse(ProviderManager.route_is_healthy(ProviderType.OLLAMA_CLOUD, "*"))
        self.assertFalse(ProviderManager.route_is_healthy(ProviderType.OLLAMA_CLOUD, "glm-5.3:cloud"))
        self.assertTrue(ProviderManager.route_is_healthy(ProviderType.OLLAMA_CLOUD, "*", ignore_pacing=True))
        self.assertTrue(ProviderManager.route_is_healthy(ProviderType.NVIDIA, "*"))
        self.assertTrue(ProviderManager.route_is_healthy(ProviderType.OPENROUTER, "*"))

    def test_zero_budget_disables_pacing(self):
        with patch.dict("os.environ", {"TAMFIS_CODE_OLLAMA_WEEKLY_BUDGET": "0"}):
            for _ in range(200):
                ollama_pacing.record_request()
            self.assertTrue(ProviderManager.route_is_healthy(ProviderType.OLLAMA_CLOUD, "*"))

    def test_counts_survive_a_new_process_and_expire_after_a_week(self):
        for _ in range(5):
            ollama_pacing.record_request()
        ollama_pacing._cache = (0.0, 0, 0)
        self.assertEqual(ollama_pacing.usage(), (5, 5))
        real_time = time.time
        with patch("time.time", lambda: real_time() + 8 * 86400):
            ollama_pacing._cache = (0.0, 0, 0)
            self.assertEqual(ollama_pacing.usage(), (0, 0))

    def test_pacing_check_is_a_cheap_cached_read(self):
        ollama_pacing.usage()
        started = time.perf_counter()
        for _ in range(20000):
            ProviderManager.route_is_healthy(ProviderType.OLLAMA_CLOUD, "*")
        self.assertLess(time.perf_counter() - started, 1.5)

    def test_pacing_reorders_but_never_strands_an_ollama_only_setup(self):
        for _ in range(15):
            ollama_pacing.record_request()
        manager = ProviderManager.__new__(ProviderManager)
        manager.clients = {ProviderType.OLLAMA_CLOUD: object()}
        manager.config = {ProviderType.OLLAMA_CLOUD.value: True}
        manager._has_valid_api_key = lambda provider: provider == ProviderType.OLLAMA_CLOUD
        chosen = ProviderManager._select_best_provider(manager, None)
        self.assertEqual(chosen, ProviderType.OLLAMA_CLOUD)

    def test_health_status_says_paced_not_circuit_open(self):
        for _ in range(15):
            ollama_pacing.record_request()
        manager = ProviderManager.__new__(ProviderManager)
        self.assertEqual(ProviderManager.provider_health_status(manager, ProviderType.OLLAMA_CLOUD), "paced")


class ResetAwareParkingTests(unittest.TestCase):
    def setUp(self):
        with providers._HEALTH_LOCK:
            self._health = dict(providers._ROUTE_HEALTH)
            providers._ROUTE_HEALTH.clear()

    def tearDown(self):
        with providers._HEALTH_LOCK:
            providers._ROUTE_HEALTH.clear()
            providers._ROUTE_HEALTH.update(self._health)

    def _parked_for(self, message):
        record_route_failure_for(
            ProviderType.OLLAMA_CLOUD, "glm-5.3:cloud", _HttpError(429, message),
            provider_config=ProviderManager.PROVIDERS[ProviderType.OLLAMA_CLOUD],
        )
        with providers._HEALTH_LOCK:
            state = providers._ROUTE_HEALTH[(ProviderType.OLLAMA_CLOUD.value, "*")]
        return state.circuit_open_until - time.monotonic()

    def test_a_stated_reset_time_shortens_the_park(self):
        parked = self._parked_for("You have reached your weekly usage limit. Resets in 25 minutes.")
        self.assertGreater(parked, 25 * 60)
        self.assertLess(parked, 27 * 60)  # reset + 60 s margin, not the flat 6 h

    def test_unknown_reset_keeps_the_conservative_park(self):
        parked = self._parked_for("weekly usage limit reached")
        self.assertGreater(parked, 5.9 * 3600)

    def test_a_long_reset_is_capped_at_the_usage_limit_park(self):
        parked = self._parked_for("weekly usage limit reached. resets in 4 days")
        self.assertLessEqual(parked, providers.USAGE_LIMIT_COOLDOWN_SECONDS + 1)


if __name__ == "__main__":
    unittest.main()
