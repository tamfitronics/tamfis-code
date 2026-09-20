"""tamfis-code remembers, across runs, which models are fast, slow or dead.

Live measurement 2026-09-19 (this CLI's own prompt + 21 tools, NVIDIA NIM): super made
its tool call in 1.0s, ultra 9.2s, lightning 14s, and kimi-k3 / glm-5.3 -- first in the
configured order -- never answered inside 60s. With a 45s first-byte timeout and an
in-memory 30s health circuit, every run rediscovered that, losing 45-90s before reaching
a model that answers in a second. Owner request: "still not as smart, fast, intelligent
and efficient as you".
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tamfis_code import route_stats

KIMI = "moonshotai/kimi-k3"
GLM = "z-ai/glm-5.3"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
LIGHT = "nvidia/nemotron-3.5-lightning-30b-a3b"
ORDER = [KIMI, GLM, SUPER, ULTRA, LIGHT]


class _Store:
    """An isolated, ENABLED route_stats store."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patches = [
            patch.object(route_stats, "_path", lambda: Path(self._tmp.name) / "route_stats.json"),
            patch.object(route_stats, "ENABLE_UNDER_TEST", True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()


class RankTests(_Store, unittest.TestCase):
    def test_with_nothing_measured_the_preference_order_stands(self):
        self.assertEqual(route_stats.rank(ORDER), ORDER)

    def test_a_model_that_timed_out_goes_last_and_nothing_is_dropped(self):
        route_stats.record_failure(KIMI)
        route_stats.record_failure(GLM)
        ranked = route_stats.rank(ORDER)
        self.assertEqual(ranked[:3], [SUPER, ULTRA, LIGHT])
        self.assertEqual(sorted(ranked), sorted(ORDER))
        self.assertEqual(set(ranked[3:]), {KIMI, GLM})

    def test_a_measured_slow_model_is_tried_after_a_fast_one(self):
        """The real numbers: ultra 9.2s vs super 1.0s -> super first even though ultra is
        listed first."""
        route_stats.record_latency(ULTRA, 9.2)
        route_stats.record_latency(SUPER, 1.0)
        self.assertEqual(route_stats.rank([ULTRA, SUPER, LIGHT]), [SUPER, LIGHT, ULTRA])

    def test_a_model_slower_than_the_absolute_limit_is_demoted_even_alone(self):
        route_stats.record_latency(LIGHT, 14.0)
        self.assertEqual(route_stats.rank([LIGHT, SUPER]), [SUPER, LIGHT])

    def test_similar_speeds_keep_the_preference_order(self):
        route_stats.record_latency(KIMI, 2.0)
        route_stats.record_latency(SUPER, 1.0)
        self.assertEqual(route_stats.rank([KIMI, SUPER]), [KIMI, SUPER])  # only 2x, and kimi is preferred

    def test_a_preferred_model_that_is_healthy_and_fast_stays_first(self):
        route_stats.record_latency(KIMI, 1.5)
        self.assertEqual(route_stats.rank(ORDER)[0], KIMI)

    def test_penalised_models_are_ordered_by_when_their_cooloff_ends(self):
        route_stats.record_failure(KIMI)
        route_stats.record_failure(GLM)
        route_stats.record_failure(GLM)  # second failure -> longer cool-off
        ranked = route_stats.rank([GLM, KIMI])
        self.assertEqual(ranked, [KIMI, GLM])  # kimi returns sooner

    def test_short_and_duplicate_inputs_are_safe(self):
        self.assertEqual(route_stats.rank([]), [])
        self.assertEqual(route_stats.rank([SUPER]), [SUPER])
        self.assertEqual(route_stats.rank([SUPER, SUPER, ULTRA]), [SUPER, ULTRA])


class RecordingTests(_Store, unittest.TestCase):
    def test_failures_escalate_and_one_success_clears_them(self):
        now = time.time()
        route_stats.record_failure(KIMI)
        first = route_stats.describe([KIMI], now=now)[0]["skipped_for"]
        route_stats.record_failure(KIMI)
        second = route_stats.describe([KIMI], now=now)[0]["skipped_for"]
        self.assertAlmostEqual(first, 300, delta=5)
        self.assertGreater(second, first * 2)
        route_stats.record_latency(KIMI, 3.0)
        self.assertFalse(route_stats.is_penalized(KIMI))
        route_stats.record_failure(KIMI)  # back to the shortest cool-off, not the escalated one
        self.assertAlmostEqual(route_stats.describe([KIMI])[0]["skipped_for"], 300, delta=5)

    def test_the_cooloff_is_capped(self):
        for _ in range(12):
            route_stats.record_failure(KIMI)
        self.assertLessEqual(route_stats.describe([KIMI])[0]["skipped_for"], 3600 + 5)

    def test_latency_is_a_running_average(self):
        route_stats.record_latency(SUPER, 1.0)
        route_stats.record_latency(SUPER, 3.0)
        value = route_stats.expected_latency(SUPER)
        self.assertGreater(value, 1.0)
        self.assertLess(value, 3.0)

    def test_the_memory_survives_a_new_process(self):
        route_stats.record_failure(KIMI)
        route_stats.record_failure(GLM)
        route_stats.record_latency(SUPER, 1.0)
        # a fresh "process": nothing cached in memory, only the file
        self.assertTrue((Path(self._tmp.name) / "route_stats.json").exists())
        self.assertTrue(route_stats.is_penalized(KIMI))
        self.assertEqual(route_stats.rank(ORDER)[0], SUPER)
        self.assertEqual(route_stats.rank(ORDER)[-2:], [KIMI, GLM])

    def test_stale_latency_is_ignored(self):
        route_stats.record_latency(SUPER, 30.0)
        self.assertIsNone(route_stats.expected_latency(SUPER, now=time.time() + route_stats.STALE_AFTER_SECONDS + 60))

    def test_junk_input_and_a_corrupt_file_never_raise(self):
        route_stats.record_latency("", 1.0)
        route_stats.record_latency(SUPER, -5)
        route_stats.record_latency(SUPER, "fast")  # type: ignore[arg-type]
        (Path(self._tmp.name) / "route_stats.json").write_text("{not json")
        self.assertEqual(route_stats.rank(ORDER), ORDER)
        route_stats.record_failure(KIMI)  # recovers by rewriting
        self.assertTrue(route_stats.is_penalized(KIMI))

    def test_the_store_is_bounded(self):
        for index in range(route_stats._MAX_MODELS + 30):
            route_stats.record_latency(f"model-{index}", 1.0)
        self.assertLessEqual(len(route_stats._load()), route_stats._MAX_MODELS)


class SafetyTests(unittest.TestCase):
    def test_it_is_inert_under_pytest_by_default(self):
        """The suite drives the real stream code with fake models; it must neither write
        the host's real memory nor have selection depend on it."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(route_stats, "_path", lambda: Path(tmp) / "route_stats.json"):
                route_stats.record_failure(KIMI)
                self.assertFalse((Path(tmp) / "route_stats.json").exists())
                self.assertEqual(route_stats.rank(ORDER), ORDER)

    def test_the_environment_kill_switch_disables_it_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TAMFIS_CODE_ROUTE_STATS": "0"}), \
                patch.object(route_stats, "ENABLE_UNDER_TEST", True), \
                patch.object(route_stats, "_path", lambda: Path(tmp) / "route_stats.json"):
            route_stats.record_failure(KIMI)
            self.assertFalse(route_stats.enabled())
            self.assertFalse((Path(tmp) / "route_stats.json").exists())


class SelectionIntegrationTests(_Store, unittest.TestCase):
    def test_the_runner_skips_a_remembered_dead_model_and_takes_the_fast_one(self):
        from tamfis_code.providers import ProviderManager, ProviderType
        from tamfis_code.runner_local import _select_public_group_model

        manager = ProviderManager.__new__(ProviderManager)
        manager.route_is_healthy = lambda provider, model: True
        config = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
        # nothing known: the configured order leads with the fast model, never Kimi/GLM
        first = _select_public_group_model(manager, ProviderType.NVIDIA, config, None, "pro")
        self.assertEqual(first, SUPER)
        # a fast model that then stalls is skipped by a NEW run, which moves to the next fast one
        route_stats.record_failure(SUPER)
        again = _select_public_group_model(manager, ProviderType.NVIDIA, config, None, "pro")
        self.assertNotIn(again, {SUPER, KIMI, GLM})

    def test_provider_side_selection_uses_the_same_memory(self):
        from tamfis_code.providers import ProviderManager, ProviderType

        manager = ProviderManager.__new__(ProviderManager)
        manager.route_is_healthy = staticmethod(lambda provider, model="": True)
        config = ProviderManager.PROVIDERS[ProviderType.NVIDIA]
        self.assertEqual(ProviderManager.select_model(manager, config, None), SUPER)
        route_stats.record_failure(SUPER)
        self.assertNotIn(ProviderManager.select_model(manager, config, None), {SUPER, KIMI, GLM})

    def test_the_fast_model_is_ahead_of_the_slow_one_in_the_configured_order(self):
        from tamfis_code.providers import ProviderManager, ProviderType

        models = ProviderManager.PROVIDERS[ProviderType.NVIDIA].models
        self.assertLess(models.index(SUPER), models.index(ULTRA))
        # the benchmarked-fast addition (owner: find more free tool-calling models) is in
        # the pool ahead of the slow lightning route
        self.assertIn("meta/muse-glimmer-30b", models)
        self.assertLess(models.index("meta/muse-glimmer-30b"), models.index("nvidia/nemotron-3.5-lightning-30b-a3b"))
        # ...and Kimi K3 / GLM 5.3 (owner ruling 2026-09-19) are the last two
        self.assertEqual(models[-2:], [KIMI, GLM])

    def test_a_timeout_or_server_error_is_remembered_but_a_rate_limit_is_not(self):
        from tamfis_code.providers import ProviderType, record_route_failure_for

        record_route_failure_for(ProviderType.NVIDIA, KIMI, TimeoutError("no response"))
        self.assertTrue(route_stats.is_penalized(KIMI))
        record_route_failure_for(ProviderType.NVIDIA, GLM, Exception("HTTP 503 unavailable"))
        self.assertTrue(route_stats.is_penalized(GLM))
        record_route_failure_for(ProviderType.NVIDIA, SUPER, Exception("HTTP 429 rate limit exceeded"))
        self.assertFalse(route_stats.is_penalized(SUPER))  # account-level, says nothing about speed
        record_route_failure_for(ProviderType.NVIDIA, ULTRA, Exception("HTTP 402 out of credits"))
        self.assertFalse(route_stats.is_penalized(ULTRA))


class StreamMeasurementTests(_Store, unittest.TestCase):
    def _stream(self, chunks, *, hang=False):
        from test_reasoning_plan import _FakeClient, _RecordingRenderer
        from tamfis_code.runner_local import _stream_one_completion

        client = _FakeClient([chunks])
        if hang:
            async def never(**kwargs):
                await asyncio.sleep(30)

            client.chat.completions.create = never
        return asyncio.run(_stream_one_completion(
            client, model=SUPER, messages=[{"role": "user", "content": "go"}], tools=[],
            renderer=_RecordingRenderer(), emit=True,
        ))

    def test_time_to_first_useful_output_is_recorded_for_a_tool_call(self):
        from test_reasoning_plan import _chunk, _delta, _tool_call_delta

        self._stream([
            _chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="c1", name="read_file", arguments='{"path":"a"}')])),
            _chunk(_delta(), finish_reason="tool_calls"),
        ])
        latency = route_stats.expected_latency(SUPER)
        self.assertIsNotNone(latency)
        self.assertLess(latency, 5)

    def test_time_to_first_useful_output_is_recorded_for_answer_text(self):
        from test_reasoning_plan import _chunk, _delta

        self._stream([_chunk(_delta(content="Here is the answer, in a few words.")), _chunk(_delta(), finish_reason="stop")])
        self.assertIsNotNone(route_stats.expected_latency(SUPER))

    def test_a_first_byte_timeout_penalises_the_model_for_the_next_run(self):
        with patch("tamfis_code.runner_local.first_byte_timeout_for", return_value=0.2):
            with self.assertRaises(asyncio.TimeoutError):
                self._stream([], hang=True)
        self.assertTrue(route_stats.is_penalized(SUPER))


class FirstByteTimeoutTests(unittest.TestCase):
    def test_a_normal_request_gets_the_short_timeout(self):
        from tamfis_code.runner_local import PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS, first_byte_timeout_for

        self.assertLessEqual(PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS, 25.0)
        self.assertEqual(first_byte_timeout_for([{"role": "user", "content": "hi"}], None), PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS)

    def test_a_huge_prompt_gets_more_time_up_to_the_old_ceiling(self):
        from tamfis_code.runner_local import PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS, first_byte_timeout_for

        big = [{"role": "user", "content": "x" * 200_000}]
        scaled = first_byte_timeout_for(big, None)
        self.assertGreater(scaled, PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS)
        self.assertLessEqual(scaled, 45.0)
        self.assertEqual(first_byte_timeout_for([{"role": "user", "content": "x" * 5_000_000}], None), 45.0)

    def test_junk_messages_do_not_raise(self):
        from tamfis_code.runner_local import first_byte_timeout_for

        self.assertGreater(first_byte_timeout_for(None, None), 0)
        self.assertGreater(first_byte_timeout_for([object(), {"content": None}], {"x": object()}), 0)


class StartupTests(unittest.TestCase):
    """`tamfis-code --version` used to import the whole openai SDK (~1.7s) before doing
    anything. It is imported when the first client is built, not at import time."""

    def test_importing_the_cli_does_not_import_the_openai_sdk(self):
        code = "import sys, tamfis_code.cli, tamfis_code.providers; print('openai' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout.strip(), "False", result.stderr[-500:])

    def test_the_class_is_still_reachable_and_patchable(self):
        from tamfis_code import providers

        self.assertEqual(providers.AsyncOpenAI.__name__, "AsyncOpenAI")
        with patch.object(providers, "AsyncOpenAI", lambda **kwargs: "fake"):
            self.assertEqual(providers._async_openai_class()(api_key="k"), "fake")


if __name__ == "__main__":
    unittest.main()
