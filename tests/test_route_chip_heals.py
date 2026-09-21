"""The footer chip "⟳ route down · credits" must go away once a request has COMPLETED after the exhaustion.

Owner paste 2026-09-21: a run that completed on TamfisGPT-Ultra still showed "⟳ route down · credits" for 15 minutes,
because one route's credit exhaustion was recorded and nothing recorded that another route then served the request.
"""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tamfis_code import state as st


class RouteChipTests(unittest.TestCase):
    def setUp(self):
        self._orig = (st.CONFIG_DIR, st.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        st.CONFIG_DIR = base / ".config"
        st.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        st.CONFIG_DIR, st.STATE_PATH = self._orig
        self.tmp.cleanup()

    def test_an_exhaustion_alone_shows_route_down(self):
        st.record_route_error(5, provider="openrouter", error="402 insufficient credits")
        self.assertIn("route down", st.route_status_compact(5))

    def test_a_request_that_completes_afterwards_heals_it(self):
        st.record_route_error(6, provider="openrouter", error="402 insufficient credits")
        st.record_route_served(6)
        self.assertEqual(st.route_status_compact(6), "")

    def test_an_exhaustion_after_the_last_served_request_shows_again(self):
        st.record_route_served(7)
        state = st.get_session_state(7)
        state.last_route_served_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        st.put_session_state(state)
        st.record_route_error(7, provider="nvidia", error="429 credits exhausted")
        self.assertIn("route down", st.route_status_compact(7))
        st.record_route_served(7)          # inside the rate-limit window, but an exhaustion needs healing
        self.assertEqual(st.route_status_compact(7), "")

    def test_serving_is_write_rate_limited_when_nothing_needs_healing(self):
        st.record_route_served(8)
        first = st.get_session_state(8).last_route_served_at
        st.record_route_served(8)
        self.assertEqual(st.get_session_state(8).last_route_served_at, first)


if __name__ == "__main__":
    unittest.main()
