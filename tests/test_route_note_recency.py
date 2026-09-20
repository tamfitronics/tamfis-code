"""The footer route note reports RECENT trouble, not the session's whole history.

Owner-reported 2026-09-20: a footer reading "⟳ rerouted · credits · 13 failovers" stayed on
screen long after routing had healed, because the count covered every failover the session had
ever recorded. Recent trouble is still shown; older events stay in /status and /routes.
"""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tamfis_code import state as state_module
from tamfis_code.state import FOOTER_ROUTE_NOTE_WINDOW_SECONDS, route_status_compact, route_status_line


def _ago(minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _diag(failover_minutes=(), exhaustion_minutes=None):
    failovers = [{"kind": "failover", "at": _ago(m)} for m in failover_minutes]
    exhausted = [{"kind": "exhaustion", "at": _ago(exhaustion_minutes)}] if exhaustion_minutes is not None else []
    return {
        "current": {"provider": "x"}, "failovers": failovers, "exhausted": exhausted,
        "last_exhaustion": exhausted[-1] if exhausted else None, "cooling": [], "events": failovers + exhausted,
    }


def _note(diag, **kwargs):
    with patch.object(state_module, "route_diagnostics", return_value=diag):
        return route_status_compact(1, **kwargs)


class FooterRouteNoteRecencyTests(unittest.TestCase):
    def test_old_trouble_no_longer_shows_in_the_footer(self):
        self.assertEqual(_note(_diag(failover_minutes=[600, 500, 400], exhaustion_minutes=300)), "")

    def test_the_reported_case_thirteen_old_failovers_and_a_stale_credit_note(self):
        self.assertEqual(_note(_diag(failover_minutes=[120 + i for i in range(13)], exhaustion_minutes=110)), "")

    def test_recent_trouble_is_still_shown(self):
        note = _note(_diag(failover_minutes=[2]))
        self.assertIn("rerouted", note)
        self.assertIn("1 failover", note)
        self.assertIn("credits", _note(_diag(failover_minutes=[2], exhaustion_minutes=1)))

    def test_only_recent_failovers_are_counted(self):
        note = _note(_diag(failover_minutes=[1, 3, 500, 600, 700]))
        self.assertIn("2 failovers", note)

    def test_a_recent_exhaustion_with_no_failover_says_the_route_is_down(self):
        self.assertIn("route down", _note(_diag(exhaustion_minutes=1)))

    def test_the_window_boundary(self):
        edge = FOOTER_ROUTE_NOTE_WINDOW_SECONDS / 60
        self.assertNotEqual(_note(_diag(failover_minutes=[edge - 1])), "")
        self.assertEqual(_note(_diag(failover_minutes=[edge + 1])), "")

    def test_an_event_with_no_readable_timestamp_is_not_shown_as_current(self):
        diag = _diag()
        diag["failovers"] = [{"kind": "failover"}, {"kind": "failover", "at": "garbage"}]
        self.assertEqual(_note(diag), "")

    def test_no_window_restores_the_full_history(self):
        note = _note(_diag(failover_minutes=[600, 500, 400], exhaustion_minutes=300), window_seconds=None)
        self.assertIn("3 failovers", note)
        self.assertIn("credits", note)

    def test_status_keeps_the_whole_history(self):
        diag = _diag(failover_minutes=[600, 500, 400], exhaustion_minutes=300)
        with patch.object(state_module, "route_diagnostics", return_value=diag):
            line = route_status_line(1)
        self.assertIn("3 failovers", line)
        self.assertIn("route exhausted", line)

    def test_never_names_a_provider(self):
        note = _note(_diag(failover_minutes=[1], exhaustion_minutes=1))
        for word in ("nvidia", "ollama", "huggingface", "openrouter"):
            self.assertNotIn(word, note.lower())


if __name__ == "__main__":
    unittest.main()
