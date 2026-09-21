"""The app must sense a newer release and tell the user -- without slowing a launch or breaking output.

Found 2026-09-21: the published manifest was frozen at 1.6.21 while the build was 1.6.78, and the REPL's
30-minute "live" poll could never see a new release because the first answer was remembered forever.
"""
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import __version__, config as config_module, self_update

BASE = self_update.RELEASE_BASE


def _manifest(version, sha="a" * 64):
    return {"version": version, "url": f"{BASE}/tamfis_code-{version}-py3-none-any.whl", "sha256": sha}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urlopen_returning(*manifests):
    """Each call answers with the next manifest (or raises for None)."""
    queue = list(manifests)
    calls = []

    def fake(request, timeout=None):
        calls.append(request.full_url)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if item is None:
            raise OSError("offline")
        return _Response(json.dumps(item).encode())

    fake.calls = calls
    return fake


def _bump(version=None, by=1):
    major, minor, patch_ = (int(x) for x in (version or __version__).split("."))
    return f"{major}.{minor}.{patch_ + by}"


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._config_dir = config_module.CONFIG_DIR
        config_module.CONFIG_DIR = Path(self.tmp.name)
        self_update._release, self_update._release_at = None, 0.0
        # never let a source checkout on this machine answer for the "published" release
        self._repo = patch.object(self_update, "_repo_version", return_value=None)
        self._repo.start()

    def tearDown(self):
        self._repo.stop()
        config_module.CONFIG_DIR = self._config_dir
        self_update._release, self_update._release_at = None, 0.0
        self.tmp.cleanup()


class DetectionTests(_Base):
    def test_a_newer_published_release_is_detected_and_remembered_on_disk(self):
        newer = _bump()
        with patch.object(self_update, "urlopen", _urlopen_returning(_manifest(newer))):
            self.assertEqual(self_update.check_update_available(), newer)
        # a NEW process (no memory) alerts instantly from the cache, with no network at all
        self_update._release, self_update._release_at = None, 0.0
        with patch.object(self_update, "urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(self_update.cached_update_available(), newer)

    def test_the_same_or_an_older_release_is_not_an_update(self):
        for version in (__version__, "1.6.21"):
            self_update._release, self_update._release_at = None, 0.0
            with patch.object(self_update, "urlopen", _urlopen_returning(_manifest(version))):
                self.assertIsNone(self_update.check_update_available())

    def test_a_release_published_mid_session_is_seen_by_the_live_poll(self):
        """The bug: the first answer was cached for the life of the process."""
        newer = _bump(by=2)
        fake = _urlopen_returning(_manifest(__version__), _manifest(newer))
        with patch.object(self_update, "urlopen", fake):
            self.assertIsNone(self_update.check_update_available())          # startup: up to date
            self.assertEqual(self_update.refresh_update_cache(), newer)       # 30 minutes later
        self.assertEqual(len(fake.calls), 2)

    def test_a_network_failure_keeps_the_last_known_answer_instead_of_saying_up_to_date(self):
        newer = _bump()
        with patch.object(self_update, "urlopen", _urlopen_returning(_manifest(newer), None)):
            self.assertEqual(self_update.refresh_update_cache(), newer)
            self.assertEqual(self_update.refresh_update_cache(), newer)       # second fetch fails

    def test_memory_cache_avoids_refetching_within_its_ttl(self):
        fake = _urlopen_returning(_manifest(_bump()))
        with patch.object(self_update, "urlopen", fake):
            self_update.check_update_available()
            self_update.check_update_available()
        self.assertEqual(len(fake.calls), 1)


class CacheSafetyTests(_Base):
    def test_a_tampered_cache_cannot_point_the_updater_at_another_host(self):
        evil = {"version": _bump(), "url": "https://evil.example/x.whl", "sha256": "b" * 64}
        self_update._write_cache(release=evil, checked_at=time.time())
        self.assertIsNone(self_update.cached_release())
        self.assertIsNone(self_update.cached_update_available())

    def test_garbage_cache_files_are_ignored(self):
        self_update._cache_path().parent.mkdir(parents=True, exist_ok=True)
        for junk in ("", "not json", "[1,2]", '{"release": 5}'):
            self_update._cache_path().write_text(junk)
            self.assertIsNone(self_update.cached_release())
            self.assertEqual(self_update.cache_age_seconds(), float("inf"))

    def test_an_invalid_manifest_from_the_network_is_rejected(self):
        for bad in ({"version": "banana", "url": f"{BASE}/x.whl", "sha256": "a" * 64},
                    {"version": "9.9.9", "url": "http://other/x.whl", "sha256": "a" * 64},
                    {"version": "9.9.9", "url": f"{BASE}/x.whl", "sha256": "zz"}):
            with patch.object(self_update, "urlopen", _urlopen_returning(bad)):
                self.assertIsNone(self_update.check_update_available())


class OneShotNoticeTests(_Base):
    def test_notice_needs_a_newer_cached_release_and_is_limited_to_once_a_day(self):
        self.assertIsNone(self_update.should_notify_oneshot())               # nothing cached
        newer = _bump()
        self_update._write_cache(release=_manifest(newer), checked_at=time.time())
        self.assertEqual(self_update.should_notify_oneshot(), newer)
        self_update.mark_oneshot_notified()
        self.assertIsNone(self_update.should_notify_oneshot())               # already told today
        self_update._write_cache(oneshot_notified_at=time.time() - 25 * 3600)
        self.assertEqual(self_update.should_notify_oneshot(), newer)

    def test_the_cli_prints_one_dim_stderr_line_only_on_a_terminal_and_never_for_machine_output(self):
        from tamfis_code import cli

        newer = _bump()
        self_update._write_cache(release=_manifest(newer), checked_at=time.time())
        tty = io.StringIO()
        tty.isatty = lambda: True
        with patch("sys.stderr", tty):
            cli._announce_update_once(["ask", "hi"])
        self.assertIn(newer, tty.getvalue())
        self.assertIn("tamfis-code update", tty.getvalue())

        self_update._write_cache(oneshot_notified_at=0)
        quiet = io.StringIO()
        quiet.isatty = lambda: True
        with patch("sys.stderr", quiet):
            cli._announce_update_once(["ask", "hi", "--output-mode", "json"])
        self.assertEqual(quiet.getvalue(), "")
        pipe = io.StringIO()
        with patch("sys.stderr", pipe):                                       # not a tty
            cli._announce_update_once(["ask", "hi"])
        self.assertEqual(pipe.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
