import tempfile
import unittest
from pathlib import Path

from tamfis_code.timeouts import adaptive_command_timeout


class AdaptiveTimeoutTests(unittest.TestCase):
    def test_large_compileall_is_not_limited_to_interactive_120_seconds(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index in range(220):
                (root / f"module_{index}.py").write_text("value = 1\n")
            timeout = adaptive_command_timeout("python3 -m compileall -q .", root, 120)
            self.assertGreater(timeout, 120)

    def test_large_test_package_gets_a_longer_floor(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index in range(220):
                (root / f"test_{index}.py").write_text("def test_ok(): pass\n")
            compile_timeout = adaptive_command_timeout("python3 -m compileall -q .", root, 120)
            test_timeout = adaptive_command_timeout("pytest -q", root, 120)
            self.assertGreater(test_timeout, compile_timeout)

    def test_normal_command_keeps_requested_timeout(self):
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(adaptive_command_timeout("printf ok", raw, 120), 120)

    def test_omitted_timeout_is_calculated_for_a_test_workload(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index in range(220):
                (root / f"test_{index}.py").write_text("def test_ok(): pass\n")
            self.assertGreater(adaptive_command_timeout("pytest -q", root, None), 120)

    def test_package_import_check_is_not_capped_at_short_probe_timeout(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index in range(220):
                (root / f"module_{index}.py").write_text("value = 1\n")
            self.assertGreater(adaptive_command_timeout("python3 -c 'import finitron'", root, 30), 30)
