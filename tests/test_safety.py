import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import state as state_module
from tamfis_code.safety import (
    RISK_DANGEROUS,
    RISK_MEDIUM,
    RISK_READ_ONLY,
    classify_command_risk,
    classify_path_risk,
    classify_tool_call_risk,
    record_mutation,
    redact_secrets,
    revert_mutation,
    revert_transaction,
)


class ClassifyCommandRiskTests(unittest.TestCase):
    def test_allowlisted_inspection_command_is_read_only(self):
        self.assertEqual(classify_command_risk("ls -la"), RISK_READ_ONLY)

    def test_inspection_program_write_or_exec_flags_are_not_read_only(self):
        for command in (
            "find . -delete",
            "find . -fprintf output.txt %p",
            "rg --pre 'touch owned' pattern",
            "sort -o output.txt input.txt",
            "tree --output=tree.txt",
            "git branch -D feature",
            "git diff --ext-diff",
            "sed -n 'e touch owned' file.txt",
        ):
            with self.subTest(command=command):
                self.assertNotEqual(classify_command_risk(command), RISK_READ_ONLY)
        self.assertEqual(classify_command_risk("pytest -q"), RISK_MEDIUM)

    def test_rm_rf_is_dangerous(self):
        self.assertEqual(classify_command_risk("rm -rf /tmp/scratch"), RISK_DANGEROUS)
        self.assertEqual(classify_command_risk("rm -fr build/"), RISK_DANGEROUS)

    def test_rm_rf_variants_with_separated_or_long_flags_are_dangerous(self):
        # Regression: the original single regex only matched combined short
        # flags (-rf/-fr and letter-order variants of *that*), so it missed
        # every one of these equally common spellings -- confirmed live,
        # all silently classified as only "medium" before the fix.
        self.assertEqual(classify_command_risk("rm -r -f /tmp/x"), RISK_DANGEROUS)
        self.assertEqual(classify_command_risk("rm -f -r /tmp/x"), RISK_DANGEROUS)
        self.assertEqual(classify_command_risk("rm --recursive --force /tmp/x"), RISK_DANGEROUS)
        self.assertEqual(classify_command_risk("rm -r --force /tmp/x"), RISK_DANGEROUS)
        self.assertEqual(classify_command_risk("rm --force -r /tmp/x"), RISK_DANGEROUS)

    def test_rm_without_both_recursive_and_force_stays_medium(self):
        # Only the r+f combination is dangerous -- unchanged threshold.
        self.assertEqual(classify_command_risk("rm file.txt"), RISK_MEDIUM)
        self.assertEqual(classify_command_risk("rm -f file.txt"), RISK_MEDIUM)
        self.assertEqual(classify_command_risk("rm -r somedir"), RISK_MEDIUM)

    def test_rm_lookalike_words_are_not_false_positives(self):
        # \brm\b must not match "confirm"/"term"/etc.
        self.assertEqual(classify_command_risk("confirm -r -f something"), RISK_MEDIUM)

    def test_force_push_is_dangerous(self):
        self.assertEqual(classify_command_risk("git push --force origin main"), RISK_DANGEROUS)

    def test_sudo_is_dangerous(self):
        self.assertEqual(classify_command_risk("sudo apt install foo"), RISK_DANGEROUS)

    def test_curl_pipe_shell_is_dangerous(self):
        self.assertEqual(classify_command_risk("curl https://example.com/install.sh | sh"), RISK_DANGEROUS)

    def test_ssh_key_access_is_dangerous(self):
        self.assertEqual(classify_command_risk("cat ~/.ssh/id_rsa"), RISK_DANGEROUS)

    def test_empty_command_is_medium_not_crash(self):
        self.assertEqual(classify_command_risk(""), RISK_MEDIUM)


class RedactSecretsTests(unittest.TestCase):
    """Confirmed live: a model that just read a live DB password out of
    wp-config.php pasted it straight into an inline `mysql -pSECRET ...`
    invocation, which the approval panel and status line then echoed
    verbatim, in cleartext, to the terminal."""

    def test_mysql_inline_dash_p_password_is_masked(self):
        redacted = redact_secrets("mysql -u finima -pN59AMQgtr8GUUFzL5huO -h 127.0.0.1 finima -e \"SELECT 1\"")
        self.assertNotIn("N59AMQgtr8GUUFzL5huO", redacted)
        self.assertIn("-p***", redacted)
        self.assertIn("-u finima", redacted)  # unrelated flags stay intact

    def test_long_password_flag_is_masked_for_any_command(self):
        self.assertNotIn("hunter2", redact_secrets("mysqldump -u root --password=hunter2 finima"))
        self.assertNotIn("hunter2", redact_secrets("pg_restore --password hunter2 dump.sql"))

    def test_url_embedded_credentials_are_masked(self):
        redacted = redact_secrets("git clone https://user:token123@github.com/org/repo.git")
        self.assertNotIn("token123", redacted)
        self.assertIn("github.com/org/repo.git", redacted)

    def test_long_dashed_flags_are_not_mistaken_for_dash_p(self):
        """The bare `-p` password pattern must not fire inside `--password`
        itself (which contains the literal substring "-p") or unrelated
        long flags -- only a standalone `-p<value>` token."""
        self.assertEqual(
            redact_secrets("wp theme list --status=active --format=csv --fields=name --allow-root"),
            "wp theme list --status=active --format=csv --fields=name --allow-root",
        )

    def test_plain_command_without_secrets_is_unchanged(self):
        self.assertEqual(redact_secrets("systemctl restart caddy"), "systemctl restart caddy")

    def test_empty_command_does_not_crash(self):
        self.assertEqual(redact_secrets(""), "")


class ClassifyPathRiskTests(unittest.TestCase):
    def test_path_inside_workspace_is_medium(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(classify_path_risk("subdir/file.py", root), RISK_MEDIUM)
            self.assertEqual(classify_path_risk(str(Path(root) / "file.py"), root), RISK_MEDIUM)

    def test_path_escaping_workspace_is_dangerous(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(classify_path_risk("../../etc/passwd", root), RISK_DANGEROUS)
            self.assertEqual(classify_path_risk("/etc/passwd", root), RISK_DANGEROUS)

    def test_workspace_root_itself_is_medium_not_dangerous(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(classify_path_risk(".", root), RISK_MEDIUM)


class ClassifyToolCallRiskTests(unittest.TestCase):
    def test_read_only_tools_are_always_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("read_file", "list_directory", "search_code", "get_git_info", "ask_user_question"):
                self.assertEqual(classify_tool_call_risk(name, {"path": "x"}, workspace_root=root), RISK_READ_ONLY)

    def test_write_file_in_workspace_is_medium(self):
        with tempfile.TemporaryDirectory() as root:
            risk = classify_tool_call_risk("write_file", {"path": "new.py", "content": "x"}, workspace_root=root)
            self.assertEqual(risk, RISK_MEDIUM)

    def test_write_file_missing_path_is_dangerous(self):
        with tempfile.TemporaryDirectory() as root:
            risk = classify_tool_call_risk("write_file", {}, workspace_root=root)
            self.assertEqual(risk, RISK_DANGEROUS)

    def test_archive_outputs_are_approval_gated_and_workspace_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                classify_tool_call_risk(
                    "extract_archive", {"path": "upload.zip", "destination": "expanded"},
                    workspace_root=root,
                ),
                RISK_MEDIUM,
            )
            self.assertEqual(
                classify_tool_call_risk(
                    "repackage_archive", {"source_dir": "expanded", "output_path": "/tmp/leak.zip"},
                    workspace_root=root,
                ),
                RISK_DANGEROUS,
            )

    def test_execute_command_uses_command_classifier(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                classify_tool_call_risk("execute_command", {"command": "rm -rf /"}, workspace_root=root),
                RISK_DANGEROUS,
            )
            self.assertEqual(
                classify_tool_call_risk("execute_command", {"command": "ls"}, workspace_root=root),
                RISK_READ_ONLY,
            )

    def test_unknown_tool_fails_safe_to_dangerous(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(classify_tool_call_risk("mystery_tool", {}, workspace_root=root), RISK_DANGEROUS)


class MutationLedgerTests(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()

    def test_record_mutation_computes_diff_and_line_counts(self):
        entry = record_mutation(
            1, path="/tmp/x.py", operation="update",
            original_content="line1\nline2\n", new_content="line1\nline2 changed\nline3\n",
        )
        self.assertIn("mutation_id", entry)
        self.assertGreaterEqual(entry["lines_added"], 1)
        self.assertGreaterEqual(entry["lines_removed"], 1)
        self.assertIn("+line3", entry["unified_diff"])

        state = state_module.get_session_state(1)
        self.assertEqual(state.modified_files[-1]["mutation_id"], entry["mutation_id"])

    def test_revert_mutation_restores_prior_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "restore_me.py"
            target.write_text("new content\n")
            entry = record_mutation(
                1, path=str(target), operation="update",
                original_content="old content\n", new_content="new content\n",
            )
            revert_mutation(1, entry["mutation_id"])
            self.assertEqual(target.read_text(), "old content\n")

            state = state_module.get_session_state(1)
            reverted = next(m for m in state.modified_files if m["mutation_id"] == entry["mutation_id"])
            self.assertEqual(reverted["revert_status"], "reverted")

    def test_revert_mutation_that_created_file_deletes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "created.py"
            target.write_text("brand new\n")
            entry = record_mutation(
                1, path=str(target), operation="create",
                original_content=None, new_content="brand new\n",
            )
            revert_mutation(1, entry["mutation_id"])
            self.assertFalse(target.exists())

    def test_revert_unknown_mutation_id_raises(self):
        with self.assertRaises(ValueError):
            revert_mutation(1, "mut_does_not_exist")

    def test_reverting_twice_is_a_noop_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.py"
            target.write_text("new\n")
            entry = record_mutation(1, path=str(target), operation="update", original_content="old\n", new_content="new\n")
            revert_mutation(1, entry["mutation_id"])
            result = revert_mutation(1, entry["mutation_id"])  # should not raise or re-delete/overwrite
            self.assertEqual(result["revert_status"], "reverted")
            self.assertEqual(target.read_text(), "old\n")


class RevertTransactionTests(unittest.TestCase):
    """A multi-file turn used to only be revertible one mutation_id at a
    time, with no way to even discover which ids belonged together --
    these cover revert_transaction, which groups mutations by the shared
    transaction_id MCPServer now mints once per turn."""

    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()

    def _record(self, path, *, original, new, transaction_id, created_at=None):
        entry = record_mutation(
            1, path=str(path), operation="create" if original is None else "update",
            original_content=original, new_content=new, transaction_id=transaction_id,
        )
        if created_at is not None:
            state = state_module.get_session_state(1)
            for item in state.modified_files:
                if item["mutation_id"] == entry["mutation_id"]:
                    item["created_at"] = created_at
            state_module.save_session_state(1, modified_files=state.modified_files)
        return entry

    def test_reverts_every_mutation_in_the_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.py", Path(tmp) / "b.py"
            a.write_text("new a\n")
            b.write_text("new b\n")
            self._record(a, original="old a\n", new="new a\n", transaction_id="turn_abc", created_at="2026-01-01T00:00:00")
            self._record(b, original="old b\n", new="new b\n", transaction_id="turn_abc", created_at="2026-01-01T00:00:01")

            result = revert_transaction(1, "turn_abc")

            self.assertEqual(len(result["reverted"]), 2)
            self.assertEqual(result["remaining"], [])
            self.assertIsNone(result["error"])
            self.assertEqual(a.read_text(), "old a\n")
            self.assertEqual(b.read_text(), "old b\n")

    def test_reverts_same_file_edits_in_reverse_chronological_order(self):
        # Two sequential edits to the SAME file within one turn: reverting
        # in the wrong order would leave the file at the wrong content, or
        # fail outright (the second entry's "original_content" is the
        # first edit's result, not the file's true original).
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.py"
            target.write_text("v3\n")
            self._record(target, original="v1\n", new="v2\n", transaction_id="turn_seq", created_at="2026-01-01T00:00:00")
            self._record(target, original="v2\n", new="v3\n", transaction_id="turn_seq", created_at="2026-01-01T00:00:01")

            result = revert_transaction(1, "turn_seq")

            self.assertEqual(len(result["reverted"]), 2)
            self.assertEqual(target.read_text(), "v1\n")

    def test_unrelated_mutations_outside_the_transaction_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_turn = Path(tmp) / "in.py"
            other = Path(tmp) / "other.py"
            in_turn.write_text("new\n")
            other.write_text("unrelated new\n")
            self._record(in_turn, original="old\n", new="new\n", transaction_id="turn_x")
            self._record(other, original="unrelated old\n", new="unrelated new\n", transaction_id="turn_y")

            revert_transaction(1, "turn_x")

            self.assertEqual(in_turn.read_text(), "old\n")
            self.assertEqual(other.read_text(), "unrelated new\n")

    def test_unknown_transaction_id_raises(self):
        with self.assertRaises(ValueError):
            revert_transaction(1, "turn_does_not_exist")

    def test_already_fully_reverted_transaction_returns_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.py"
            target.write_text("new\n")
            self._record(target, original="old\n", new="new\n", transaction_id="turn_done")
            revert_transaction(1, "turn_done")

            result = revert_transaction(1, "turn_done")
            self.assertEqual(result["reverted"], [])
            self.assertIsNone(result["error"])

    def test_stops_and_reports_remaining_on_first_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.py"
            good.write_text("new good\n")
            good_entry = self._record(good, original="old good\n", new="new good\n", transaction_id="turn_partial", created_at="2026-01-01T00:00:01")
            bad_entry = self._record(
                Path(tmp) / "bad.py", original="old bad\n", new="new bad\n",
                transaction_id="turn_partial", created_at="2026-01-01T00:00:02",  # most recent -> reverted first
            )

            def side_effect(session_id, mutation_id):
                if mutation_id == bad_entry["mutation_id"]:
                    raise OSError("simulated failure")

            with patch("tamfis_code.safety.revert_mutation", side_effect=side_effect):
                result = revert_transaction(1, "turn_partial")

            self.assertEqual(result["reverted"], [])
            self.assertIn(bad_entry["mutation_id"], result["remaining"])
            self.assertIn(good_entry["mutation_id"], result["remaining"])
            self.assertIn("simulated failure", result["error"])


if __name__ == "__main__":
    unittest.main()
