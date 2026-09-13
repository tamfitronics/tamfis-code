"""Regression tests for runner_local.py's vision/image-attachment support.
Closes the Codex view_image.rs parity gap -- the feature clearly exists
(is_vision_image_path, build_vision_content_blocks, and
_messages_with_vision_content are all real, and runner_local.py's own
docstring for run_local_agent_turn references them for splicing
image_url content blocks into the most recent user message for routes
that support vision) but had no dedicated test file anywhere in the suite.
"""
import tempfile
import unittest
from pathlib import Path

from tamfis_code.runner_local import (
    MAX_VISION_ATTACHMENT_BYTES,
    _messages_with_vision_content,
    build_vision_content_blocks,
    is_vision_image_path,
)


class IsVisionImagePathTests(unittest.TestCase):
    def test_recognised_image_extensions_are_true(self):
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            self.assertTrue(is_vision_image_path(f"/tmp/pic{ext}"))

    def test_case_insensitive(self):
        self.assertTrue(is_vision_image_path("/tmp/PIC.PNG"))

    def test_non_image_extensions_are_false(self):
        for ext in (".pdf", ".docx", ".py", ".txt", ""):
            self.assertFalse(is_vision_image_path(f"/tmp/file{ext}"))


class BuildVisionContentBlocksTests(unittest.TestCase):
    def test_a_real_image_becomes_a_base64_data_uri_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "pic.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
            blocks = build_vision_content_blocks([str(img)])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "image_url")
        self.assertTrue(blocks[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_a_non_image_path_is_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "doc.pdf"
            doc.write_bytes(b"%PDF-1.4")
            blocks = build_vision_content_blocks([str(doc)])
        self.assertEqual(blocks, [])

    def test_a_missing_path_is_silently_skipped_not_raised(self):
        blocks = build_vision_content_blocks(["/definitely/does/not/exist.png"])
        self.assertEqual(blocks, [])

    def test_an_oversized_image_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.jpg"
            big.write_bytes(b"0" * (MAX_VISION_ATTACHMENT_BYTES + 10))
            blocks = build_vision_content_blocks([str(big)])
        self.assertEqual(blocks, [])

    def test_mixed_paths_only_keep_the_real_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").write_bytes(b"pngdata")
            (root / "b.pdf").write_bytes(b"pdfdata")
            (root / "c.jpg").write_bytes(b"jpgdata")
            blocks = build_vision_content_blocks([
                str(root / "a.png"), str(root / "b.pdf"), str(root / "c.jpg"),
            ])
        self.assertEqual(len(blocks), 2)


class MessagesWithVisionContentTests(unittest.TestCase):
    def test_splices_image_blocks_into_the_target_user_message(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "look at this"},
        ]
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}]
        patched = _messages_with_vision_content(messages, 1, blocks)
        self.assertEqual(patched[1]["content"][0], {"type": "text", "text": "look at this"})
        self.assertEqual(patched[1]["content"][1], blocks[0])

    def test_does_not_mutate_the_original_messages_list_or_message(self):
        messages = [{"role": "user", "content": "look at this"}]
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}]
        _messages_with_vision_content(messages, 0, blocks)
        self.assertEqual(messages[0]["content"], "look at this")

    def test_no_blocks_returns_the_same_messages_untouched(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertIs(_messages_with_vision_content(messages, 0, None), messages)
        self.assertIs(_messages_with_vision_content(messages, 0, []), messages)

    def test_no_vision_message_index_returns_the_same_messages_untouched(self):
        messages = [{"role": "user", "content": "hi"}]
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}]
        self.assertIs(_messages_with_vision_content(messages, None, blocks), messages)

    def test_a_non_user_target_message_is_left_untouched(self):
        messages = [{"role": "assistant", "content": "ok"}]
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}]
        result = _messages_with_vision_content(messages, 0, blocks)
        self.assertEqual(result, messages)


if __name__ == "__main__":
    unittest.main()
