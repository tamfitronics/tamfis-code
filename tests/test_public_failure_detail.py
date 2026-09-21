"""Failure text shown to users must not expose backend internals (GPU/engine/stack-trace detail).

Live report 2026-09-21: "Task failed: TamfisGPT-Pro streaming failed: Engine loop is not running. Inspect the
stacktrace to find the original error: OutOfMemoryError('CUDA out of memory. Tried to allocate 1.74 GiB. GPU 0
has a total capacity of 47.37 GiB ...".
"""
import unittest
from types import SimpleNamespace

from tamfis_code.runner_local import _public_failure_detail

MANAGER = SimpleNamespace(provider_error_status=lambda error: getattr(error, "status_code", None))
LEAK_WORDS = ("CUDA", "GPU", "GiB", "OutOfMemory", "Engine loop", "stacktrace", "vllm", "torch", "NCCL")


class PublicFailureDetailTests(unittest.TestCase):
    def test_a_gpu_crash_is_reported_in_plain_words(self):
        error = RuntimeError(
            "Engine loop is not running. Inspect the stacktrace to find the original error: "
            "OutOfMemoryError('CUDA out of memory. Tried to allocate 1.74 GiB. GPU 0 has a total capacity of "
            "47.37 GiB of which 773.88 MiB is free. Including non-PyTorch memory, this process has 46.60 GiB "
            "memory in use.')"
        )
        detail = _public_failure_detail(MANAGER, error)
        for word in LEAK_WORDS:
            self.assertNotIn(word.lower(), detail.lower(), word)
        self.assertIn("capacity", detail)

    def test_other_internals_are_hidden_too(self):
        for text in ("Traceback (most recent call last): File vllm/engine.py", "NCCL error: unhandled system error",
                     "torch.cuda.OutOfMemoryError", "device-side assert triggered"):
            detail = _public_failure_detail(MANAGER, RuntimeError(text))
            for word in LEAK_WORDS + ("Traceback", "device-side"):
                self.assertNotIn(word.lower(), detail.lower(), (text, word))

    def test_ordinary_failures_keep_their_existing_wording(self):
        self.assertIn("busy", _public_failure_detail(MANAGER, SimpleNamespace(status_code=429)))
        self.assertIn("credit", _public_failure_detail(MANAGER, SimpleNamespace(status_code=402)))
        self.assertEqual(_public_failure_detail(MANAGER, ValueError("bad arguments")), "bad arguments")


if __name__ == "__main__":
    unittest.main()
