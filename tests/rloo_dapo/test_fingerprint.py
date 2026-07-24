from __future__ import annotations

import unittest
from pathlib import Path

from ksllm4rec_rloo_dapo.fingerprint import (
    runtime_code_fingerprint,
    validate_runtime_signature,
)


class FingerprintTest(unittest.TestCase):
    def test_explicit_runtime_files_are_content_addressed(self) -> None:
        root = Path(__file__).resolve().parents[2]
        result = runtime_code_fingerprint(
            root,
            runtime_code_files=["src/ksllm4rec_rloo_dapo/contract.py"],
        )
        self.assertEqual(
            set(result["files"]), {"src/ksllm4rec_rloo_dapo/contract.py"}
        )
        self.assertEqual(len(result["sha256"]), 64)

    def test_self_inconsistent_signature_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "self-inconsistent"):
            validate_runtime_signature({"sha256": "0" * 64, "inputs": {}})


if __name__ == "__main__":
    unittest.main()
