from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_rloo_dapo import contract
from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.fingerprint import (
    discover_runtime_code_files,
    runtime_code_fingerprint,
    runtime_signature,
    validate_runtime_signature,
)


class FingerprintTest(unittest.TestCase):
    def test_explicit_runtime_files_are_content_addressed(self) -> None:
        root = Path(__file__).resolve().parents[2]
        result = runtime_code_fingerprint(
            root,
            runtime_code_files=["src/ksllm4rec_rloo_dapo/contract.py"],
        )
        self.assertEqual(set(result["files"]), {"src/ksllm4rec_rloo_dapo/contract.py"})
        self.assertEqual(len(result["sha256"]), 64)

    def test_self_inconsistent_signature_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "self-inconsistent"):
            validate_runtime_signature({"sha256": "0" * 64, "inputs": {}})

    def test_default_runtime_files_include_all_five_cpu_suites(self) -> None:
        root = Path(__file__).resolve().parents[2]
        relative = {
            path.relative_to(root).as_posix()
            for path in discover_runtime_code_files(root)
        }
        for suite in ("rloo_dapo", "rloo", "sft", "grpo", "orpo"):
            with self.subTest(suite=suite):
                prefix = f"tests/{suite}/test_"
                self.assertTrue(any(path.startswith(prefix) for path in relative))
        self.assertNotIn(
            "scripts/rloo_dapo/"
            "frontier_sft_lora64_lr1p5em4_wd1em3_step372_t12/status.sh",
            relative,
        )

    def test_runtime_signature_binds_the_frozen_execution_target(self) -> None:
        file_record = {"path": "/frozen/input", "size": 1, "sha256": "1" * 64}
        with (
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.frozen_model_inputs",
                return_value={"model": "frozen"},
            ),
            patch(
                "ksllm4rec_rloo_dapo.fingerprint._file",
                return_value=file_record,
            ),
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.snapshot_directory",
                return_value={"trie.bin": {"size": 1, "sha256": "2" * 64}},
            ),
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.software_versions",
                return_value={"python": "test"},
            ),
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.runtime_code_fingerprint",
                return_value={"sha256": "3" * 64, "files": {}},
            ),
        ):
            signature = runtime_signature(approved_config(contract.SFT372_PROFILE))
        self.assertEqual(
            signature["inputs"]["execution_target"],
            {
                "device": contract.EXECUTION_DEVICE,
                "cuda_visible_devices": contract.EXECUTION_CUDA_VISIBLE_DEVICES,
                "gpu_identity": dict(contract.EXPECTED_GPU_IDENTITY),
            },
        )
        validate_runtime_signature(signature)

    def test_legacy_signature_does_not_claim_a_frozen_gpu(self) -> None:
        file_record = {"path": "/frozen/input", "size": 1, "sha256": "1" * 64}
        with (
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.frozen_model_inputs",
                return_value={"model": "frozen"},
            ),
            patch("ksllm4rec_rloo_dapo.fingerprint._file", return_value=file_record),
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.snapshot_directory",
                return_value={},
            ),
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.software_versions",
                return_value={},
            ),
            patch(
                "ksllm4rec_rloo_dapo.fingerprint.runtime_code_fingerprint",
                return_value={"sha256": "3" * 64, "files": {}},
            ),
        ):
            signature = runtime_signature(approved_config(contract.PROFILE))
        self.assertNotIn("execution_target", signature["inputs"])


if __name__ == "__main__":
    unittest.main()
