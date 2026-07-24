from __future__ import annotations

import unittest

from transformers import AutoTokenizer

from ksllm4rec_grpo.prompt import encode_prompt
from ksllm4rec_rloo.integrity import sha256_file
from ksllm4rec_rloo.modeling import validate_adapter_contract
from ksllm4rec_rloo_dapo import contract
from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.gates import expected_input_sha256


class FrozenProfileTest(unittest.TestCase):
    def test_profile_paths_are_isolated_and_frozen(self) -> None:
        original = contract.frozen_profile(contract.PROFILE)
        sft372 = contract.frozen_profile(contract.SFT372_PROFILE)
        self.assertNotEqual(original.config_path, sft372.config_path)
        self.assertNotEqual(original.pilot_dir, sft372.pilot_dir)
        self.assertNotEqual(original.run_dir, sft372.run_dir)
        self.assertNotEqual(original.log_dir, sft372.log_dir)
        for profile in (original, sft372):
            with self.subTest(profile=profile.name):
                self.assertTrue(profile.config_path.is_file())
                self.assertEqual(
                    profile.config_path.parent,
                    contract.PROJECT_ROOT / "configs" / "rloo",
                )
                self.assertEqual(
                    profile.pilot_dir.parent,
                    contract.PROJECT_ROOT / "artifacts" / "rloo" / "pilots",
                )

    def test_gpu_identity_is_an_explicit_immutable_contract(self) -> None:
        self.assertEqual(
            dict(contract.EXPECTED_GPU_IDENTITY),
            {
                "device": "cuda:0",
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": "e1f24bb9-9bcb-c31f-52ba-fb5f63a146d5",
                "total_memory": 25_756_696_576,
                "compute_capability": [8, 9],
            },
        )
        self.assertEqual(contract.EXECUTION_DEVICE, "cuda:0")
        self.assertEqual(contract.EXECUTION_CUDA_VISIBLE_DEVICES, "0")
        with self.assertRaises(TypeError):
            contract.EXPECTED_GPU_IDENTITY["name"] = "other"  # type: ignore[index]

    def test_sft372_files_match_the_frozen_identity(self) -> None:
        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        self.assertEqual(
            sha256_file(profile.sft_adapter / "adapter_model.safetensors"),
            profile.sft_adapter_sha256,
        )
        self.assertEqual(
            sha256_file(profile.sft_adapter / "adapter_config.json"),
            profile.sft_config_sha256,
        )
        self.assertEqual(
            sha256_file(profile.tokenizer / "tokenizer.json"),
            profile.tokenizer_sha256,
        )
        self.assertEqual(
            sha256_file(profile.tokenizer / "tokenizer_config.json"),
            profile.tokenizer_config_sha256,
        )
        self.assertEqual(
            expected_input_sha256(approved_config(contract.SFT372_PROFILE))[
                "sft_adapter"
            ],
            profile.sft_adapter_sha256,
        )

    def test_sft372_adapter_matches_the_shared_lora_contract(self) -> None:
        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        report = validate_adapter_contract(profile.sft_adapter)
        self.assertEqual(report.rank, 64)
        self.assertEqual(report.alpha, 64)
        self.assertEqual(report.source_dropout, 0.05)
        self.assertEqual(report.target_modules, contract.LORA_TARGET_MODULES)

    def test_sft372_tokenizer_is_behaviorally_equal_to_original(self) -> None:
        original = contract.frozen_profile(contract.PROFILE)
        sft372 = contract.frozen_profile(contract.SFT372_PROFILE)
        left = AutoTokenizer.from_pretrained(original.tokenizer, local_files_only=True)
        right = AutoTokenizer.from_pretrained(sft372.tokenizer, local_files_only=True)
        system = "You are a recommendation engine."
        prompt = "Choose the next item from the user's history."
        self.assertEqual(
            encode_prompt(left, system, prompt, cutoff_len=16_384),
            encode_prompt(right, system, prompt, cutoff_len=16_384),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        self.assertEqual(
            left.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            ),
            right.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            ),
        )


if __name__ == "__main__":
    unittest.main()
