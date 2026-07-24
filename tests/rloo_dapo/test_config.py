from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ksllm4rec_rloo_dapo import contract
from ksllm4rec_rloo_dapo.config import approved_config, load_config, validate_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "configs/rloo/frontier_sft_epoch2_lora64_g16_rloo_dapo_t12_k1_kvcache.yaml"
)
SFT372_CONFIG = ROOT / (
    "configs/rloo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_"
    "g16_rloo_dapo_t12_k1_kvcache.yaml"
)


class ConfigTest(unittest.TestCase):
    def test_yaml_is_the_exact_approved_contract(self) -> None:
        self.assertEqual(load_config(CONFIG), approved_config())

    def test_sft372_yaml_is_the_exact_approved_contract(self) -> None:
        self.assertEqual(
            load_config(SFT372_CONFIG), approved_config(contract.SFT372_PROFILE)
        )

    def test_profiles_have_isolated_model_and_output_paths(self) -> None:
        old = approved_config()
        new = approved_config(contract.SFT372_PROFILE)
        self.assertNotEqual(old["model"]["sft_adapter"], new["model"]["sft_adapter"])
        self.assertNotEqual(old["model"]["tokenizer"], new["model"]["tokenizer"])
        self.assertNotEqual(old["output"]["run_dir"], new["output"]["run_dir"])
        self.assertNotEqual(old["output"]["log_dir"], new["output"]["log_dir"])

    def test_cross_profile_adapter_mix_is_rejected(self) -> None:
        value = approved_config(contract.SFT372_PROFILE)
        value["model"]["sft_adapter"] = approved_config()["model"]["sft_adapter"]
        with self.assertRaisesRegex(ValueError, "mismatches"):
            validate_config(value)

    def test_unknown_profile_is_rejected(self) -> None:
        value = approved_config()
        value["profile"] = "arbitrary-adapter"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_config(value)

    def test_anchor_key_is_forbidden(self) -> None:
        value = approved_config()
        value["anchor"] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_config(value)

    def test_every_behavioral_change_is_rejected(self) -> None:
        changes = (
            ("rollout", "temperature", 1.0),
            ("rollout", "max_active_sequences", 8),
            ("sampling", "effective_groups_per_window", 8),
            ("loss", "clip_ratio_high", 1.2),
            ("train", "scheduler", "cosine"),
        )
        for section, key, replacement in changes:
            with self.subTest(section=section, key=key):
                value = copy.deepcopy(approved_config())
                value[section][key] = replacement
                with self.assertRaisesRegex(ValueError, "mismatches"):
                    validate_config(value)


if __name__ == "__main__":
    unittest.main()
