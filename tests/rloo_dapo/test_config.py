from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ksllm4rec_rloo_dapo.config import approved_config, load_config, validate_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rloo/frontier_sft_epoch2_lora64_g16_rloo_dapo_t12_k1_kvcache.yaml"


class ConfigTest(unittest.TestCase):
    def test_yaml_is_the_exact_approved_contract(self) -> None:
        self.assertEqual(load_config(CONFIG), approved_config())

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
