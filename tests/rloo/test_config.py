from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from ksllm4rec_rloo.config import approved_config, load_config, validate_config
from ksllm4rec_rloo.contract import GROUP_SIZE, REWARD_VALUES, SFT_ADAPTER


class ConfigContractTest(unittest.TestCase):
    def test_exact_approved_config_loads(self) -> None:
        value = approved_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approved.yaml"
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
            loaded = load_config(path)

        self.assertEqual(loaded["model"]["sft_adapter"], str(SFT_ADAPTER))
        self.assertEqual(loaded["model"]["lora_rank"], 64)
        self.assertEqual(loaded["rollout"]["num_generations"], GROUP_SIZE)
        self.assertEqual(loaded["reward"]["same_ab"], REWARD_VALUES["same_ab"])
        self.assertFalse(loaded["reward"]["divide_by_std"])
        self.assertTrue(loaded["loss"]["reference_free"])

    def test_reference_forced_gt_and_kl_fields_are_rejected(self) -> None:
        for key, value in (
            ("reference_adapter", "reference"),
            ("reference_beta", 0.02),
            ("forced_gt", True),
            ("add_gt_probability", 0.5),
            ("kl_beta", 0.1),
        ):
            with self.subTest(key=key):
                config = approved_config()
                config["loss"][key] = value
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    validate_config(config)

    def test_reward_or_group_size_drift_is_rejected(self) -> None:
        changed_reward = approved_config()
        changed_reward["reward"]["same_a"] = 0.05
        with self.assertRaisesRegex(ValueError, "reward.same_a"):
            validate_config(changed_reward)

        changed_group = approved_config()
        changed_group["rollout"]["num_generations"] = 8
        with self.assertRaisesRegex(ValueError, "rollout.num_generations"):
            validate_config(changed_group)

    def test_unknown_and_missing_keys_are_rejected(self) -> None:
        unknown = approved_config()
        unknown["reward"]["temperature_magic"] = 3
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_config(unknown)

        missing = approved_config()
        del missing["anchor"]["max_weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_config(missing)

    def test_non_mapping_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
