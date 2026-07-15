from __future__ import annotations

import unittest

from omegaconf import OmegaConf

from ksllm4rec_orpo.contract import validate_training_contract


class OrpoContractTest(unittest.TestCase):
    def setUp(self) -> None:
        config = OmegaConf.to_container(
            OmegaConf.load("configs/orpo/onereason_lora_orpo.yaml"), resolve=True
        )
        self.custom = dict(config.pop("custom_orpo"))
        self.config = config

    def test_accepts_the_approved_config(self) -> None:
        report = validate_training_contract(
            self.config, self.custom, "config_check"
        )
        self.assertTrue(report["clean_base_required"])
        self.assertEqual(report["epochs"], 2.0)

    def test_rejects_preloaded_adapter(self) -> None:
        self.config["adapter_name_or_path"] = "/tmp/adapter"
        with self.assertRaisesRegex(ValueError, "clean base"):
            validate_training_contract(self.config, self.custom, "config_check")

    def test_rejects_reference_based_dpo(self) -> None:
        self.config["pref_loss"] = "sigmoid"
        with self.assertRaisesRegex(ValueError, "pref_loss"):
            validate_training_contract(self.config, self.custom, "config_check")

    def test_rejects_one_epoch(self) -> None:
        self.config["num_train_epochs"] = 1.0
        with self.assertRaisesRegex(ValueError, "num_train_epochs"):
            validate_training_contract(self.config, self.custom, "config_check")


if __name__ == "__main__":
    unittest.main()
