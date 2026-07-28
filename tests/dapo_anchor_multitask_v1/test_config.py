from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import yaml

from ksllm4rec_dapo_anchor_multitask_v1._infra.rloo_integrity import canonical_sha256
from ksllm4rec_dapo_anchor_multitask_v1.config import build_config, validate_config
from ksllm4rec_dapo_anchor_multitask_v1.trainer import _calibration_report


class ConfigContractTest(unittest.TestCase):
    def test_default_config_exposes_the_confirmed_multitask_contract(self) -> None:
        config = build_config()

        self.assertEqual(config["spec_version"], "DAPO-ANCHOR-MULTITASK-V1.0")
        self.assertEqual(config["profile"], "frontier_sft372_dapo_anchor_multitask_v1")
        self.assertEqual(config["data"]["recommendation_groups"], 17_016)
        self.assertEqual(config["data"]["text_to_sid_groups"], 10_597)
        self.assertEqual(config["data"]["source_blocks"], 532)
        self.assertEqual(config["rollout"]["num_generations"], 16)
        self.assertEqual(config["rollout"]["temperature"], 1.2)
        self.assertEqual(
            config["reward"],
            {
                "exact": 1.0,
                "same_ab": 0.15,
                "same_a": 0.05,
                "same_domain": 0.01,
                "other_domain": 0.0,
            },
        )
        self.assertEqual(config["sampling"]["overflow"], "retain_all")
        self.assertFalse(config["sampling"]["replenishment"])
        self.assertFalse(config["anchor"]["independent_optimizer_step"])
        self.assertEqual(config["train"]["source_epochs"], 1)
        self.assertEqual(config["train"]["scheduler_step_unit"], "policy_optimizer_step")
        self.assertEqual(config["experiments"]["arms"], ["A", "B", "C"])
        validate_config(config)

    def test_config_rejects_old_output_and_semantic_overrides(self) -> None:
        config = build_config(output_run_dir=Path("/tmp/multitask-new-run"))
        config["reward"]["same_ab"] = 0.4
        with self.assertRaisesRegex(ValueError, "reward.same_ab"):
            validate_config(config, allow_output_override=True)

    def test_checked_in_yaml_is_exactly_the_default_config(self) -> None:
        root = Path("/home/lyc/REC_PROJECTS/KSLLM4REC/configs/rloo")
        for arm in ("A", "B", "C"):
            suffix = "" if arm == "C" else f"_arm_{arm.lower()}"
            path = root / (
                f"frontier_sft372_g16_dapo_anchor_multitask_v1{suffix}.yaml"
            )
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, build_config(arm=arm))

    def test_experiment_arms_have_independent_state_paths(self) -> None:
        configs = {arm: build_config(arm=arm) for arm in ("A", "B", "C")}

        for key_path in (
            ("anchor", "calibration_path"),
            ("output", "log_dir"),
            ("output", "run_dir"),
        ):
            values = {
                configs[arm][key_path[0]][key_path[1]] for arm in configs
            }
            self.assertEqual(len(values), 3)

    def test_calibration_report_rejects_a_silent_zero_anchor_weight(self) -> None:
        config = build_config()
        inputs = {"spec": "test-calibration"}
        signature = {"inputs": inputs, "sha256": canonical_sha256(inputs)}
        report = {
            "schema_version": 1,
            "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
            "base_signature_sha256": signature["sha256"],
            "source_blocks": 16,
            "policy_optimizer_steps": 0,
            "model_unchanged": True,
            "rl_gradient_norms": [0.5, 1.5],
            "rl_reference_gradient_norm": 1.0,
            "anchor_raw_gradient_norm": 2.0,
            "lambda_calibrated": 0.05,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor_calibration.json"
            config["anchor"]["calibration_path"] = str(path)
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                _calibration_report(config, signature)["lambda_calibrated"], 0.05
            )
            report["lambda_calibrated"] = 0.0
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside the frozen range"):
                _calibration_report(config, signature)


if __name__ == "__main__":
    unittest.main()
