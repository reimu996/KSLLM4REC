from __future__ import annotations

from pathlib import Path
import unittest
import yaml

from ksllm4rec_dapo_anchor_multitask_v1_1 import contract
from ksllm4rec_dapo_anchor_multitask_v1_1.config import build_config, validate_config


class ConfigContractTest(unittest.TestCase):
    def test_default_config_exposes_the_confirmed_multitask_contract(self) -> None:
        config = build_config()

        self.assertEqual(config["spec_version"], contract.SPEC_VERSION)
        self.assertEqual(config["profile"], contract.PROFILE)
        self.assertEqual(config["data"]["recommendation_groups"], 17_016)
        self.assertEqual(config["data"]["text_to_sid_groups"], 10_597)
        self.assertEqual(config["data"]["source_blocks"], 532)
        self.assertEqual(contract.SOURCE_GROUPS_PER_EPOCH, 27_613)
        self.assertEqual(contract.TOTAL_SOURCE_GROUPS, 55_226)
        self.assertEqual(contract.TOTAL_SOURCE_BLOCKS, 1_064)
        self.assertEqual(contract.TOTAL_CANDIDATES, 883_616)
        self.assertEqual(config["rollout"]["num_generations"], 16)
        self.assertEqual(config["rollout"]["temperature"], 1.2)
        self.assertEqual(
            config["reward"],
            {
                "exact": 1.0,
                "same_ab": 0.40,
                "same_a": 0.15,
                "same_domain": 0.01,
                "other_domain": 0.0,
            },
        )
        self.assertEqual(config["sampling"]["overflow"], "retain_all")
        self.assertFalse(config["sampling"]["replenishment"])
        self.assertEqual(
            config["sampling"]["shuffle"], "sha256_seed_epoch_task_group_id"
        )
        self.assertTrue(config["sampling"]["epoch_boundary_flush"])
        self.assertFalse(config["anchor"]["independent_optimizer_step"])
        self.assertEqual(config["anchor"]["routes"], ["all_normalized_groups"])
        self.assertEqual(
            config["anchor"]["coverage"], "all_normalized_groups_once_per_epoch"
        )
        self.assertEqual(config["train"]["source_epochs"], 2)
        self.assertEqual(config["train"]["scheduler_step_unit"], "policy_optimizer_step")
        self.assertEqual(
            config["experiments"]["active"],
            {
                "arm": "D",
                "recommendation_enabled": True,
                "text_to_sid_enabled": True,
                "reward_profile": "old",
            },
        )
        self.assertEqual(config["experiments"]["arms"], ["A", "B", "C", "D"])
        validate_config(config)

    def test_config_rejects_old_output_and_semantic_overrides(self) -> None:
        config = build_config(output_run_dir=Path("/tmp/multitask-new-run"))
        config["reward"]["same_ab"] = 0.15
        with self.assertRaisesRegex(ValueError, "reward.same_ab"):
            validate_config(config, allow_output_override=True)

    def test_checked_in_yaml_is_exactly_the_default_config(self) -> None:
        root = Path("/home/lyc/REC_PROJECTS/KSLLM4REC/configs/rloo")
        for arm in ("A", "B", "C", "D"):
            suffix = "" if arm == "C" else f"_arm_{arm.lower()}"
            path = root / (
                f"frontier_sft372_g16_dapo_anchor_multitask_v1_1_e2{suffix}.yaml"
            )
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, build_config(arm=arm))

    def test_experiment_arms_have_independent_state_paths(self) -> None:
        configs = {arm: build_config(arm=arm) for arm in ("A", "B", "C", "D")}

        for key_path in (
            ("output", "log_dir"),
            ("output", "run_dir"),
        ):
            values = {
                configs[arm][key_path[0]][key_path[1]] for arm in configs
            }
            self.assertEqual(len(values), 4)

    def test_c_remains_new_reward_while_d_is_multitask_old_reward(self) -> None:
        c_config = build_config(arm="C")
        d_config = build_config(arm="D")

        self.assertEqual(c_config["experiments"]["active"]["reward_profile"], "new")
        self.assertEqual(c_config["reward"]["same_ab"], 0.15)
        self.assertTrue(c_config["experiments"]["active"]["text_to_sid_enabled"])
        self.assertEqual(d_config["experiments"]["active"]["reward_profile"], "old")
        self.assertEqual(d_config["reward"]["same_ab"], 0.40)
        self.assertEqual(d_config["reward"]["same_a"], 0.15)
        self.assertTrue(d_config["experiments"]["active"]["text_to_sid_enabled"])

    def test_config_rejects_a_stale_three_arm_matrix(self) -> None:
        config = build_config()
        config["experiments"]["arms"] = ["A", "B", "C"]

        with self.assertRaisesRegex(ValueError, "experiments.arms"):
            validate_config(config)

if __name__ == "__main__":
    unittest.main()
