from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.rollout import rollout_seed
from ksllm4rec_rloo.temperature import (
    _validate_candidate_audit,
    aggregate_audit,
    build_comparison,
    group_derived_fields,
    load_temperature_config,
    parameter_fingerprint,
    verify_determinism,
)
from ksllm4rec_rloo.integrity import canonical_sha256


PROJECT_ROOT = Path("/home/lyc/REC_PROJECTS/KSLLM4REC")
CONFIG = PROJECT_ROOT / "configs/rloo/frontier_epoch2_temperature_t100_t120.yaml"


def _candidate(index: int, tier: str, reward: float, sid: str) -> dict:
    return {
        "candidate_index": index,
        "seed": index,
        "sid": sid,
        "reward": reward,
        "reward_tier": tier,
        "token_ids": [index],
        "sampled_logps": [-0.25],
        "replay_logps": [-0.25],
        "decision_mask": [True],
        "legal_action_counts": [4],
        "legal_entropies": [0.5],
        "max_abs_sample_replay_logp_difference": 0.0,
        "decisions": [
            {
                "stage": "a",
                "entropy_nats": 0.5,
                "normalized_entropy": 0.25,
                "legal_action_count": 4,
            }
        ],
    }


def _row(index: int, tiers: list[str], rewards: list[float], *, unique: int) -> dict:
    candidates = [
        _candidate(
            candidate_index,
            tier,
            reward,
            f"sid-{candidate_index % unique}",
        )
        for candidate_index, (tier, reward) in enumerate(zip(tiers, rewards, strict=True))
    ]
    informative = len(set(rewards)) > 1
    return {
        "group_index": index,
        "group_id": f"group-{index}",
        "candidate_count": 16,
        "candidates": candidates,
        "reward_mean": sum(rewards) / 16,
        "reward_max": max(rewards),
        "informative_rloo": informative,
        "equal_reward": not informative,
        "uniform_tier": tiers[0] if len(set(tiers)) == 1 else None,
        "any_exact": "exact" in tiers,
        "unique_sid_count": unique,
        "duplicate_slots": 16 - unique,
        "all_same_sid": unique == 1,
        "decision_count": 16,
        "entropy_sum_nats": 8.0,
        "normalized_entropy_sum": 4.0,
        "max_abs_sample_replay_logp_difference": 0.0,
        "all_legal": True,
        "all_finite": True,
        "gt_injection_count": 0,
        "backward_passes": 0,
        "optimizer_updates": 0,
    }


class TemperatureConfigTest(unittest.TestCase):
    def test_checked_in_config_matches_the_frozen_contract(self) -> None:
        config = load_temperature_config(CONFIG)
        self.assertEqual(config["sampling"]["temperatures"], [1.0, 1.2])
        self.assertEqual(config["sampling"]["epoch_index"], 3)
        self.assertTrue(config["safety"]["inference_only"])

    def test_smoke_uses_the_frozen_first_eight_calibration_groups(self) -> None:
        calibration = json.loads(
            (
                PROJECT_ROOT
                / "configs/rloo/frontier_sft_epoch2_lora64_calibration_ids.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            canonical_sha256(calibration["group_ids"][:8]),
            "d1fbac82f2bde7b0da99d34444648e95849c99d3f50b8bea3f071eb0b53f47fe",
        )


class TemperatureMetricTest(unittest.TestCase):
    def test_aggregate_definitions_are_slot_and_group_weighted_as_documented(self) -> None:
        informative = _row(
            0,
            ["exact"] + ["same_domain"] * 15,
            [1.0] + [0.01] * 15,
            unique=16,
        )
        uniform = _row(1, ["same_domain"] * 16, [0.01] * 16, unique=8)
        metrics = aggregate_audit([informative, uniform])
        self.assertEqual(metrics["groups"], 2)
        self.assertEqual(metrics["slots"], 32)
        self.assertEqual(metrics["informative_rloo_groups"], 1)
        self.assertEqual(metrics["equal_reward_groups"], 1)
        self.assertEqual(metrics["uniform_tier_group_counts"]["same_domain"], 1)
        self.assertEqual(metrics["exact_slots"], 1)
        self.assertEqual(metrics["any_exact_groups"], 1)
        self.assertEqual(metrics["mean_unique_sids_per_group"], 12.0)
        self.assertEqual(metrics["duplicate_slot_rate"], 8 / 32)
        self.assertEqual(metrics["average_legal_action_entropy_nats"], 0.5)
        self.assertEqual(metrics["average_normalized_legal_action_entropy"], 0.25)

    def test_group_fields_are_recomputed_from_candidates(self) -> None:
        row = _row(0, ["same_ab"] * 16, [0.4] * 16, unique=4)
        derived = group_derived_fields(row["candidates"])
        self.assertEqual(derived["reward_mean"], 0.4)
        self.assertEqual(derived["uniform_tier"], "same_ab")
        self.assertEqual(derived["duplicate_slots"], 12)
        row["informative_rloo"] = True
        self.assertNotEqual(row["informative_rloo"], derived["informative_rloo"])

    def test_512_groups_have_exactly_8192_slots(self) -> None:
        rows = [
            _row(index, ["same_domain"] * 16, [0.01] * 16, unique=12)
            for index in range(512)
        ]
        metrics = aggregate_audit(rows)
        self.assertEqual(metrics["groups"], 512)
        self.assertEqual(metrics["slots"], 8192)
        self.assertEqual(
            sum(metrics["uniform_tier_group_counts"].values())
            + metrics["informative_rloo_groups"],
            512,
        )

    def test_identical_arms_do_not_pass_the_predeclared_direction_gates(self) -> None:
        rows = [
            _row(index, ["same_domain"] * 16, [0.01] * 16, unique=12)
            for index in range(20)
        ]
        config = load_temperature_config(CONFIG)
        comparison = build_comparison(config, rows, rows)
        self.assertEqual(comparison["decision"], "not_promising")
        self.assertFalse(comparison["decision_gates"]["informative_rloo_group_rate"])
        self.assertFalse(comparison["automatic_training_started"])


class TemperatureSafetyTest(unittest.TestCase):
    def test_parameter_fingerprint_detects_a_tensor_change(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        before = parameter_fingerprint(model)
        with torch.no_grad():
            model.weight[0, 0].add_(1.0)
        after = parameter_fingerprint(model)
        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertEqual(before["tensors"], after["tensors"])

    def test_determinism_requires_identical_semantic_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            row = {"group_id": "g", "candidates": [{"candidate_index": 0}]}
            payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for arm in ("t100", "t120"):
                (left / f"{arm}_audit.jsonl").write_text(payload, encoding="utf-8")
                (right / f"{arm}_audit.jsonl").write_text(payload, encoding="utf-8")
            report = verify_determinism(
                left_root=left,
                right_root=right,
                expected_groups=1,
            )
            self.assertTrue(report["passed"])
            self.assertTrue(report["arms"]["t100"]["exact_match"])

    def test_determinism_rejects_one_changed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            for arm in ("t100", "t120"):
                (left / f"{arm}_audit.jsonl").write_text(
                    '{"candidate":0}\n', encoding="utf-8"
                )
                (right / f"{arm}_audit.jsonl").write_text(
                    '{"candidate":1}\n', encoding="utf-8"
                )
            with self.assertRaisesRegex(RuntimeError, "payloads differ"):
                verify_determinism(
                    left_root=left,
                    right_root=right,
                    expected_groups=1,
                )

    def test_candidate_verifier_uses_canonical_reward_and_entropy(self) -> None:
        sid = Sid("video", 1, 2, 3)

        class Grammar:
            domain_token_ids = {"video": 10}
            a_offset = 100
            b_offset = 200
            c_offset = 300

            @staticmethod
            def allowed_next(prefix):
                return [10, 11] if not prefix else [99]

            @staticmethod
            def parse(tokens):
                if list(tokens) != [10, 99]:
                    raise ValueError("illegal")
                return sid

        entropy = 0.5
        candidate = {
            "candidate_index": 0,
            "seed": rollout_seed("group", 3, 0),
            "sid": sid.render(),
            "token_ids": [10, 99],
            "reward": 1.0,
            "reward_tier": "exact",
            "sampled_logps": [-0.2, 0.0],
            "replay_logps": [-0.2, 0.0],
            "decision_mask": [True, False],
            "legal_action_counts": [2, 1],
            "legal_entropies": [entropy, 0.0],
            "decisions": [
                {
                    "token_position": 0,
                    "stage": "domain",
                    "legal_action_count": 2,
                    "selected_token_id": 10,
                    "sampled_logp": -0.2,
                    "replay_logp": -0.2,
                    "entropy_nats": entropy,
                    "normalized_entropy": entropy / math.log(2),
                }
            ],
            "max_abs_sample_replay_logp_difference": 0.0,
        }
        result = _validate_candidate_audit(
            candidate,
            grammar=Grammar(),
            positives=[sid],
            group_id="group",
            epoch_index=3,
            maximum_logp_difference=1.0e-5,
        )
        self.assertEqual(result.sid, sid)
        tampered = json.loads(json.dumps(candidate))
        tampered["decisions"][0]["normalized_entropy"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "decision details"):
            _validate_candidate_audit(
                tampered,
                grammar=Grammar(),
                positives=[sid],
                group_id="group",
                epoch_index=3,
                maximum_logp_difference=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
