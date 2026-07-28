from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ksllm4rec_dapo_anchor_multitask_v1._infra.orpo_data import Sid
from ksllm4rec_dapo_anchor_multitask_v1.config import build_config
from ksllm4rec_dapo_anchor_multitask_v1.source_blocks import build_source_blocks
from ksllm4rec_dapo_anchor_multitask_v1.verify import (
    validate_experiment_matrix,
    validate_probe64_report,
    validate_run_summary,
    validate_run_directory_contents,
    validate_source_group_rows,
    validate_window_rows,
    verify_run_directory,
)


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block in build_source_blocks():
        for task, group_range, prefix in (
            ("recommendation", block.recommendation, "rec"),
            ("item_text_to_sid", block.text_to_sid, "text"),
        ):
            for source_index in range(group_range.start, group_range.stop):
                if task == "recommendation" and source_index < 33:
                    route = "rl"
                elif (task, source_index) in {
                    ("recommendation", 33),
                    ("item_text_to_sid", 0),
                }:
                    route = "anchor"
                else:
                    route = "skip"
                rows.append(
                    {
                        "block_index": block.index,
                        "task": task,
                        "source_index": source_index,
                        "group_id": f"{prefix}-{source_index}",
                        "rollout_count": 16,
                        "route": route,
                        "policy_step_at_sample": 0,
                    }
                )
    return rows


class SourceGroupVerificationTest(unittest.TestCase):
    def test_arm_c_consumes_every_floor_partitioned_source_group_once(self) -> None:
        result = validate_source_group_rows(_source_rows(), build_config(arm="C"))

        self.assertEqual(result.source_groups, 27_613)
        self.assertEqual(result.candidate_rollouts, 441_808)
        self.assertEqual(result.completed_source_blocks, 532)

    def test_rejects_a_duplicate_even_when_the_total_row_count_is_unchanged(self) -> None:
        rows = _source_rows()
        rows[-1] = dict(rows[-2])

        with self.assertRaisesRegex(ValueError, "source plan"):
            validate_source_group_rows(rows, build_config(arm="C"))

    def test_rejects_an_unknown_objective_route(self) -> None:
        rows = _source_rows()
        rows[0]["route"] = "rl_only"

        with self.assertRaisesRegex(ValueError, "Invalid source row"):
            validate_source_group_rows(rows, build_config(arm="C"))


def _step(
    group_ids: list[str], *, learning_rate: float, ratio_min: float = 0.9
) -> dict[str, object]:
    return {
        "loss": 0.25,
        "learning_rate": learning_rate,
        "decision_tokens": len(group_ids) * 16,
        "max_replay_logp_difference": 1.0e-6,
        "ratio_min": ratio_min,
        "ratio_max": 1.1,
        "ratio_mean": 1.0,
        "clipped_tokens": 0,
        "below_low_tokens": 0,
        "above_high_tokens": 0,
        "group_ids": group_ids,
        "tasks": ["recommendation"] * len(group_ids),
    }


def _window(start: int = 0) -> dict[str, object]:
    group_ids = [f"rec-{index}" for index in range(start, start + 33)]
    chunks = [group_ids[index : index + 8] for index in range(0, 33, 8)]
    return {
        "optimization_window_index": start // 33,
        "first_source_block": 0,
        "next_source_block": 2,
        "snapshot_policy_step": 0,
        "completed_policy_steps": 5,
        "rl_groups": 33,
        "anchor_groups": 2,
        "policy_optimizer_steps": 5,
        "anchor_optimizer_steps": 0,
        "policy_gradient_norms": [1.0, 2.0, 3.0, 4.0, 5.0],
        "rl_reference_gradient_norm": 3.0,
        "lambda_cap": 0.15,
        "lambda_effective": 0.05,
        "anchor_raw_gradient_norm": 2.0,
        "anchor_scaled_gradient_norm": 0.1,
        "anchor_to_rl_gradient_ratio": 1.0 / 30.0,
        "combined_gradient_norm": 3.05,
        "steps": [
            _step(chunk, learning_rate=1.0e-7 if index < 4 else 2.0e-7)
            for index, chunk in enumerate(chunks)
        ],
        "anchor_group_results": [
            {
                "task": "recommendation",
                "group_id": "rec-33",
                "set_nll": 2.0,
                "all_gt_decision_nll_sum": 4.0,
                "gt_sid_count": 1,
                "decision_tokens": 4,
            },
            {
                "task": "item_text_to_sid",
                "group_id": "text-0",
                "set_nll": 3.0,
                "all_gt_decision_nll_sum": 5.0,
                "gt_sid_count": 1,
                "decision_tokens": 4,
            },
        ],
        "anchor_max_replay_logp_difference": 1.0e-6,
        "k": 1,
    }


class WindowVerificationTest(unittest.TestCase):
    def test_accepts_k1_variable_minibatches_and_merged_anchor(self) -> None:
        result = validate_window_rows([_window()], build_config())

        self.assertEqual(result.optimization_windows, 1)
        self.assertEqual(result.policy_optimizer_steps, 5)
        self.assertEqual(result.rl_groups, 33)

    def test_accepts_a_window_with_no_anchor_candidates(self) -> None:
        row = _window()
        row.update(
            {
                "anchor_groups": 0,
                "lambda_cap": 0.0,
                "lambda_effective": 0.0,
                "anchor_raw_gradient_norm": 0.0,
                "anchor_scaled_gradient_norm": 0.0,
                "anchor_to_rl_gradient_ratio": 0.0,
                "anchor_group_results": [],
                "anchor_max_replay_logp_difference": 0.0,
            }
        )

        validate_window_rows([row], build_config())

    def test_ratio_outside_bounds_need_not_activate_loss_clip_for_both_advantage_signs(self) -> None:
        row = _window()
        first = row["steps"][0]
        first.update(
            {
                "ratio_min": 0.7,
                "ratio_mean": 0.99,
                "below_low_tokens": 1,
                "clipped_tokens": 0,
            }
        )

        validate_window_rows([row], build_config())

    def test_later_minibatch_may_drift_after_previous_policy_updates(self) -> None:
        row = _window()
        row["steps"][1]["max_replay_logp_difference"] = 0.67

        validate_window_rows([row], build_config())

    def test_first_minibatch_must_match_the_frozen_rollout_policy(self) -> None:
        row = _window()
        row["steps"][0]["max_replay_logp_difference"] = 0.01

        with self.assertRaisesRegex(ValueError, "first policy minibatch"):
            validate_window_rows([row], build_config())

    def test_rejects_an_independent_anchor_optimizer_step(self) -> None:
        row = _window()
        row["anchor_optimizer_steps"] = 1

        with self.assertRaisesRegex(ValueError, "independent optimizer"):
            validate_window_rows([row], build_config())

    def test_rejects_a_group_reused_across_policy_minibatches(self) -> None:
        row = _window()
        row["steps"][1]["group_ids"][0] = row["steps"][0]["group_ids"][0]

        with self.assertRaisesRegex(ValueError, "K=1"):
            validate_window_rows([row], build_config())


def _group_rows(window: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in window["steps"]:
        for task, group_id in zip(step["tasks"], step["group_ids"], strict=True):
            rows.append(
                {
                    "optimization_window_index": 0,
                    "source_block": 0,
                    "source_index": int(str(group_id).split("-")[-1]),
                    "task": task,
                    "group_id": group_id,
                    "route": "rl",
                    "rewards": [0.01] * 16,
                    "advantages": [0.0] * 16,
                    "tiers": ["same_domain"] * 16,
                    "candidate_sids": [f"sid-{index}" for index in range(16)],
                    "gt_injection_count": 0,
                    "k": 1,
                }
            )
    return rows


class RunSummaryVerificationTest(unittest.TestCase):
    def test_recomputes_summary_counters_from_the_logs(self) -> None:
        config = build_config()
        source_result = validate_source_group_rows(_source_rows(), config)
        window = _window()
        window_result = validate_window_rows([window], config)
        summary = {
            "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
            "arm": "C",
            "completed_source_blocks": 532,
            "source_groups": 27_613,
            "candidate_rollouts": 441_808,
            "optimization_windows": 1,
            "completed_policy_steps": 5,
            "final_auxiliary_flush_steps": 0,
            "final_auxiliary_flush": None,
            "total_optimizer_steps": 5,
            "rl_groups": 33,
            "complete": True,
            "last_checkpoint": "/tmp/checkpoint",
        }

        validate_run_summary(
            summary,
            source=source_result,
            windows=window_result,
            group_rows=_group_rows(window),
            config=config,
        )

    def test_accepts_float32_serialized_reward_values(self) -> None:
        config = build_config()
        source_result = validate_source_group_rows(_source_rows(), config)
        window = _window()
        rows = _group_rows(window)
        for row in rows:
            row["rewards"] = [0.15000000596046448] * 16
        summary = {
            "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
            "arm": "C",
            "completed_source_blocks": 532,
            "source_groups": 27_613,
            "candidate_rollouts": 441_808,
            "optimization_windows": 1,
            "completed_policy_steps": 5,
            "final_auxiliary_flush_steps": 0,
            "final_auxiliary_flush": None,
            "total_optimizer_steps": 5,
            "rl_groups": 33,
            "complete": True,
            "last_checkpoint": "/tmp/checkpoint",
        }

        validate_run_summary(
            summary,
            source=source_result,
            windows=validate_window_rows([window], config),
            group_rows=rows,
            config=config,
        )

    def test_group_log_may_keep_source_order_instead_of_minibatch_order(self) -> None:
        config = build_config()
        source_result = validate_source_group_rows(_source_rows(), config)
        window = _window()
        rows = list(reversed(_group_rows(window)))
        summary = {
            "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
            "arm": "C",
            "completed_source_blocks": 532,
            "source_groups": 27_613,
            "candidate_rollouts": 441_808,
            "optimization_windows": 1,
            "completed_policy_steps": 5,
            "final_auxiliary_flush_steps": 0,
            "final_auxiliary_flush": None,
            "total_optimizer_steps": 5,
            "rl_groups": 33,
            "complete": True,
            "last_checkpoint": "/tmp/checkpoint",
        }

        validate_run_summary(
            summary,
            source=source_result,
            windows=validate_window_rows([window], config),
            group_rows=rows,
            config=config,
        )

    def test_group_log_order_may_differ_from_the_stable_minibatch_shuffle(self) -> None:
        config = build_config()
        source_result = validate_source_group_rows(_source_rows(), config)
        window = _window()
        group_rows = list(reversed(_group_rows(window)))
        summary = {
            "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
            "arm": "C",
            "completed_source_blocks": 532,
            "source_groups": 27_613,
            "candidate_rollouts": 441_808,
            "optimization_windows": 1,
            "completed_policy_steps": 5,
            "final_auxiliary_flush_steps": 0,
            "final_auxiliary_flush": None,
            "total_optimizer_steps": 5,
            "rl_groups": 33,
            "complete": True,
            "last_checkpoint": "/tmp/checkpoint",
        }

        validate_run_summary(
            summary,
            source=source_result,
            windows=validate_window_rows([window], config),
            group_rows=group_rows,
            config=config,
        )

    def test_rejects_a_summary_with_a_false_candidate_total(self) -> None:
        config = build_config()
        source_result = validate_source_group_rows(_source_rows(), config)
        window = _window()
        summary = {
            "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
            "arm": "C",
            "completed_source_blocks": 532,
            "source_groups": 27_613,
            "candidate_rollouts": 441_807,
            "optimization_windows": 1,
            "completed_policy_steps": 5,
            "final_auxiliary_flush_steps": 0,
            "final_auxiliary_flush": None,
            "total_optimizer_steps": 5,
            "rl_groups": 33,
            "complete": True,
            "last_checkpoint": "/tmp/checkpoint",
        }

        with self.assertRaisesRegex(ValueError, "candidate_rollouts"):
            validate_run_summary(
                summary,
                source=source_result,
                windows=validate_window_rows([window], config),
                group_rows=_group_rows(window),
                config=config,
            )

    def test_rejects_source_rl_routing_that_differs_from_window_membership(self) -> None:
        config = build_config()
        rows = _source_rows()
        rows[0]["route"] = "skip"
        source_result = validate_source_group_rows(rows, config)
        window = _window()

        with self.assertRaisesRegex(ValueError, "Source routes and RL"):
            validate_run_summary(
                {
                    "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
                    "arm": "C",
                    "completed_source_blocks": 532,
                    "source_groups": 27_613,
                    "candidate_rollouts": 441_808,
                    "optimization_windows": 1,
                    "completed_policy_steps": 5,
                    "final_auxiliary_flush_steps": 0,
                    "final_auxiliary_flush": None,
                    "total_optimizer_steps": 5,
                    "rl_groups": 33,
                    "complete": True,
                    "last_checkpoint": "/tmp/checkpoint",
                },
                source=source_result,
                windows=validate_window_rows([window], config),
                group_rows=_group_rows(window),
                config=config,
            )

    def test_accepts_a_zero_step_source_end_anchor_audit_without_rl_reference(self) -> None:
        config = build_config()
        rows = _source_rows()
        for row in rows:
            row["route"] = "skip"
        rows[-1]["route"] = "anchor"
        source_result = validate_source_group_rows(rows, config)
        windows = validate_window_rows([], config)
        auxiliary = {
            "first_source_block": 0,
            "next_source_block": 532,
            "snapshot_policy_step": 0,
            "anchor_groups": 1,
            "gt_sid_rows": 1,
            "decision_tokens": 4,
            "optimizer_steps": 0,
            "scheduler_advanced": False,
            "rl_reference_gradient_norm": 0.0,
            "lambda_cap": 0.0,
            "lambda_effective": 0.0,
            "anchor_raw_gradient_norm": 2.0,
            "anchor_scaled_gradient_norm": 0.0,
            "anchor_to_rl_gradient_ratio": 0.0,
            "combined_gradient_norm": 0.0,
            "anchor_set_nll_mean": 3.0,
            "anchor_max_replay_logp_difference": 1.0e-6,
        }

        validate_run_summary(
            {
                "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
                "arm": "C",
                "completed_source_blocks": 532,
                "source_groups": 27_613,
                "candidate_rollouts": 441_808,
                "optimization_windows": 0,
                "completed_policy_steps": 0,
                "final_auxiliary_flush_steps": 0,
                "final_auxiliary_flush": auxiliary,
                "total_optimizer_steps": 0,
                "rl_groups": 0,
                "complete": True,
                "last_checkpoint": "/tmp/checkpoint",
            },
            source=source_result,
            windows=windows,
            group_rows=[],
            config=config,
        )


class ExperimentMatrixVerificationTest(unittest.TestCase):
    def test_a_b_c_differ_only_by_the_confirmed_variables(self) -> None:
        differences = validate_experiment_matrix(
            {arm: build_config(arm=arm) for arm in ("A", "B", "C")}
        )

        self.assertEqual(
            differences["A:B"],
            {
                "experiments.active.arm",
                "experiments.active.reward_profile",
                "reward.same_a",
                "reward.same_ab",
            },
        )
        self.assertEqual(
            differences["B:C"],
            {
                "experiments.active.arm",
                "experiments.active.text_to_sid_enabled",
            },
        )

    def test_rejects_an_unconfirmed_training_difference(self) -> None:
        configs = {arm: build_config(arm=arm) for arm in ("A", "B", "C")}
        configs["C"]["train"]["learning_rate"] = 2.0e-6

        with self.assertRaisesRegex(ValueError, "train.learning_rate"):
            validate_experiment_matrix(configs)

    def test_rejects_shared_experiment_state_paths(self) -> None:
        configs = {arm: build_config(arm=arm) for arm in ("A", "B", "C")}
        configs["B"]["output"]["run_dir"] = configs["A"]["output"]["run_dir"]

        with self.assertRaisesRegex(ValueError, "different run/log/calibration"):
            validate_experiment_matrix(configs)


class _ProbeTrie:
    def __init__(self, accepted: set[Sid]) -> None:
        self.accepted = accepted

    def contains(self, sid: Sid) -> bool:
        return sid in self.accepted


def _probe_expected(source_line: int = 17) -> dict[str, object]:
    candidates = [Sid("video", 0, 0, index).render() for index in range(64)]
    return {
        "task": "recommendation",
        "source_row_sha256": "a" * 64,
        "source_line": source_line,
        "target_sid": candidates[36],
    }


def _probe_result() -> dict[str, object]:
    candidates = [Sid("video", 0, 0, index).render() for index in range(64)]
    return {
        "probe_index": 0,
        "task": "recommendation",
        "source_row_sha256": "a" * 64,
        "source_line": 17,
        "target_sid": candidates[36],
        "prompt_tokens": 42,
        "candidate_sids": candidates,
        "hit_rank": 37,
        "pass_at_64": True,
    }


def _probe_report(result: dict[str, object] | None = None) -> dict[str, object]:
    value = _probe_result() if result is None else result
    return {
        "schema_version": 1,
        "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
        "complete": True,
        "num_candidates": 64,
        "require_unique_candidates": True,
        "cache_implementation": "offloaded",
        "selected_indices": [0],
        "result_count": 1,
        "pass_count": int(bool(value["pass_at_64"])),
        "results": [value],
    }


class Probe64ReportVerificationTest(unittest.TestCase):
    def test_recomputes_pass64_from_all_64_unique_legal_candidates(self) -> None:
        result = _probe_result()
        trie = _ProbeTrie({Sid.parse(value) for value in result["candidate_sids"]})
        count = validate_probe64_report(
            _probe_report(result),
            expected_rows=[_probe_expected()],
            trie=trie,
            require_complete=True,
        )

        self.assertEqual(count, 1)

    def test_rejects_a_report_that_duplicates_one_of_64_candidates(self) -> None:
        result = _probe_result()
        trie = _ProbeTrie({Sid.parse(value) for value in result["candidate_sids"]})
        result["candidate_sids"][-1] = result["candidate_sids"][0]

        with self.assertRaisesRegex(ValueError, "64 unique"):
            validate_probe64_report(
                _probe_report(result),
                expected_rows=[_probe_expected()],
                trie=trie,
                require_complete=True,
            )

    def test_rejects_a_well_formed_sid_absent_from_the_fixed_trie(self) -> None:
        result = _probe_result()
        accepted = {Sid.parse(value) for value in result["candidate_sids"][:-1]}

        with self.assertRaisesRegex(ValueError, "trie SIDs"):
            validate_probe64_report(
                _probe_report(result),
                expected_rows=[_probe_expected()],
                trie=_ProbeTrie(accepted),
                require_complete=True,
            )

    def test_rejects_a_result_bound_to_another_fixed_source_row(self) -> None:
        result = _probe_result()
        result["source_line"] = 18
        trie = _ProbeTrie({Sid.parse(value) for value in result["candidate_sids"]})

        with self.assertRaisesRegex(ValueError, "fixed source row"):
            validate_probe64_report(
                _probe_report(result),
                expected_rows=[_probe_expected()],
                trie=trie,
                require_complete=True,
            )

    def test_complete_verification_rejects_a_partial_probe(self) -> None:
        result = _probe_result()
        report = _probe_report(result)
        report["complete"] = False
        trie = _ProbeTrie({Sid.parse(value) for value in result["candidate_sids"]})

        with self.assertRaisesRegex(ValueError, "cover every fixed row"):
            validate_probe64_report(
                report,
                expected_rows=[_probe_expected(), _probe_expected(source_line=18)],
                trie=trie,
                require_complete=True,
            )


class RunDirectoryVerificationTest(unittest.TestCase):
    def test_rejects_every_uncontracted_top_level_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "health.jsonl").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "uncontracted artifacts"):
                validate_run_directory_contents(root)

    def test_reads_and_cross_checks_the_complete_run_artifacts(self) -> None:
        config = build_config()
        window = _window()
        summary = {
            "spec_version": "DAPO-ANCHOR-MULTITASK-V1.0",
            "arm": "C",
            "completed_source_blocks": 532,
            "source_groups": 27_613,
            "candidate_rollouts": 441_808,
            "optimization_windows": 1,
            "completed_policy_steps": 5,
            "final_auxiliary_flush_steps": 0,
            "final_auxiliary_flush": None,
            "total_optimizer_steps": 5,
            "rl_groups": 33,
            "complete": True,
            "last_checkpoint": "/tmp/checkpoint",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (
                ("source_groups.jsonl", _source_rows()),
                ("groups.jsonl", _group_rows(window)),
                ("windows.jsonl", [window]),
            ):
                (root / name).write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                    ),
                    encoding="utf-8",
                )
            (root / "run_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            result = verify_run_directory(root, config)

        self.assertEqual(result.source.candidate_rollouts, 441_808)
        self.assertEqual(result.windows.policy_optimizer_steps, 5)


if __name__ == "__main__":
    unittest.main()
