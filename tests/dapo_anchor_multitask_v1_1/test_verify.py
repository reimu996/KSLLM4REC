from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from ksllm4rec_dapo_anchor_multitask_v1_1 import contract
from ksllm4rec_dapo_anchor_multitask_v1_1.config import build_config
from ksllm4rec_dapo_anchor_multitask_v1_1.source_blocks import (
    SOURCE_BLOCKS,
    build_epoch_source_plans,
    source_plan_manifest,
)
from ksllm4rec_dapo_anchor_multitask_v1_1.verify import (
    validate_anchor_group_rows,
    validate_run_directory_contents,
    validate_run_summary,
    validate_experiment_matrix,
    validate_source_group_rows,
    validate_window_rows,
    verify_run_directory,
)


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


@lru_cache(maxsize=1)
def _fixture() -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, object],
    tuple[dict[str, object], ...],
]:
    """One exact two-epoch source plan with 32 RL groups in each first block."""

    recommendation_ids = tuple(
        f"recommendation-{index:05d}"
        for index in range(contract.EXPECTED_RECOMMENDATION_GROUPS)
    )
    text_ids = tuple(
        f"text-to-sid-{index:05d}" for index in range(contract.EXPECTED_TEXT_GROUPS)
    )
    plans = build_epoch_source_plans(recommendation_ids, text_ids, seed=42)
    manifest = source_plan_manifest(
        plans, recommendation_ids, text_ids, seed=42
    )
    source_rows: list[dict[str, object]] = []
    for plan in plans:
        for block in plan.blocks:
            task_rows: list[tuple[str, int, str]] = []
            for position in range(block.recommendation.start, block.recommendation.stop):
                source_index = plan.recommendation_order[position]
                task_rows.append(
                    ("recommendation", source_index, recommendation_ids[source_index])
                )
            for position in range(block.text_to_sid.start, block.text_to_sid.stop):
                source_index = plan.text_to_sid_order[position]
                task_rows.append(("item_text_to_sid", source_index, text_ids[source_index]))
            for position, (task, source_index, group_id) in enumerate(task_rows):
                source_rows.append(
                    {
                        "block_index": block.index,
                        "epoch_index": block.epoch_index,
                        "epoch_block_index": block.epoch_block_index,
                        "task": task,
                        "source_index": source_index,
                        "group_id": group_id,
                        "rollout_count": contract.GROUP_SIZE,
                        "route": (
                            "rl_and_anchor"
                            if block.epoch_block_index == 0 and position < 32
                            else "anchor"
                        ),
                        "policy_step_at_sample": (
                            0
                            if block.index == 0
                            else 4
                            if block.index < SOURCE_BLOCKS
                            else 4
                            if block.index == SOURCE_BLOCKS
                            else 8
                        ),
                    }
                )
    assert len(source_rows) == contract.TOTAL_SOURCE_GROUPS
    return recommendation_ids, text_ids, manifest, tuple(source_rows)


def _source_rows() -> list[dict[str, object]]:
    return [dict(row) for row in _fixture()[3]]


def _first_block_rows(epoch_index: int) -> list[dict[str, object]]:
    block = epoch_index * SOURCE_BLOCKS
    return [row for row in _source_rows() if row["block_index"] == block]


def _gradient_fields() -> dict[str, object]:
    return {
        "lambda_cap": 0.05,
        "lambda_effective": 0.05,
        "anchor_raw_gradient_norm": 4.0,
        "anchor_scaled_gradient_norm": 0.2,
        "anchor_to_rl_gradient_ratio": 0.1,
        "combined_gradient_norm": 2.1,
    }


def _anchor_result(row: dict[str, object]) -> dict[str, object]:
    return {
        "epoch_index": row["epoch_index"],
        "source_block": row["block_index"],
        "epoch_block_index": row["epoch_block_index"],
        "task": row["task"],
        "group_id": row["group_id"],
        "set_nll": 2.0,
        "all_gt_decision_nll_sum": 2.0,
        "gt_sid_count": 1,
        "decision_tokens": 4,
    }


def _step(rows: list[dict[str, object]], learning_rate: float) -> dict[str, object]:
    return {
        "loss": 0.25,
        "learning_rate": learning_rate,
        "decision_tokens": 16 * len(rows),
        "max_replay_logp_difference": 1.0e-6,
        "ratio_min": 0.9,
        "ratio_max": 1.1,
        "ratio_mean": 1.0,
        "clipped_tokens": 0,
        "below_low_tokens": 0,
        "above_high_tokens": 0,
        "group_ids": [str(row["group_id"]) for row in rows],
        "tasks": [str(row["task"]) for row in rows],
        "epoch_indexes": [int(row["epoch_index"]) for row in rows],
    }


def _window(epoch_index: int) -> dict[str, object]:
    source_rows = _first_block_rows(epoch_index)
    rl_rows = [row for row in source_rows if row["route"] == "rl_and_anchor"]
    assert len(rl_rows) == 32
    steps = [
        _step(
            rl_rows[start : start + 8],
            1.0e-7 if epoch_index == 0 else 2.0e-7,
        )
        for start in range(0, len(rl_rows), 8)
    ]
    first_block = epoch_index * SOURCE_BLOCKS
    snapshot = 4 * epoch_index
    return {
        "optimization_window_index": epoch_index,
        "first_source_block": first_block,
        "next_source_block": first_block + 1,
        "snapshot_policy_step": snapshot,
        "completed_policy_steps": snapshot + 4,
        "rl_groups": len(rl_rows),
        "anchor_groups": len(source_rows),
        "policy_optimizer_steps": 4,
        "anchor_optimizer_steps": 0,
        "policy_gradient_norms": [2.0] * 4,
        "rl_reference_gradient_norm": 2.0,
        **_gradient_fields(),
        "steps": steps,
        "anchor_group_results": [_anchor_result(row) for row in source_rows],
        "anchor_max_replay_logp_difference": 1.0e-6,
        "k": 1,
    }


def _windows() -> list[dict[str, object]]:
    return [_window(0), _window(1)]


def _group_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window_index, source_rows in enumerate((_first_block_rows(0), _first_block_rows(1))):
        for source in (row for row in source_rows if row["route"] == "rl_and_anchor"):
            rows.append(
                {
                    "optimization_window_index": window_index,
                    "source_block": source["block_index"],
                    "epoch_index": source["epoch_index"],
                    "epoch_block_index": source["epoch_block_index"],
                    "source_index": source["source_index"],
                    "task": source["task"],
                    "group_id": source["group_id"],
                    "route": "rl_and_anchor",
                    "rewards": [1.0] + [0.0] * 15,
                    "advantages": [15.0 / 16.0] + [-1.0 / 16.0] * 15,
                    "tiers": ["exact"] + ["other_domain"] * 15,
                    "candidate_sids": [f"sid-{index}" for index in range(16)],
                    "gt_injection_count": 0,
                    "k": 1,
                }
            )
    return rows


def _anchor_log_rows() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for row in _source_rows():
        epoch_index = int(row["epoch_index"])
        merged = int(row["epoch_block_index"]) == 0
        values.append(
            {
                **_anchor_result(row),
                "phase": "merged_policy_window" if merged else "epoch_end_anchor_flush",
                "unit_index": epoch_index,
            }
        )
    return values


def _flush_rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for epoch_index in range(contract.SOURCE_EPOCHS):
        source_rows = [
            row
            for row in _source_rows()
            if int(row["epoch_index"]) == epoch_index and int(row["epoch_block_index"]) > 0
        ]
        result.append(
            {
                "epoch_index": epoch_index,
                "first_source_block": epoch_index * SOURCE_BLOCKS + 1,
                "next_source_block": (epoch_index + 1) * SOURCE_BLOCKS,
                "snapshot_policy_step": 4 * (epoch_index + 1),
                "anchor_groups": len(source_rows),
                "gt_sid_rows": len(source_rows),
                "decision_tokens": 4 * len(source_rows),
                "optimizer_steps": 1,
                "scheduler_advanced": False,
                "rl_reference_gradient_norm": 2.0,
                **_gradient_fields(),
                "anchor_set_nll_mean": 2.0,
                "anchor_max_replay_logp_difference": 1.0e-6,
            }
        )
    return result


def _summary() -> dict[str, object]:
    _, _, manifest, _ = _fixture()
    return {
        "spec_version": contract.SPEC_VERSION,
        "arm": "D",
        "completed_source_blocks": contract.TOTAL_SOURCE_BLOCKS,
        "completed_epochs": contract.SOURCE_EPOCHS,
        "source_plan_sha256": manifest["sha256"],
        "epoch_plan_sha256s": [record["sha256"] for record in manifest["epochs"]],
        "source_groups": contract.TOTAL_SOURCE_GROUPS,
        "candidate_rollouts": contract.TOTAL_CANDIDATES,
        "optimization_windows": 2,
        "completed_policy_steps": 8,
        "anchor_groups": contract.TOTAL_SOURCE_GROUPS,
        "auxiliary_flush_steps": 2,
        "epoch_auxiliary_flushes": _flush_rows(),
        "total_optimizer_steps": 10,
        "rl_groups": 64,
        "complete": True,
        "last_checkpoint": "/tmp/checkpoint-block-001064-window-000002-update-000010",
    }


def _source_result():
    recommendation_ids, text_ids, manifest, _ = _fixture()
    return validate_source_group_rows(
        _source_rows(),
        build_config(),
        plan_manifest=manifest,
        recommendation_group_ids=recommendation_ids,
        text_to_sid_group_ids=text_ids,
    )


class SourcePlanVerificationTest(unittest.TestCase):
    def test_two_epochs_have_exact_normalized_coverage_and_anchor_membership(self) -> None:
        result = _source_result()

        self.assertEqual(result.source_groups, 55_226)
        self.assertEqual(result.candidate_rollouts, 883_616)
        self.assertEqual(result.completed_source_blocks, 1_064)
        self.assertEqual(len(result.anchor_memberships), 55_226)
        self.assertEqual(len(set(result.source_memberships)), 55_226)
        self.assertEqual(
            len({(task, group_id) for _, task, group_id in result.source_memberships}),
            27_613,
        )

    def test_rejects_a_plan_that_breaks_seeded_order_or_source_binding(self) -> None:
        recommendation_ids, text_ids, manifest, _ = _fixture()
        corrupted = deepcopy(manifest)
        corrupted["epochs"][0]["recommendation_group_ids"][0] = "forged-group-id"

        with self.assertRaisesRegex(ValueError, "SHA-256 order|hash"):
            validate_source_group_rows(
                _source_rows(),
                build_config(),
                plan_manifest=corrupted,
                recommendation_group_ids=recommendation_ids,
                text_to_sid_group_ids=text_ids,
            )

    def test_rejects_rehashed_plan_that_lies_about_original_source_index(self) -> None:
        recommendation_ids, text_ids, manifest, _ = _fixture()
        corrupted = deepcopy(manifest)
        indices = corrupted["epochs"][0]["recommendation_source_indices"]
        indices[0], indices[1] = indices[1], indices[0]
        payload = {
            "schema_version": corrupted["schema_version"],
            "seed": corrupted["seed"],
            "source_epochs": corrupted["source_epochs"],
            "source_blocks_per_epoch": corrupted["source_blocks_per_epoch"],
            "epochs": corrupted["epochs"],
        }
        corrupted["sha256"] = sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(ValueError, "does not bind"):
            validate_source_group_rows(
                _source_rows(),
                build_config(),
                plan_manifest=corrupted,
                recommendation_group_ids=recommendation_ids,
                text_to_sid_group_ids=text_ids,
            )

    def test_rejects_skip_because_every_normalized_group_must_enter_anchor(self) -> None:
        recommendation_ids, text_ids, manifest, _ = _fixture()
        rows = _source_rows()
        rows[100]["route"] = "skip"

        with self.assertRaisesRegex(ValueError, "must enter Anchor"):
            validate_source_group_rows(
                rows,
                build_config(),
                plan_manifest=manifest,
                recommendation_group_ids=recommendation_ids,
                text_to_sid_group_ids=text_ids,
            )


class WindowAndAnchorVerificationTest(unittest.TestCase):
    def test_epoch_key_permits_second_epoch_but_rejects_same_epoch_reuse(self) -> None:
        windows = _windows()
        result = validate_window_rows(windows, build_config())
        self.assertEqual(result.policy_optimizer_steps, 8)
        self.assertEqual(result.rl_groups, 64)

        windows[1]["steps"][0]["group_ids"][0] = windows[0]["steps"][0]["group_ids"][0]
        windows[1]["steps"][0]["tasks"][0] = windows[0]["steps"][0]["tasks"][0]
        windows[1]["steps"][0]["epoch_indexes"][0] = 0
        with self.assertRaisesRegex(ValueError, "epoch-keyed policy group"):
            validate_window_rows(windows, build_config())

    def test_anchor_log_covers_every_source_group_once(self) -> None:
        source = _source_result()
        windows = validate_window_rows(_windows(), build_config())
        anchors = validate_anchor_group_rows(
            _anchor_log_rows(), source=source, windows=windows
        )
        self.assertEqual(anchors.anchor_groups, contract.TOTAL_SOURCE_GROUPS)

    def test_rejects_missing_anchor_log_group(self) -> None:
        source = _source_result()
        windows = validate_window_rows(_windows(), build_config())
        rows = _anchor_log_rows()
        rows.pop()

        with self.assertRaisesRegex(ValueError, "cover every normalized source group"):
            validate_anchor_group_rows(rows, source=source, windows=windows)

    def test_run_summary_recomputes_two_epoch_counts_and_auxiliary_steps(self) -> None:
        source = _source_result()
        windows = validate_window_rows(_windows(), build_config())
        validate_run_summary(
            _summary(),
            source=source,
            windows=windows,
            group_rows=_group_rows(),
            anchor_group_rows=_anchor_log_rows(),
            epoch_auxiliary_flush_rows=_flush_rows(),
            config=build_config(),
        )

    def test_rejects_two_anchor_only_flushes_for_one_epoch(self) -> None:
        source = _source_result()
        windows = validate_window_rows(_windows(), build_config())
        flushes = _flush_rows()
        flushes.append(deepcopy(flushes[0]))

        with self.assertRaisesRegex(ValueError, "At most one"):
            validate_run_summary(
                _summary(),
                source=source,
                windows=windows,
                group_rows=_group_rows(),
                anchor_group_rows=_anchor_log_rows(),
                epoch_auxiliary_flush_rows=flushes,
                config=build_config(),
            )


class RunDirectoryVerificationTest(unittest.TestCase):
    def test_rejects_every_uncontracted_top_level_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "health.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "uncontracted artifacts"):
                validate_run_directory_contents(root)

    def test_reads_complete_v11_artifacts_with_explicit_frozen_source_ids(self) -> None:
        recommendation_ids, text_ids, manifest, _ = _fixture()
        config = build_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                "resolved_config.json": json.dumps(config),
                "runtime_signature.json": json.dumps({}),
                "source_epoch_plan.json": json.dumps(manifest),
                "source_groups.jsonl": _jsonl(_source_rows()),
                "groups.jsonl": _jsonl(_group_rows()),
                "windows.jsonl": _jsonl(_windows()),
                "anchor_groups.jsonl": _jsonl(_anchor_log_rows()),
                "epoch_auxiliary_flushes.jsonl": _jsonl(_flush_rows()),
                "run_summary.json": json.dumps(_summary()),
            }
            for name, value in artifacts.items():
                (root / name).write_text(value, encoding="utf-8")
            result = verify_run_directory(
                root,
                config,
                recommendation_group_ids=recommendation_ids,
                text_to_sid_group_ids=text_ids,
            )

        self.assertEqual(result.source.candidate_rollouts, 883_616)
        self.assertEqual(result.anchor_groups, 55_226)


class ExperimentMatrixVerificationTest(unittest.TestCase):
    def test_four_arm_matrix_allows_only_reward_task_and_output_differences(self) -> None:
        differences = validate_experiment_matrix(
            {arm: build_config(arm=arm) for arm in contract.EXPERIMENT_ARMS}
        )

        self.assertEqual(
            differences,
            {
                "A:B": {
                    "experiments.active.arm",
                    "experiments.active.reward_profile",
                    "reward.same_a",
                    "reward.same_ab",
                },
                "A:C": {
                    "experiments.active.arm",
                    "experiments.active.reward_profile",
                    "experiments.active.text_to_sid_enabled",
                    "reward.same_a",
                    "reward.same_ab",
                },
                "A:D": {
                    "experiments.active.arm",
                    "experiments.active.text_to_sid_enabled",
                },
                "B:C": {
                    "experiments.active.arm",
                    "experiments.active.text_to_sid_enabled",
                },
                "B:D": {
                    "experiments.active.arm",
                    "experiments.active.reward_profile",
                    "experiments.active.text_to_sid_enabled",
                    "reward.same_a",
                    "reward.same_ab",
                },
                "C:D": {
                    "experiments.active.arm",
                    "experiments.active.reward_profile",
                    "reward.same_a",
                    "reward.same_ab",
                },
            },
        )

    def test_four_arm_matrix_rejects_a_non_reward_difference(self) -> None:
        configs = {arm: build_config(arm=arm) for arm in contract.EXPERIMENT_ARMS}
        configs["D"]["rollout"]["temperature"] = 1.0

        with self.assertRaisesRegex(ValueError, "rollout.temperature"):
            validate_experiment_matrix(configs)

    def test_four_arm_matrix_rejects_d_reward_or_task_regression(self) -> None:
        configs = {arm: build_config(arm=arm) for arm in contract.EXPERIMENT_ARMS}
        configs["D"]["reward"]["same_ab"] = 0.15

        with self.assertRaisesRegex(ValueError, "reward.same_ab"):
            validate_experiment_matrix(configs)

        configs = {arm: build_config(arm=arm) for arm in contract.EXPERIMENT_ARMS}
        configs["D"]["experiments"]["active"]["text_to_sid_enabled"] = False
        with self.assertRaisesRegex(ValueError, "text_to_sid_enabled"):
            validate_experiment_matrix(configs)


if __name__ == "__main__":
    unittest.main()
