"""Deterministic structural verifiers for the V1.1 exact two-epoch run."""

from __future__ import annotations

import json
from hashlib import sha256
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from . import contract
from ._infra.orpo_data import Sid
from .reward import ObjectiveRoute
from .source_blocks import SOURCE_BLOCKS, build_source_blocks


SourceIdentity = tuple[int, str, str]


@dataclass(frozen=True)
class SourceVerificationResult:
    source_groups: int
    candidate_rollouts: int
    completed_source_blocks: int
    source_plan_sha256: str
    epoch_plan_sha256s: tuple[str, ...]
    source_memberships: tuple[SourceIdentity, ...]
    source_locations: tuple[tuple[int, int, int, str, str], ...]
    rl_memberships: tuple[SourceIdentity, ...]
    anchor_memberships: tuple[SourceIdentity, ...]


@dataclass(frozen=True)
class WindowVerificationResult:
    optimization_windows: int
    policy_optimizer_steps: int
    rl_groups: int
    anchor_groups: int
    policy_memberships: tuple[tuple[int, int, str, str], ...]
    anchor_memberships: tuple[tuple[int, int, str, str], ...]
    window_ranges: tuple[tuple[int, int, int], ...]
    policy_steps_by_window: tuple[int, ...]
    next_source_block: int


@dataclass(frozen=True)
class RunVerificationResult:
    source: SourceVerificationResult
    windows: WindowVerificationResult
    anchor_groups: int


@dataclass(frozen=True)
class AnchorVerificationResult:
    anchor_groups: int
    memberships: tuple[SourceIdentity, ...]
    merged_memberships: tuple[SourceIdentity, ...]
    flush_memberships: tuple[SourceIdentity, ...]


@dataclass(frozen=True)
class _SourcePlan:
    sha256: str
    epoch_sha256s: tuple[str, ...]
    expected_rows: tuple[tuple[int, int, int, str, int, str], ...]


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer.")
    return result


def _require_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has an invalid schema.")


def _ordered_group_ids(
    group_ids: Sequence[str], *, seed: int, epoch_index: int, task: str
) -> list[str]:
    return sorted(
        (str(group_id) for group_id in group_ids),
        key=lambda group_id: (
            sha256(
                f"dapo-anchor-v1.1|{seed}|{epoch_index}|{task}|{group_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
            group_id,
        ),
    )


def _validate_source_epoch_plan(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    recommendation_group_ids: Sequence[str] | None = None,
    text_to_sid_group_ids: Sequence[str] | None = None,
) -> _SourcePlan:
    """Rebuild every ordered source row from the persisted deterministic plan."""

    required = {
        "schema_version",
        "seed",
        "source_epochs",
        "source_blocks_per_epoch",
        "epochs",
        "sha256",
    }
    _require_keys(plan, required, name="source_epoch_plan.json")
    if _as_int(plan["schema_version"], name="source plan schema_version") != 1:
        raise ValueError("source_epoch_plan.json has an unsupported schema version.")
    seed = _as_int(plan["seed"], name="source plan seed")
    if seed != _as_int(config["train"]["seed"], name="train.seed"):
        raise ValueError("source_epoch_plan.json seed differs from resolved_config.json.")
    if _as_int(plan["source_epochs"], name="source plan source_epochs") != contract.SOURCE_EPOCHS:
        raise ValueError("source_epoch_plan.json must contain exactly two epochs.")
    if (
        _as_int(plan["source_blocks_per_epoch"], name="source plan blocks")
        != SOURCE_BLOCKS
    ):
        raise ValueError("source_epoch_plan.json has the wrong blocks-per-epoch value.")
    if not _is_sha256(plan["sha256"]):
        raise ValueError("source_epoch_plan.json has an invalid SHA-256.")
    epochs = plan["epochs"]
    if not isinstance(epochs, list) or len(epochs) != contract.SOURCE_EPOCHS:
        raise ValueError("source_epoch_plan.json must contain two epoch records.")

    expected_rec = None if recommendation_group_ids is None else tuple(
        str(value) for value in recommendation_group_ids
    )
    expected_text = None if text_to_sid_group_ids is None else tuple(
        str(value) for value in text_to_sid_group_ids
    )
    if expected_rec is not None and len(expected_rec) != contract.EXPECTED_RECOMMENDATION_GROUPS:
        raise ValueError("Supplied recommendation source IDs differ from the frozen count.")
    if expected_text is not None and len(expected_text) != contract.EXPECTED_TEXT_GROUPS:
        raise ValueError("Supplied text-to-SID source IDs differ from the frozen count.")

    expected_rows: list[tuple[int, int, int, str, int, str]] = []
    epoch_hashes: list[str] = []
    normalized_epochs: list[dict[str, Any]] = []
    epoch_required = {
        "epoch_index",
        "sha256",
        "recommendation_source_indices",
        "recommendation_group_ids",
        "text_to_sid_source_indices",
        "text_to_sid_group_ids",
    }
    for expected_epoch, record in enumerate(epochs):
        if not isinstance(record, Mapping):
            raise ValueError("source_epoch_plan.json epoch record must be an object.")
        _require_keys(record, epoch_required, name="source_epoch_plan.json epoch record")
        epoch_index = _as_int(record["epoch_index"], name="source plan epoch_index")
        if epoch_index != expected_epoch:
            raise ValueError("source_epoch_plan.json epoch indices must be [0, 1].")
        if not _is_sha256(record["sha256"]):
            raise ValueError("source_epoch_plan.json epoch SHA-256 is invalid.")

        def read_task(
            prefix: str, count: int, expected_ids: tuple[str, ...] | None
        ) -> tuple[list[int], list[str]]:
            indices_value = record[f"{prefix}_source_indices"]
            ids_value = record[f"{prefix}_group_ids"]
            if not isinstance(indices_value, list) or not isinstance(ids_value, list):
                raise ValueError("source_epoch_plan.json task order must be lists.")
            if len(indices_value) != count or len(ids_value) != count:
                raise ValueError("source_epoch_plan.json task order has the wrong length.")
            indices = [
                _as_int(value, name=f"{prefix} source index")
                for value in indices_value
            ]
            ids = [str(value) for value in ids_value]
            if any(not value for value in ids) or len(set(ids)) != count:
                raise ValueError("source_epoch_plan.json task group IDs must be unique.")
            if set(indices) != set(range(count)):
                raise ValueError("source_epoch_plan.json task source indices are not a permutation.")
            if ids != _ordered_group_ids(
                ids, seed=seed, epoch_index=epoch_index, task=(
                    "recommendation" if prefix == "recommendation" else "item_text_to_sid"
                )
            ):
                raise ValueError("source_epoch_plan.json task order is not the seeded SHA-256 order.")
            if expected_ids is not None and any(
                expected_ids[source_index] != group_id
                for source_index, group_id in zip(indices, ids, strict=True)
            ):
                raise ValueError("source_epoch_plan.json does not bind group IDs to frozen source indices.")
            return indices, ids

        recommendation_indices, recommendation_ids = read_task(
            "recommendation", contract.EXPECTED_RECOMMENDATION_GROUPS, expected_rec
        )
        text_indices, text_ids = read_task(
            "text_to_sid", contract.EXPECTED_TEXT_GROUPS, expected_text
        )
        epoch_payload = {
            "schema_version": 1,
            "epoch_index": epoch_index,
            "recommendation_group_ids": recommendation_ids,
            "text_to_sid_group_ids": text_ids,
        }
        if _canonical_sha256(epoch_payload) != record["sha256"]:
            raise ValueError("source_epoch_plan.json epoch hash does not match its order.")
        epoch_hashes.append(str(record["sha256"]))
        normalized_epochs.append(
            {
                "epoch_index": epoch_index,
                "sha256": str(record["sha256"]),
                "recommendation_source_indices": recommendation_indices,
                "recommendation_group_ids": recommendation_ids,
                "text_to_sid_source_indices": text_indices,
                "text_to_sid_group_ids": text_ids,
            }
        )
        for block in build_source_blocks(
            recommendation_groups=contract.EXPECTED_RECOMMENDATION_GROUPS,
            text_to_sid_groups=contract.EXPECTED_TEXT_GROUPS,
            blocks=SOURCE_BLOCKS,
            epoch_index=epoch_index,
            global_offset=epoch_index * SOURCE_BLOCKS,
        ):
            for position in range(block.recommendation.start, block.recommendation.stop):
                expected_rows.append(
                    (
                        block.index,
                        epoch_index,
                        block.epoch_block_index,
                        "recommendation",
                        recommendation_indices[position],
                        recommendation_ids[position],
                    )
                )
            for position in range(block.text_to_sid.start, block.text_to_sid.stop):
                expected_rows.append(
                    (
                        block.index,
                        epoch_index,
                        block.epoch_block_index,
                        "item_text_to_sid",
                        text_indices[position],
                        text_ids[position],
                    )
                )
    top_payload = {
        "schema_version": 1,
        "seed": seed,
        "source_epochs": contract.SOURCE_EPOCHS,
        "source_blocks_per_epoch": SOURCE_BLOCKS,
        "epochs": normalized_epochs,
    }
    if _canonical_sha256(top_payload) != plan["sha256"]:
        raise ValueError("source_epoch_plan.json hash does not match its epoch plans.")
    if len(expected_rows) != contract.TOTAL_SOURCE_GROUPS:
        raise RuntimeError("Rebuilt source plan differs from the frozen two-epoch count.")
    return _SourcePlan(
        sha256=str(plan["sha256"]),
        epoch_sha256s=tuple(epoch_hashes),
        expected_rows=tuple(expected_rows),
    )


def validate_source_group_rows(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    plan_manifest: Mapping[str, Any],
    recommendation_group_ids: Sequence[str] | None = None,
    text_to_sid_group_ids: Sequence[str] | None = None,
) -> SourceVerificationResult:
    """Verify exactly two normalized source epochs and universal Anchor routing."""

    if int(config["train"]["source_epochs"]) != contract.SOURCE_EPOCHS:
        raise ValueError("V1.1 requires exactly two source epochs.")
    if int(config["data"]["source_blocks"]) != SOURCE_BLOCKS:
        raise ValueError("V1.1 requires exactly 532 source blocks per epoch.")
    if not bool(config["experiments"]["active"]["text_to_sid_enabled"]):
        raise ValueError("V1.1 complete-run verification requires the multitask arm.")
    plan = _validate_source_epoch_plan(
        plan_manifest,
        config,
        recommendation_group_ids=recommendation_group_ids,
        text_to_sid_group_ids=text_to_sid_group_ids,
    )
    values = list(rows)
    if len(values) != contract.TOTAL_SOURCE_GROUPS:
        raise ValueError("Source log does not contain exactly 55,226 normalized groups.")
    expected_row_keys = {
        "block_index",
        "epoch_index",
        "epoch_block_index",
        "task",
        "source_index",
        "group_id",
        "rollout_count",
        "route",
        "policy_step_at_sample",
    }
    observed: list[tuple[int, int, int, str, int, str]] = []
    memberships: list[SourceIdentity] = []
    locations: list[tuple[int, int, int, str, str]] = []
    rl_memberships: list[SourceIdentity] = []
    previous_policy_step = 0
    allowed_routes = {ObjectiveRoute.ANCHOR, ObjectiveRoute.RL_AND_ANCHOR}
    for index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ValueError(f"Source row {index} is not an object.")
        _require_keys(row, expected_row_keys, name=f"Source row {index}")
        try:
            block_index = _as_int(row["block_index"], name="source block_index")
            epoch_index = _as_int(row["epoch_index"], name="source epoch_index")
            epoch_block_index = _as_int(
                row["epoch_block_index"], name="source epoch_block_index"
            )
            task = str(row["task"])
            source_index = _as_int(row["source_index"], name="source source_index")
            group_id = str(row["group_id"])
            rollout_count = _as_int(row["rollout_count"], name="source rollout_count")
            policy_step = _as_int(
                row["policy_step_at_sample"], name="source policy_step_at_sample"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source row {index}.") from exc
        route_value = str(row["route"])
        if route_value not in {item.value for item in ObjectiveRoute}:
            raise ValueError("Every V1.1 source group must enter Anchor; skip/RL-only is forbidden.")
        route = ObjectiveRoute(route_value)
        if rollout_count != contract.GROUP_SIZE:
            raise ValueError(
                f"Source row {index} must contain {contract.GROUP_SIZE} rollouts."
            )
        if policy_step < 0 or policy_step < previous_policy_step:
            raise ValueError("Source policy-step snapshots must be non-negative and ordered.")
        previous_policy_step = policy_step
        if route not in allowed_routes:
            raise ValueError("Every V1.1 source group must enter Anchor; skip/RL-only is forbidden.")
        observed.append(
            (block_index, epoch_index, epoch_block_index, task, source_index, group_id)
        )
        identity = (epoch_index, task, group_id)
        memberships.append(identity)
        locations.append((block_index, epoch_index, epoch_block_index, task, group_id))
        if route is ObjectiveRoute.RL_AND_ANCHOR:
            rl_memberships.append(identity)
    if tuple(observed) != plan.expected_rows:
        raise ValueError(
            "Observed source log differs from the persisted two-epoch SHA-256 plan."
        )
    if len(set(memberships)) != contract.TOTAL_SOURCE_GROUPS:
        raise ValueError("Epoch-keyed source identity appears more than once.")
    return SourceVerificationResult(
        source_groups=len(values),
        candidate_rollouts=len(values) * contract.GROUP_SIZE,
        completed_source_blocks=contract.TOTAL_SOURCE_BLOCKS,
        source_plan_sha256=plan.sha256,
        epoch_plan_sha256s=plan.epoch_sha256s,
        source_memberships=tuple(memberships),
        source_locations=tuple(locations),
        rl_memberships=tuple(rl_memberships),
        anchor_memberships=tuple(memberships),
    )


def _finite(value: Any, *, name: str, non_negative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number) or (non_negative and number < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return number


def validate_window_rows(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> WindowVerificationResult:
    """Verify K=1 updates, epoch-keyed identities, and merged Anchor budgets."""

    low = _finite(config["loss"]["clip_ratio_low"], name="clip low")
    high = _finite(config["loss"]["clip_ratio_high"], name="clip high")
    if low != contract.CLIP_RATIO_LOW or high != contract.CLIP_RATIO_HIGH:
        raise ValueError("Asymmetric clip bounds must be [0.8, 1.28].")
    max_batch = int(config["loss"]["minibatch_groups"])
    target_ratio = float(config["anchor"]["target_gradient_ratio"])
    max_anchor_weight = float(config["anchor"]["max_weight"])
    max_policy_replay = float(config["rollout"]["canonical_replay_max_logp_difference"])
    max_anchor_replay = float(config["anchor"]["replay_max_logp_difference"])
    epsilon = float(config["anchor"]["gradient_epsilon"])
    if epsilon <= 0.0:
        raise ValueError("Anchor gradient epsilon must be positive.")
    window_keys = {
        "optimization_window_index", "first_source_block", "next_source_block",
        "snapshot_policy_step", "completed_policy_steps", "rl_groups", "anchor_groups",
        "policy_optimizer_steps", "anchor_optimizer_steps", "policy_gradient_norms",
        "rl_reference_gradient_norm", "lambda_cap", "lambda_effective",
        "anchor_raw_gradient_norm", "anchor_scaled_gradient_norm",
        "anchor_to_rl_gradient_ratio", "combined_gradient_norm", "steps",
        "anchor_group_results", "anchor_max_replay_logp_difference", "k",
    }
    step_keys = {
        "loss", "learning_rate", "decision_tokens", "max_replay_logp_difference",
        "ratio_min", "ratio_max", "ratio_mean", "clipped_tokens",
        "below_low_tokens", "above_high_tokens", "group_ids", "tasks", "epoch_indexes",
    }
    anchor_result_keys = {
        "epoch_index", "source_block", "epoch_block_index", "task", "group_id",
        "set_nll", "all_gt_decision_nll_sum", "gt_sid_count", "decision_tokens",
    }
    values = list(rows)
    seen_policy: set[SourceIdentity] = set()
    seen_anchor: set[SourceIdentity] = set()
    policy_memberships: list[tuple[int, int, str, str]] = []
    anchor_memberships: list[tuple[int, int, str, str]] = []
    ranges: list[tuple[int, int, int]] = []
    policy_steps_by_window: list[int] = []
    total_steps = 0
    total_rl_groups = 0
    total_anchor_groups = 0
    expected_policy_cursor = 0
    previous_next_source_block = 0
    for row_index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ValueError("Optimization window must be an object.")
        _require_keys(row, window_keys, name="Optimization window")
        if _as_int(row["optimization_window_index"], name="optimization_window_index") != row_index:
            raise ValueError("Optimization window indices must be contiguous from zero.")
        first_block = _as_int(row["first_source_block"], name="first_source_block")
        next_block = _as_int(row["next_source_block"], name="next_source_block")
        if not 0 <= first_block < next_block <= contract.TOTAL_SOURCE_BLOCKS:
            raise ValueError("Optimization window source-block range is invalid.")
        if first_block < previous_next_source_block:
            raise ValueError("Optimization windows overlap in source-block coverage.")
        previous_next_source_block = next_block
        ranges.append((row_index, first_block, next_block))
        if _as_int(row["k"], name="K") != 1:
            raise ValueError("K=1 is required for every optimization window.")
        if _as_int(row["anchor_optimizer_steps"], name="anchor_optimizer_steps") != 0:
            raise ValueError("Anchor must not use an independent optimizer step.")
        policy_steps = _as_int(row["policy_optimizer_steps"], name="policy_optimizer_steps")
        steps = row["steps"]
        if not isinstance(steps, list) or policy_steps <= 0 or len(steps) != policy_steps:
            raise ValueError("policy_optimizer_steps must equal the logged minibatches.")
        snapshot = _as_int(row["snapshot_policy_step"], name="snapshot_policy_step")
        completed = _as_int(row["completed_policy_steps"], name="completed_policy_steps")
        if snapshot != expected_policy_cursor or completed != snapshot + policy_steps:
            raise ValueError("Policy-step cursors are inconsistent across windows.")
        expected_policy_cursor = completed
        norms_value = row["policy_gradient_norms"]
        if not isinstance(norms_value, list) or len(norms_value) != policy_steps:
            raise ValueError("Every policy step needs one raw gradient norm.")
        norms = [_finite(value, name="policy gradient norm", non_negative=True) for value in norms_value]
        reference = _finite(row["rl_reference_gradient_norm"], name="RL reference gradient norm", non_negative=True)
        if not math.isclose(reference, float(median(norms)), rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("RL reference gradient norm must be the policy-step median.")

        window_groups: list[SourceIdentity] = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError("Each policy step must be a mapping.")
            _require_keys(step, step_keys, name="Policy minibatch")
            ids, tasks, epochs = step["group_ids"], step["tasks"], step["epoch_indexes"]
            if not isinstance(ids, (list, tuple)) or not isinstance(tasks, (list, tuple)) or not isinstance(epochs, (list, tuple)):
                raise ValueError("Every policy minibatch must log task/group/epoch identities.")
            if not 1 <= len(ids) <= max_batch or len(tasks) != len(ids) or len(epochs) != len(ids):
                raise ValueError("Policy minibatches must contain 1..8 epoch/task/group tuples.")
            if step_index + 1 < policy_steps and len(ids) != max_batch:
                raise ValueError("Only the final policy minibatch may contain fewer than eight groups.")
            identities = [
                (_as_int(epoch, name="policy epoch_index"), str(task), str(group_id))
                for epoch, task, group_id in zip(epochs, tasks, ids, strict=True)
            ]
            if any(epoch not in range(contract.SOURCE_EPOCHS) for epoch, _, _ in identities):
                raise ValueError("Policy minibatch has an invalid epoch index.")
            if len(set(identities)) != len(identities) or any(identity in seen_policy for identity in identities):
                raise ValueError("K=1 forbids reusing an epoch-keyed policy group.")
            seen_policy.update(identities)
            window_groups.extend(identities)
            policy_memberships.extend((row_index, *identity) for identity in identities)
            decision_tokens = _as_int(step["decision_tokens"], name="decision_tokens")
            clipped = _as_int(step["clipped_tokens"], name="clipped_tokens")
            below = _as_int(step["below_low_tokens"], name="below_low_tokens")
            above = _as_int(step["above_high_tokens"], name="above_high_tokens")
            if min(decision_tokens, clipped, below, above) < 0:
                raise ValueError("Policy token counters must be non-negative.")
            if clipped > below + above or below + above > decision_tokens:
                raise ValueError("Clip token counters are inconsistent.")
            ratio_min = _finite(step["ratio_min"], name="ratio_min")
            ratio_mean = _finite(step["ratio_mean"], name="ratio_mean")
            ratio_max = _finite(step["ratio_max"], name="ratio_max")
            if not 0.0 < ratio_min <= ratio_mean <= ratio_max:
                raise ValueError("Policy ratios must be positive and ordered.")
            if ratio_min < low and below == 0:
                raise ValueError("A ratio below 0.8 must increment below_low_tokens.")
            if ratio_max > high and above == 0:
                raise ValueError("A ratio above 1.28 must increment above_high_tokens.")
            loss = _finite(step["loss"], name="loss")
            del loss
            learning_rate = _finite(step["learning_rate"], name="learning_rate", non_negative=True)
            per_level = int(config["train"]["warmup_steps_per_level"])
            levels = int(config["train"]["warmup_policy_steps"]) // per_level
            policy_step = snapshot + step_index + 1
            level = min((policy_step - 1) // per_level + 1, levels)
            expected_lr = float(config["train"]["learning_rate"]) * level / levels
            if not math.isclose(learning_rate, expected_lr, rel_tol=1.0e-9, abs_tol=1.0e-15):
                raise ValueError("Policy-step learning rate differs from the frozen schedule.")
            replay = _finite(step["max_replay_logp_difference"], name="max_replay_logp_difference", non_negative=True)
            if step_index == 0 and replay > max_policy_replay:
                raise ValueError("The first policy minibatch replay logp difference exceeds the frozen-policy gate.")
        if _as_int(row["rl_groups"], name="rl_groups") != len(window_groups):
            raise ValueError("rl_groups differs from the policy minibatch membership.")

        raw = _finite(row["anchor_raw_gradient_norm"], name="Anchor raw gradient norm", non_negative=True)
        scaled = _finite(row["anchor_scaled_gradient_norm"], name="Anchor scaled gradient norm", non_negative=True)
        cap = _finite(row["lambda_cap"], name="lambda_cap", non_negative=True)
        effective = _finite(row["lambda_effective"], name="lambda_effective", non_negative=True)
        ratio = _finite(row["anchor_to_rl_gradient_ratio"], name="Anchor/RL gradient ratio", non_negative=True)
        _finite(row["combined_gradient_norm"], name="combined gradient norm", non_negative=True)
        if effective > min(cap, max_anchor_weight) + 1.0e-12:
            raise ValueError("lambda_effective exceeds its configured/cap limit.")
        if not math.isclose(scaled, raw * effective, rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("Scaled Anchor gradient norm is inconsistent.")
        if reference > 0.0 and raw > 0.0:
            expected_cap = target_ratio * reference / max(raw, epsilon)
            if not math.isclose(cap, expected_cap, rel_tol=1.0e-6, abs_tol=1.0e-12):
                raise ValueError("lambda_cap is inconsistent with the 10% budget.")
            if not math.isclose(ratio, scaled / reference, rel_tol=1.0e-6, abs_tol=1.0e-12):
                raise ValueError("Anchor/RL gradient ratio is inconsistent.")
        elif any(value != 0.0 for value in (cap, effective, scaled, ratio)):
            raise ValueError("Zero RL or Anchor gradient must disable the Anchor merge.")
        if ratio > target_ratio + 1.0e-12:
            raise ValueError("Scaled Anchor gradient exceeds 10% of the RL reference gradient.")

        anchor_results = row["anchor_group_results"]
        anchor_count = _as_int(row["anchor_groups"], name="anchor_groups")
        if not isinstance(anchor_results, list) or len(anchor_results) != anchor_count:
            raise ValueError("anchor_groups differs from anchor_group_results.")
        anchor_ids: list[SourceIdentity] = []
        for item in anchor_results:
            if not isinstance(item, Mapping):
                raise ValueError("Every Anchor group result must be a mapping.")
            _require_keys(item, anchor_result_keys, name="Anchor group result")
            epoch_index = _as_int(item["epoch_index"], name="Anchor epoch_index")
            source_block = _as_int(item["source_block"], name="Anchor source_block")
            epoch_block = _as_int(item["epoch_block_index"], name="Anchor epoch_block_index")
            if not 0 <= source_block < contract.TOTAL_SOURCE_BLOCKS or epoch_index != source_block // SOURCE_BLOCKS or epoch_block != source_block % SOURCE_BLOCKS:
                raise ValueError("Anchor group result has inconsistent epoch/source-block counters.")
            if not first_block <= source_block < next_block:
                raise ValueError("Anchor group lies outside its policy window source range.")
            identity = (epoch_index, str(item["task"]), str(item["group_id"]))
            anchor_ids.append(identity)
            _finite(item["set_nll"], name="Anchor set NLL", non_negative=True)
            _finite(item["all_gt_decision_nll_sum"], name="Anchor all-GT decision NLL", non_negative=True)
            if _as_int(item["gt_sid_count"], name="Anchor gt_sid_count") <= 0 or _as_int(item["decision_tokens"], name="Anchor decision_tokens") <= 0:
                raise ValueError("Every Anchor group must score GT decision tokens.")
        if len(set(anchor_ids)) != len(anchor_ids) or any(identity in seen_anchor for identity in anchor_ids):
            raise ValueError("An epoch-keyed source group entered merged Anchor more than once.")
        if not set(window_groups).issubset(set(anchor_ids)):
            raise ValueError("Every RL group must also enter Anchor in V1.1.")
        seen_anchor.update(anchor_ids)
        anchor_memberships.extend((row_index, *identity) for identity in anchor_ids)
        replay = _finite(row["anchor_max_replay_logp_difference"], name="Anchor replay logp difference", non_negative=True)
        if replay > max_anchor_replay:
            raise ValueError("Anchor replay logp difference exceeds the frozen gate.")
        total_steps += policy_steps
        policy_steps_by_window.append(policy_steps)
        total_rl_groups += len(window_groups)
        total_anchor_groups += anchor_count
    return WindowVerificationResult(
        optimization_windows=len(values),
        policy_optimizer_steps=total_steps,
        rl_groups=total_rl_groups,
        anchor_groups=total_anchor_groups,
        policy_memberships=tuple(policy_memberships),
        anchor_memberships=tuple(anchor_memberships),
        window_ranges=tuple(ranges),
        policy_steps_by_window=tuple(policy_steps_by_window),
        next_source_block=previous_next_source_block,
    )


def _validate_group_rows(
    rows: Iterable[Mapping[str, Any]], windows: WindowVerificationResult, config: Mapping[str, Any]
) -> tuple[int, set[SourceIdentity]]:
    values = list(rows)
    expected_keys = {
        "optimization_window_index", "source_block", "epoch_index", "epoch_block_index",
        "source_index", "task", "group_id", "route", "rewards", "advantages",
        "tiers", "candidate_sids", "gt_injection_count", "k",
    }
    observed: list[tuple[int, SourceIdentity]] = []
    allowed_rewards = {float(value) for value in config["reward"].values()}
    for index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ValueError(f"RL group row {index} is not an object.")
        _require_keys(row, expected_keys, name=f"RL group row {index}")
        window_index = _as_int(row["optimization_window_index"], name="RL window index")
        source_block = _as_int(row["source_block"], name="RL source_block")
        epoch_index = _as_int(row["epoch_index"], name="RL epoch_index")
        epoch_block_index = _as_int(row["epoch_block_index"], name="RL epoch_block_index")
        source_index = _as_int(row["source_index"], name="RL source_index")
        if window_index < 0 or source_index < 0 or not 0 <= source_block < contract.TOTAL_SOURCE_BLOCKS:
            raise ValueError(f"RL group row {index} has invalid source counters.")
        if epoch_index != source_block // SOURCE_BLOCKS or epoch_block_index != source_block % SOURCE_BLOCKS:
            raise ValueError("RL group epoch/source-block counters are inconsistent.")
        identity = (epoch_index, str(row["task"]), str(row["group_id"]))
        if str(row["route"]) != ObjectiveRoute.RL_AND_ANCHOR.value:
            raise ValueError("Every RL group must use the RL_AND_ANCHOR route in V1.1.")
        if _as_int(row["k"], name="RL K") != 1:
            raise ValueError("RL group log violates K=1.")
        if _as_int(row["gt_injection_count"], name="GT injection count") != 0:
            raise ValueError("RL candidates must not contain GT injection.")
        sequences = [row["rewards"], row["advantages"], row["tiers"], row["candidate_sids"]]
        if any(not isinstance(value, list) or len(value) != contract.GROUP_SIZE for value in sequences):
            raise ValueError("Every RL group must retain all 16 candidate records.")
        rewards = [_finite(value, name="reward") for value in row["rewards"]]
        advantages = [_finite(value, name="advantage") for value in row["advantages"]]
        if any(not any(math.isclose(value, allowed, rel_tol=0.0, abs_tol=1.0e-6) for allowed in allowed_rewards) for value in rewards):
            raise ValueError("RL group contains a reward outside the active arm table.")
        if not math.isclose(sum(advantages), 0.0, abs_tol=1.0e-5):
            raise ValueError("RLOO group advantages must sum to zero.")
        observed.append((window_index, identity))
    if len(set(observed)) != len(observed):
        raise ValueError("groups.jsonl repeats an epoch-keyed K=1 policy group.")
    expected = set(windows.policy_memberships)
    if set((window_index, *identity) for window_index, identity in observed) != expected:
        raise ValueError("groups.jsonl membership differs from windows.jsonl.")
    return len(values), {identity for _, identity in observed}


def validate_anchor_group_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: SourceVerificationResult,
    windows: WindowVerificationResult,
) -> AnchorVerificationResult:
    """Require one persisted GT-set NLL result for every normalized source group."""

    expected_keys = {
        "epoch_index", "source_block", "epoch_block_index", "task", "group_id",
        "set_nll", "all_gt_decision_nll_sum", "gt_sid_count", "decision_tokens",
        "phase", "unit_index",
    }
    locations = {
        (epoch_index, task, group_id): (block_index, epoch_block_index)
        for block_index, epoch_index, epoch_block_index, task, group_id in source.source_locations
    }
    merged_by_window: dict[int, set[SourceIdentity]] = {}
    for window_index, epoch_index, task, group_id in windows.anchor_memberships:
        merged_by_window.setdefault(window_index, set()).add((epoch_index, task, group_id))
    values = list(rows)
    memberships: list[SourceIdentity] = []
    merged: list[SourceIdentity] = []
    flushed: list[SourceIdentity] = []
    for index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ValueError(f"Anchor log row {index} is not an object.")
        _require_keys(row, expected_keys, name=f"Anchor log row {index}")
        epoch_index = _as_int(row["epoch_index"], name="Anchor log epoch_index")
        source_block = _as_int(row["source_block"], name="Anchor log source_block")
        epoch_block = _as_int(row["epoch_block_index"], name="Anchor log epoch_block_index")
        identity = (epoch_index, str(row["task"]), str(row["group_id"]))
        expected_location = locations.get(identity)
        if expected_location is None:
            raise ValueError("Anchor log contains a group absent from the source plan.")
        if expected_location != (source_block, epoch_block):
            raise ValueError("Anchor log source counters differ from the source plan.")
        _finite(row["set_nll"], name="Anchor log set_nll", non_negative=True)
        _finite(row["all_gt_decision_nll_sum"], name="Anchor log all-GT NLL", non_negative=True)
        if _as_int(row["gt_sid_count"], name="Anchor log gt_sid_count") <= 0 or _as_int(row["decision_tokens"], name="Anchor log decision_tokens") <= 0:
            raise ValueError("Every Anchor log row must score at least one GT decision token.")
        phase = str(row["phase"])
        unit_index = _as_int(row["unit_index"], name="Anchor log unit_index")
        if phase == "merged_policy_window":
            if identity not in merged_by_window.get(unit_index, set()):
                raise ValueError("Merged Anchor log membership differs from windows.jsonl.")
            merged.append(identity)
        elif phase == "epoch_end_anchor_flush":
            if unit_index != epoch_index:
                raise ValueError("Epoch-end Anchor log must use its epoch as unit_index.")
            flushed.append(identity)
        else:
            raise ValueError("Anchor log phase is not in the V1.1 contract.")
        memberships.append(identity)
    expected_memberships = set(source.anchor_memberships)
    if len(memberships) != len(set(memberships)):
        raise ValueError("An epoch-keyed source group entered Anchor more than once.")
    if set(memberships) != expected_memberships:
        raise ValueError("anchor_groups.jsonl does not cover every normalized source group once.")
    expected_merged = {
        (epoch_index, task, group_id)
        for _, epoch_index, task, group_id in windows.anchor_memberships
    }
    if set(merged) != expected_merged:
        raise ValueError("Merged Anchor log and windows.jsonl do not have identical membership.")
    return AnchorVerificationResult(
        anchor_groups=len(memberships),
        memberships=tuple(memberships),
        merged_memberships=tuple(merged),
        flush_memberships=tuple(flushed),
    )


def _validate_epoch_auxiliary_flush_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: SourceVerificationResult,
    windows: WindowVerificationResult,
    anchors: AnchorVerificationResult,
    config: Mapping[str, Any],
) -> int:
    """Validate the at-most-one Anchor-only optimizer step at each epoch end."""

    required = {
        "epoch_index", "first_source_block", "next_source_block",
        "snapshot_policy_step", "anchor_groups", "gt_sid_rows", "decision_tokens",
        "optimizer_steps", "scheduler_advanced", "rl_reference_gradient_norm",
        "lambda_cap", "lambda_effective", "anchor_raw_gradient_norm",
        "anchor_scaled_gradient_norm", "anchor_to_rl_gradient_ratio",
        "combined_gradient_norm", "anchor_set_nll_mean",
        "anchor_max_replay_logp_difference",
    }
    by_epoch: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Epoch auxiliary row {index} is not an object.")
        _require_keys(row, required, name=f"Epoch auxiliary row {index}")
        epoch_index = _as_int(row["epoch_index"], name="auxiliary epoch_index")
        if epoch_index not in range(contract.SOURCE_EPOCHS) or epoch_index in by_epoch:
            raise ValueError("At most one Anchor-only flush is allowed per epoch.")
        by_epoch[epoch_index] = row

    locations = {
        (epoch_index, task, group_id): block_index
        for block_index, epoch_index, _, task, group_id in source.source_locations
    }
    merged = set(anchors.merged_memberships)
    flushed = set(anchors.flush_memberships)
    target_ratio = float(config["anchor"]["target_gradient_ratio"])
    max_weight = float(config["anchor"]["max_weight"])
    epsilon = float(config["anchor"]["gradient_epsilon"])
    replay_limit = float(config["anchor"]["replay_max_logp_difference"])
    for epoch_index in range(contract.SOURCE_EPOCHS):
        epoch_start = epoch_index * SOURCE_BLOCKS
        epoch_end = epoch_start + SOURCE_BLOCKS
        source_epoch = {
            identity for identity in source.anchor_memberships if identity[0] == epoch_index
        }
        expected_flush = source_epoch - merged
        actual_flush = {identity for identity in flushed if identity[0] == epoch_index}
        row = by_epoch.get(epoch_index)
        if not expected_flush:
            if row is not None or actual_flush:
                raise ValueError("An epoch without residual Anchor groups cannot flush.")
            continue
        if row is None:
            raise ValueError("Residual Anchor groups require one epoch-end flush record.")
        if actual_flush != expected_flush:
            raise ValueError("Epoch-end Anchor log coverage differs from residual source groups.")
        residual_blocks = {locations[identity] for identity in expected_flush}
        first_block = min(residual_blocks)
        if residual_blocks != set(range(first_block, epoch_end)):
            raise ValueError("Epoch-end Anchor-only coverage must be a complete suffix of the epoch.")
        if _as_int(row["first_source_block"], name="auxiliary first_source_block") != first_block or _as_int(row["next_source_block"], name="auxiliary next_source_block") != epoch_end:
            raise ValueError("Epoch auxiliary source-block range differs from the residual suffix.")
        completed_before_flush = sum(
            windows.policy_steps_by_window[window_index]
            for window_index, _, next_block in windows.window_ranges
            if next_block <= epoch_end
        )
        if _as_int(row["snapshot_policy_step"], name="auxiliary snapshot_policy_step") != completed_before_flush:
            raise ValueError("Epoch auxiliary policy-step snapshot is inconsistent.")
        if _as_int(row["anchor_groups"], name="auxiliary anchor_groups") != len(expected_flush):
            raise ValueError("Epoch auxiliary anchor_groups differs from residual coverage.")
        if _as_int(row["gt_sid_rows"], name="auxiliary gt_sid_rows") < len(expected_flush) or _as_int(row["decision_tokens"], name="auxiliary decision_tokens") <= 0:
            raise ValueError("Epoch auxiliary record has no complete GT-set coverage.")
        if _as_int(row["optimizer_steps"], name="auxiliary optimizer_steps") != 1 or row["scheduler_advanced"] is not False:
            raise ValueError("Epoch auxiliary flush must take one optimizer step without advancing the scheduler.")
        reference = _finite(row["rl_reference_gradient_norm"], name="auxiliary RL reference", non_negative=True)
        raw = _finite(row["anchor_raw_gradient_norm"], name="auxiliary raw gradient", non_negative=True)
        scaled = _finite(row["anchor_scaled_gradient_norm"], name="auxiliary scaled gradient", non_negative=True)
        cap = _finite(row["lambda_cap"], name="auxiliary lambda_cap", non_negative=True)
        effective = _finite(row["lambda_effective"], name="auxiliary lambda_effective", non_negative=True)
        ratio = _finite(row["anchor_to_rl_gradient_ratio"], name="auxiliary gradient ratio", non_negative=True)
        _finite(row["combined_gradient_norm"], name="auxiliary combined gradient", non_negative=True)
        _finite(row["anchor_set_nll_mean"], name="auxiliary Anchor set NLL", non_negative=True)
        replay = _finite(row["anchor_max_replay_logp_difference"], name="auxiliary replay difference", non_negative=True)
        if reference <= 0.0 or raw <= 0.0 or effective <= 0.0:
            raise ValueError("Epoch auxiliary flush requires positive RL reference, Anchor gradient, and lambda.")
        if replay > replay_limit:
            raise ValueError("Epoch auxiliary replay difference exceeds the frozen gate.")
        expected_cap = target_ratio * reference / max(raw, epsilon)
        if not math.isclose(cap, expected_cap, rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("Epoch auxiliary lambda cap is inconsistent with the 10% budget.")
        if effective > min(cap, max_weight) + 1.0e-12 or not math.isclose(scaled, raw * effective, rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("Epoch auxiliary scaled Anchor gradient is inconsistent.")
        if not math.isclose(ratio, scaled / reference, rel_tol=1.0e-6, abs_tol=1.0e-12) or ratio > target_ratio + 1.0e-12:
            raise ValueError("Epoch auxiliary Anchor gradient exceeds its budget.")
    return len(rows)


def validate_run_summary(
    summary: Mapping[str, Any],
    *,
    source: SourceVerificationResult,
    windows: WindowVerificationResult,
    group_rows: Iterable[Mapping[str, Any]],
    anchor_group_rows: Iterable[Mapping[str, Any]],
    epoch_auxiliary_flush_rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """Cross-check full source, RL, Anchor, and epoch-end flush coverage."""

    rl_groups, group_memberships = _validate_group_rows(group_rows, windows, config)
    source_rl = set(source.rl_memberships)
    window_rl = {(epoch, task, group_id) for _, epoch, task, group_id in windows.policy_memberships}
    if source_rl != window_rl or source_rl != group_memberships:
        raise ValueError("Source routes and RL optimization membership differ.")
    anchors = validate_anchor_group_rows(anchor_group_rows, source=source, windows=windows)
    auxiliary_rows = list(epoch_auxiliary_flush_rows)
    auxiliary_steps = _validate_epoch_auxiliary_flush_rows(
        auxiliary_rows, source=source, windows=windows, anchors=anchors, config=config
    )
    summary_keys = {
        "spec_version", "arm", "completed_source_blocks", "completed_epochs",
        "source_plan_sha256", "epoch_plan_sha256s", "source_groups", "candidate_rollouts",
        "optimization_windows", "completed_policy_steps", "anchor_groups",
        "auxiliary_flush_steps", "epoch_auxiliary_flushes", "total_optimizer_steps",
        "rl_groups", "complete", "last_checkpoint",
    }
    _require_keys(summary, summary_keys, name="run_summary.json")
    expected = {
        "spec_version": contract.SPEC_VERSION,
        "arm": config["experiments"]["active"]["arm"],
        "completed_source_blocks": contract.TOTAL_SOURCE_BLOCKS,
        "completed_epochs": contract.SOURCE_EPOCHS,
        "source_plan_sha256": source.source_plan_sha256,
        "epoch_plan_sha256s": list(source.epoch_plan_sha256s),
        "source_groups": contract.TOTAL_SOURCE_GROUPS,
        "candidate_rollouts": contract.TOTAL_CANDIDATES,
        "optimization_windows": windows.optimization_windows,
        "completed_policy_steps": windows.policy_optimizer_steps,
        "anchor_groups": anchors.anchor_groups,
        "auxiliary_flush_steps": auxiliary_steps,
        "epoch_auxiliary_flushes": auxiliary_rows,
        "total_optimizer_steps": windows.policy_optimizer_steps + auxiliary_steps,
        "rl_groups": rl_groups,
        "complete": True,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"run_summary.{key} differs from recomputed logs.")
    if not isinstance(summary["last_checkpoint"], str) or not summary["last_checkpoint"]:
        raise ValueError("A complete run must identify its final checkpoint.")


_RUN_TOP_LEVEL_ALLOWLIST = {
    "resolved_config.json",
    "runtime_signature.json",
    "source_epoch_plan.json",
    "source_groups.jsonl",
    "groups.jsonl",
    "windows.jsonl",
    "anchor_groups.jsonl",
    "epoch_auxiliary_flushes.jsonl",
    "run_summary.json",
    "recovery",
    *(f"epoch-{epoch:03d}-source-{percent:03d}-adapter" for epoch in range(contract.SOURCE_EPOCHS) for percent in (25, 50, 75, 100)),
}


def validate_run_directory_contents(run_dir: str | Path) -> None:
    """Reject monitoring files and every other uncontracted top-level artifact."""

    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in _RUN_TOP_LEVEL_ALLOWLIST)
    if unexpected:
        raise ValueError(f"Run directory contains uncontracted artifacts: {unexpected}")


def _recursive_difference_paths(
    left: Any, right: Any, prefix: tuple[str, ...] = ()
) -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: set[str] = set()
        for key in sorted(set(left) | set(right), key=str):
            path = (*prefix, str(key))
            if key not in left or key not in right:
                result.add(".".join(path))
            else:
                result.update(_recursive_difference_paths(left[key], right[key], path))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        result = set()
        for index in range(max(len(left), len(right))):
            path = (*prefix, str(index))
            if index >= len(left) or index >= len(right):
                result.add(".".join(path))
            else:
                result.update(_recursive_difference_paths(left[index], right[index], path))
        return result
    return set() if left == right else {".".join(prefix)}


def validate_experiment_matrix(
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Permit only the confirmed A/B/C/D reward and task-factor differences."""

    if set(configs) != set(contract.EXPERIMENT_ARMS):
        raise ValueError("Experiment matrix must contain exactly arms A, B, C, and D.")
    for arm in contract.EXPERIMENT_ARMS:
        active = configs[arm].get("experiments", {}).get("active", {})
        if active.get("arm") != arm:
            raise ValueError(f"Experiment matrix key {arm} contradicts active.arm.")
    allowed = {
        "A:B": {
            "experiments.active.arm",
            "experiments.active.reward_profile",
            "reward.same_a",
            "reward.same_ab",
        },
        "B:C": {
            "experiments.active.arm",
            "experiments.active.text_to_sid_enabled",
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
    }
    isolation_paths = {"output.log_dir", "output.run_dir"}
    result: dict[str, set[str]] = {}
    for pair, expected in allowed.items():
        left, right = pair.split(":")
        actual = _recursive_difference_paths(configs[left], configs[right])
        missing_isolation = isolation_paths - actual
        if missing_isolation:
            raise ValueError(
                "Experiment arms must use different run/log paths: "
                f"{sorted(missing_isolation)[0]}"
            )
        actual -= isolation_paths
        if actual != expected:
            unexpected = sorted(actual - expected)
            missing = sorted(expected - actual)
            detail = unexpected[0] if unexpected else missing[0]
            raise ValueError(
                f"Experiment pair {pair} has an unconfirmed or missing difference: {detail}."
            )
        result[pair] = actual
    return result


def validate_probe64_report(
    report: Mapping[str, Any],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    trie: Any,
    require_complete: bool,
) -> int:
    """Bind every reported beam candidate to the fixed probe and SID trie."""

    if int(report.get("schema_version", -1)) != 1:
        raise ValueError("Probe report must use schema_version=1.")
    if report.get("spec_version") != contract.SPEC_VERSION:
        raise ValueError("Probe report belongs to another spec.")
    if int(report.get("num_candidates", -1)) != contract.EVALUATION_CANDIDATES:
        raise ValueError("Probe report must declare num_candidates=64.")
    if report.get("require_unique_candidates") is not True:
        raise ValueError("Probe report must require unique candidates.")
    if report.get("cache_implementation") != "offloaded":
        raise ValueError("Probe report must use the exact offloaded KV cache.")

    rows = tuple(expected_rows)
    selected = report.get("selected_indices")
    results = report.get("results")
    if not isinstance(selected, list) or not isinstance(results, list):
        raise ValueError("Probe report must contain selected_indices and results lists.")
    try:
        indices = tuple(int(value) for value in selected)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probe selected_indices must be integers.") from exc
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("Probe selected_indices must be non-empty, unique, and sorted.")
    if indices[0] < 0 or indices[-1] >= len(rows):
        raise ValueError("Probe selected_indices exceed the fixed probe.")
    if len(results) != len(indices) or int(report.get("result_count", -1)) != len(results):
        raise ValueError("Probe result_count differs from selected_indices.")
    complete = indices == tuple(range(len(rows)))
    if report.get("complete") is not complete:
        raise ValueError("Probe complete flag differs from selected_indices.")
    if require_complete and not complete:
        raise ValueError("A complete Probe64 report must cover every fixed row.")

    result_fields = {
        "probe_index",
        "task",
        "source_row_sha256",
        "source_line",
        "target_sid",
        "prompt_tokens",
        "candidate_sids",
        "hit_rank",
        "pass_at_64",
    }
    pass_count = 0
    for position, (probe_index, result) in enumerate(
        zip(indices, results, strict=True)
    ):
        expected = rows[probe_index]
        if not isinstance(result, Mapping) or set(result) != result_fields:
            raise ValueError(f"Probe result {position} has an invalid schema.")
        expected_identity = {
            "probe_index": probe_index,
            "task": str(expected["task"]),
            "source_row_sha256": str(expected["source_row_sha256"]),
            "source_line": int(expected["source_line"]),
            "target_sid": str(expected["target_sid"]),
        }
        observed_identity = {
            key: int(result[key]) if key in {"probe_index", "source_line"} else str(result[key])
            for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise ValueError(f"Probe result {position} differs from its fixed source row.")
        if int(result["prompt_tokens"]) <= 0:
            raise ValueError(f"Probe result {position} has no prompt tokens.")
        candidates = result["candidate_sids"]
        if not isinstance(candidates, list) or len(candidates) != 64:
            raise ValueError(f"Probe result {position} must contain 64 unique legal SIDs.")
        parsed: list[Sid] = []
        try:
            for value in candidates:
                sid = Sid.parse(str(value))
                if sid.render() != value or not bool(trie.contains(sid)):
                    raise ValueError(value)
                parsed.append(sid)
            target = Sid.parse(str(result["target_sid"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Probe result {position} must contain 64 unique trie SIDs."
            ) from exc
        rendered = [sid.render() for sid in parsed]
        if len(set(rendered)) != 64:
            raise ValueError(f"Probe result {position} must contain 64 unique legal SIDs.")
        hit_rank = next(
            (rank for rank, candidate in enumerate(parsed, start=1) if candidate == target),
            None,
        )
        if result["hit_rank"] != hit_rank:
            raise ValueError(f"Probe result {position} reports the wrong hit_rank.")
        passed = hit_rank is not None
        if result["pass_at_64"] is not passed:
            raise ValueError(f"Probe result {position} reports the wrong pass_at_64 value.")
        pass_count += int(passed)
    if int(report.get("pass_count", -1)) != pass_count:
        raise ValueError("Probe pass_count differs from the recomputed result.")
    return len(results)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank.")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path.name}:{line_number} is not a JSON object.")
            rows.append(value)
    return rows


def _read_json_object(path: Path, *, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON.") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain one JSON object.")
    return value


def _frozen_source_group_ids(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read the immutable normalized inputs so source indices are not self-attested."""

    from .data import (
        load_recommendation_target_groups,
        load_text_to_sid_target_groups,
    )

    recommendation = load_recommendation_target_groups(
        Path(config["output"]["recommendation_groups_dir"]) / "groups.jsonl"
    )
    text_to_sid = load_text_to_sid_target_groups(
        Path(config["output"]["text_to_sid_groups_dir"]) / "groups.jsonl"
    )
    return (
        tuple(group.group_id for group in recommendation),
        tuple(group.group_id for group in text_to_sid),
    )


def verify_run_directory(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    recommendation_group_ids: Sequence[str] | None = None,
    text_to_sid_group_ids: Sequence[str] | None = None,
) -> RunVerificationResult:
    """Load and cross-check one complete V1.1 two-epoch run directory."""

    root = Path(run_dir)
    validate_run_directory_contents(root)
    resolved = _read_json_object(root / "resolved_config.json", name="resolved_config.json")
    if dict(resolved) != dict(config):
        raise ValueError("resolved_config.json differs from the requested verification config.")
    plan = _read_json_object(root / "source_epoch_plan.json", name="source_epoch_plan.json")
    if (recommendation_group_ids is None) != (text_to_sid_group_ids is None):
        raise ValueError("Supply both normalized source ID sequences or neither.")
    if recommendation_group_ids is None:
        recommendation_group_ids, text_to_sid_group_ids = _frozen_source_group_ids(config)
    source = validate_source_group_rows(
        _read_jsonl(root / "source_groups.jsonl"),
        config,
        plan_manifest=plan,
        recommendation_group_ids=recommendation_group_ids,
        text_to_sid_group_ids=text_to_sid_group_ids,
    )
    windows = validate_window_rows(_read_jsonl(root / "windows.jsonl"), config)
    group_rows = _read_jsonl(root / "groups.jsonl")
    anchor_group_rows = _read_jsonl(root / "anchor_groups.jsonl")
    flush_path = root / "epoch_auxiliary_flushes.jsonl"
    epoch_auxiliary_flush_rows = (
        _read_jsonl(flush_path) if flush_path.exists() else []
    )
    summary = _read_json_object(root / "run_summary.json", name="run_summary.json")
    validate_run_summary(
        summary,
        source=source,
        windows=windows,
        group_rows=group_rows,
        anchor_group_rows=anchor_group_rows,
        epoch_auxiliary_flush_rows=epoch_auxiliary_flush_rows,
        config=config,
    )
    return RunVerificationResult(
        source=source,
        windows=windows,
        anchor_groups=len(anchor_group_rows),
    )


__all__ = [
    "SourceVerificationResult",
    "RunVerificationResult",
    "WindowVerificationResult",
    "AnchorVerificationResult",
    "validate_experiment_matrix",
    "validate_anchor_group_rows",
    "validate_probe64_report",
    "validate_run_summary",
    "validate_run_directory_contents",
    "validate_source_group_rows",
    "validate_window_rows",
    "verify_run_directory",
]
