"""Formal DAPO-Anchor-Multitask V1.1 exact two-epoch orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import contract
from ._infra.dapo_fingerprint import validate_runtime_signature
from ._infra.dapo_scoring import dense_scoring_mode
from ._infra.grpo_constraint import RecommendationGrammar
from ._infra.grpo_trie import SidPrefixTrie
from ._infra.rloo_checkpoint import save_policy_atomic
from ._infra.rloo_modeling import PolicyModel, load_policy_model
from .checkpoint import (
    PendingBuffer,
    RecoveryState,
    load_latest_recovery,
    restore_training_state,
    save_recovery_checkpoint,
)
from .collection import (
    PendingOptimizationBuffer,
    SourceGroupAudit,
    sample_source_block,
)
from .data import (
    SidTask,
    load_recommendation_target_groups,
    load_text_to_sid_target_groups,
)
from .engine import PolicyStepLRScheduler
from .source_blocks import (
    EpochSourcePlan,
    SOURCE_BLOCKS,
    build_epoch_source_plans,
    flatten_epoch_source_blocks,
    source_plan_manifest,
)
from .training import (
    build_optimizer,
    flush_pending_anchor,
    optimize_pending_buffer,
    trainable_parameters,
)
from .verify import validate_run_directory_contents


class RecoveryAuditInjectedFailure(RuntimeError):
    """Intentional pre-checkpoint interruption, available only to offline audits."""


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_deterministic_runtime(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_global_seed(seed)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _truncate_jsonl(path: Path, rows: int) -> None:
    if rows < 0:
        raise ValueError("JSONL row count must be non-negative.")
    if not path.exists():
        if rows:
            raise RuntimeError(f"Missing JSONL with {rows} committed rows: {path}")
        # Every contracted JSONL artifact exists even when its committed row
        # count is zero, so recovery can compare the two trajectories exactly.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        return
    with path.open("rb") as handle:
        kept = [handle.readline() for _ in range(rows)]
        if any(not line for line in kept):
            raise RuntimeError(f"JSONL has fewer than {rows} committed rows: {path}")
    temporary = path.with_name(f".{path.name}.truncate")
    with temporary.open("wb") as handle:
        handle.writelines(kept)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object.")
    return value


def _prepare_run_directory(
    output: Path,
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
    plan_manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if output.exists() and not output.is_dir():
        raise FileExistsError(output)
    if output.exists() and next(output.iterdir(), None) is not None:
        validate_run_directory_contents(output)
        if not resume:
            raise FileExistsError(f"Non-empty output directory: {output}")
        if _read_json(output / "resolved_config.json", "resolved config") != config:
            raise RuntimeError("Existing run config differs from this invocation.")
        if _read_json(output / "runtime_signature.json", "runtime signature") != signature:
            raise RuntimeError("Existing run signature differs from this invocation.")
        if _read_json(output / "source_epoch_plan.json", "source epoch plan") != plan_manifest:
            raise RuntimeError("Existing source epoch plan differs from this invocation.")
        return
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "resolved_config.json", dict(config))
    _atomic_json(output / "runtime_signature.json", dict(signature))
    _atomic_json(output / "source_epoch_plan.json", dict(plan_manifest))


def _validate_output_directory(
    config: Mapping[str, Any],
    output: Path,
    *,
    audit_mode: bool,
) -> None:
    """Keep formal output immutable while allowing a separate recovery-audit root."""

    formal_output = Path(config["output"]["run_dir"]).resolve()
    if output == formal_output:
        if audit_mode:
            raise ValueError("Recovery audit must not write into the formal run directory.")
        return
    if not audit_mode:
        raise ValueError("Formal training output_dir differs from the frozen config.")
    audit_root = contract.RECOVERY_AUDIT_ROOT.resolve()
    if not output.is_relative_to(audit_root):
        raise ValueError("Recovery audit output is outside the approved audit root.")


def _load_inputs(config: Mapping[str, Any]):
    recommendation = load_recommendation_target_groups(
        Path(config["output"]["recommendation_groups_dir"]) / "groups.jsonl"
    )
    text = load_text_to_sid_target_groups(
        Path(config["output"]["text_to_sid_groups_dir"]) / "groups.jsonl"
    )
    if len(recommendation) != int(config["data"]["recommendation_groups"]):
        raise RuntimeError("Recommendation group count differs from config.")
    if len(text) != int(config["data"]["text_to_sid_groups"]):
        raise RuntimeError("Text-to-SID group count differs from config.")
    trie = SidPrefixTrie.load(
        Path(config["output"]["trie_dir"]),
        expected_leaf_count=int(config["trie"]["unique_sids"]),
    )
    plans = build_epoch_source_plans(
        [group.group_id for group in recommendation],
        [group.group_id for group in text],
        seed=int(config["train"]["seed"]),
        source_epochs=int(config["train"]["source_epochs"]),
        blocks=int(config["data"]["source_blocks"]),
    )
    return recommendation, text, trie, plans, flatten_epoch_source_blocks(plans)


def _orders_for_block(
    plans: Sequence[EpochSourcePlan], block_index: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0 <= block_index < contract.TOTAL_SOURCE_BLOCKS:
        raise ValueError("Global source block is outside the two-epoch contract.")
    epoch_index = block_index // SOURCE_BLOCKS
    plan = plans[epoch_index]
    if plan.epoch_index != epoch_index:
        raise RuntimeError("Epoch plan index differs from the global source cursor.")
    return plan.recommendation_order, plan.text_to_sid_order


def _grammars(bundle: PolicyModel, trie: SidPrefixTrie):
    return {
        SidTask.RECOMMENDATION: RecommendationGrammar(
            bundle.tokenizer, trie, mode="train_recommendation"
        ),
        SidTask.ITEM_TEXT_TO_SID: RecommendationGrammar(
            bundle.tokenizer, trie, mode="train_text_to_sid"
        ),
    }


def _optimizer_step_values(optimizer: torch.optim.Optimizer) -> set[int]:
    result: set[int] = set()
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        result.add(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
    return result


def _assert_optimizer_state_step(
    optimizer: torch.optim.Optimizer, expected_step: int
) -> None:
    observed = _optimizer_step_values(optimizer)
    if expected_step == 0 and not observed:
        return
    if observed != {int(expected_step)}:
        raise RuntimeError(
            "AdamW state step differs from committed optimizer count: "
            f"expected={expected_step}, observed={sorted(observed)}."
        )


def _source_log_rows(audits: Sequence[SourceGroupAudit]) -> list[dict[str, Any]]:
    return [
        {
            "block_index": item.block_index,
            "epoch_index": item.epoch_index,
            "epoch_block_index": item.epoch_block_index,
            "task": item.task.value,
            "source_index": item.source_index,
            "group_id": item.group_id,
            "rollout_count": item.rollout_count,
            "route": item.route.value,
            "policy_step_at_sample": item.policy_step_at_sample,
        }
        for item in audits
    ]


def _policy_group_rows(pending: PendingOptimizationBuffer, window_index: int):
    rows = []
    for item in pending.rl_groups:
        rows.append(
            {
                "optimization_window_index": int(window_index),
                "source_block": item.source_block,
                "epoch_index": item.epoch_index,
                "epoch_block_index": item.epoch_block_index,
                "source_index": item.source_index,
                "task": item.group.task.value,
                "group_id": item.group.group_id,
                "route": item.route.value,
                "rewards": [float(value) for value in item.reward.rewards.tolist()],
                "advantages": [
                    float(value) for value in item.reward.advantages.tolist()
                ],
                "tiers": list(item.reward.tiers),
                "candidate_sids": [
                    candidate.sid.render() for candidate in item.candidates
                ],
                "gt_injection_count": 0,
                "k": 1,
            }
        )
    return rows


def _window_row(result, pending: PendingOptimizationBuffer) -> dict[str, Any]:
    execution = result.execution
    anchor = execution.anchor_output
    anchor_groups = [] if anchor is None else [asdict(item) for item in anchor.group_results]
    return {
        "optimization_window_index": result.optimization_window_index,
        "first_source_block": result.first_source_block,
        "next_source_block": result.next_source_block,
        "snapshot_policy_step": pending.snapshot_policy_step,
        "completed_policy_steps": (
            pending.snapshot_policy_step + result.policy_optimizer_steps
        ),
        "rl_groups": len(pending.rl_groups),
        "anchor_groups": len(pending.anchor_groups),
        "policy_optimizer_steps": result.policy_optimizer_steps,
        "anchor_optimizer_steps": 0,
        "policy_gradient_norms": list(execution.policy_gradient_norms),
        "rl_reference_gradient_norm": execution.rl_reference_gradient_norm,
        "lambda_cap": execution.gradient_merge.lambda_cap,
        "lambda_effective": execution.gradient_merge.lambda_effective,
        "anchor_raw_gradient_norm": (
            execution.gradient_merge.anchor_raw_gradient_norm
        ),
        "anchor_scaled_gradient_norm": (
            execution.gradient_merge.anchor_scaled_gradient_norm
        ),
        "anchor_to_rl_gradient_ratio": (
            execution.gradient_merge.anchor_to_rl_gradient_ratio
        ),
        "combined_gradient_norm": execution.gradient_merge.combined_gradient_norm,
        "steps": [asdict(item) for item in execution.policy_outputs],
        "anchor_group_results": anchor_groups,
        "anchor_max_replay_logp_difference": (
            0.0 if anchor is None else anchor.max_replay_logp_difference
        ),
        "k": 1,
    }


def _anchor_group_rows(
    anchor_output: Any,
    *,
    phase: str,
    unit_index: int,
) -> list[dict[str, Any]]:
    """Persist one GT-set likelihood record for every covered source group."""

    if anchor_output is None:
        return []
    return [
        {
            **asdict(item),
            "phase": str(phase),
            "unit_index": int(unit_index),
        }
        for item in anchor_output.group_results
    ]


def _epoch_auxiliary_row(
    execution,
    pending: PendingOptimizationBuffer,
    *,
    epoch_index: int,
    rl_reference_gradient_norm: float,
) -> dict[str, Any]:
    """Persist one epoch-end Anchor-only exception in the recovery transaction."""

    if pending.first_source_block is None or not pending.anchor_groups:
        raise ValueError("A final auxiliary audit requires pending Anchor groups.")
    anchor = execution.anchor_output
    if anchor is None or len(anchor.group_results) != len(pending.anchor_groups):
        raise RuntimeError("Final auxiliary output differs from the pending Anchor groups.")
    merge = execution.gradient_merge
    gt_sid_rows = sum(int(item.gt_sid_count) for item in anchor.group_results)
    decision_tokens = sum(int(item.decision_tokens) for item in anchor.group_results)
    if gt_sid_rows <= 0 or decision_tokens <= 0:
        raise RuntimeError("Final auxiliary audit contains no GT decision rows.")
    return {
        "epoch_index": int(epoch_index),
        "first_source_block": int(pending.first_source_block),
        "next_source_block": int(pending.next_source_block),
        "snapshot_policy_step": int(pending.snapshot_policy_step),
        "anchor_groups": len(pending.anchor_groups),
        "gt_sid_rows": gt_sid_rows,
        "decision_tokens": decision_tokens,
        "optimizer_steps": int(execution.optimizer_steps),
        "scheduler_advanced": False,
        "rl_reference_gradient_norm": float(rl_reference_gradient_norm),
        "lambda_cap": float(merge.lambda_cap),
        "lambda_effective": float(merge.lambda_effective),
        "anchor_raw_gradient_norm": float(merge.anchor_raw_gradient_norm),
        "anchor_scaled_gradient_norm": float(merge.anchor_scaled_gradient_norm),
        "anchor_to_rl_gradient_ratio": float(merge.anchor_to_rl_gradient_ratio),
        "combined_gradient_norm": float(merge.combined_gradient_norm),
        "anchor_set_nll_mean": sum(
            float(item.set_nll) for item in anchor.group_results
        )
        / float(len(anchor.group_results)),
        "anchor_max_replay_logp_difference": float(
            anchor.max_replay_logp_difference
        ),
    }


def _parameter_digest(bundle: PolicyModel) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in trainable_parameters(bundle):
            digest.update(
                parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


def _checkpoint_state(
    *,
    pending: PendingOptimizationBuffer,
    source_plan: Mapping[str, Any],
    optimization_window_index: int,
    completed_policy_steps: int,
    auxiliary_flush_steps: int,
    last_rl_reference_grad_norm: float | None,
    source_log_rows: int,
    group_log_rows: int,
    window_log_rows: int,
    anchor_log_rows: int,
    epoch_auxiliary_flushes: Sequence[Mapping[str, Any]],
) -> RecoveryState:
    payload = None
    if pending.rl_groups or pending.anchor_groups:
        payload = PendingBuffer(
            snapshot_policy_step=pending.snapshot_policy_step,
            payload=pending,
        )
    next_source_block = int(pending.next_source_block)
    epoch_index, next_block_in_epoch = divmod(next_source_block, SOURCE_BLOCKS)
    epoch_hashes = tuple(
        str(value["sha256"]) for value in source_plan["epochs"]
    )
    return RecoveryState(
        next_source_block=next_source_block,
        epoch_index=epoch_index,
        next_block_in_epoch=next_block_in_epoch,
        source_plan_sha256=str(source_plan["sha256"]),
        epoch_plan_sha256s=epoch_hashes,
        optimization_window_index=int(optimization_window_index),
        completed_policy_steps=int(completed_policy_steps),
        total_optimizer_steps=int(completed_policy_steps + auxiliary_flush_steps),
        auxiliary_flush_steps=int(auxiliary_flush_steps),
        pending_buffer=payload,
        last_rl_reference_grad_norm=last_rl_reference_grad_norm,
        source_log_rows=int(source_log_rows),
        group_log_rows=int(group_log_rows),
        window_log_rows=int(window_log_rows),
        anchor_log_rows=int(anchor_log_rows),
        epoch_auxiliary_flushes=tuple(dict(value) for value in epoch_auxiliary_flushes),
    )


def _prune_recovery(root: Path, retain: int) -> None:
    checkpoints = sorted(
        path for path in root.glob("checkpoint-block-*") if path.is_dir()
    )
    for path in checkpoints[:-int(retain)]:
        shutil.rmtree(path)


def _periodic_checkpoint_due(
    next_source_block: int,
    *,
    requested_stop: int,
    total_blocks: int,
    interval: int,
) -> bool:
    """Checkpoint partial runs, but leave source exhaustion to finalization."""

    if not 0 <= next_source_block <= requested_stop <= total_blocks:
        raise ValueError("Checkpoint source cursors are inconsistent.")
    if interval <= 0:
        raise ValueError("Checkpoint interval must be positive.")
    if next_source_block == total_blocks:
        return False
    return next_source_block % interval == 0 or next_source_block == requested_stop


def _save_milestones(
    bundle: PolicyModel, output: Path, completed_source_blocks: int
) -> None:
    milestones = {133: 25, 266: 50, 399: 75, 532: 100}
    for epoch_index in range(contract.SOURCE_EPOCHS):
        epoch_completed = min(
            max(int(completed_source_blocks) - epoch_index * SOURCE_BLOCKS, 0),
            SOURCE_BLOCKS,
        )
        for block, percentage in milestones.items():
            target = output / f"epoch-{epoch_index:03d}-source-{percentage:03d}-adapter"
            if epoch_completed >= block and not target.exists():
                save_policy_atomic(bundle, target)


def _run_training_impl(
    config: dict[str, Any],
    runtime_signature: dict[str, Any],
    *,
    output_dir: Path,
    device: str | torch.device,
    resume: bool,
    stop_after_source_blocks: int | None,
    aligned_cache: bool,
    audit_mode: bool,
    inject_fault_after_source_blocks: int | None,
) -> dict[str, Any]:
    validate_runtime_signature(runtime_signature)
    output = Path(output_dir).resolve()
    _validate_output_directory(config, output, audit_mode=audit_mode)
    if inject_fault_after_source_blocks is not None:
        if not audit_mode:
            raise ValueError("Injected recovery faults are available only to offline audits.")
        if int(inject_fault_after_source_blocks) <= 0:
            raise ValueError("Injected recovery fault cursor must be positive.")
    recommendation, text, trie, plans, blocks = _load_inputs(config)
    plan = source_plan_manifest(
        plans,
        [group.group_id for group in recommendation],
        [group.group_id for group in text],
        seed=int(config["train"]["seed"]),
    )
    _prepare_run_directory(
        output, config, runtime_signature, plan, resume=resume
    )
    configure_deterministic_runtime(int(config["train"]["seed"]))
    recovery_root = output / "recovery"
    recovery = load_latest_recovery(recovery_root, runtime_signature) if resume else None
    if recovery is None:
        state = RecoveryState(
            next_source_block=0,
            epoch_index=0,
            next_block_in_epoch=0,
            source_plan_sha256=str(plan["sha256"]),
            epoch_plan_sha256s=tuple(str(item["sha256"]) for item in plan["epochs"]),
            optimization_window_index=0,
            completed_policy_steps=0,
            total_optimizer_steps=0,
            auxiliary_flush_steps=0,
            pending_buffer=None,
            last_rl_reference_grad_norm=None,
            source_log_rows=0,
            group_log_rows=0,
            window_log_rows=0,
            anchor_log_rows=0,
            epoch_auxiliary_flushes=(),
        )
        policy_path = None
        training_state = None
    else:
        policy_path, state, training_state = recovery
        expected_epoch_hashes = tuple(str(item["sha256"]) for item in plan["epochs"])
        if (
            state.source_plan_sha256 != str(plan["sha256"])
            or state.epoch_plan_sha256s != expected_epoch_hashes
        ):
            raise RuntimeError("Recovery source plan differs from this invocation.")
    source_log = output / "source_groups.jsonl"
    group_log = output / "groups.jsonl"
    window_log = output / "windows.jsonl"
    anchor_log = output / "anchor_groups.jsonl"
    epoch_flush_log = output / "epoch_auxiliary_flushes.jsonl"
    _truncate_jsonl(source_log, state.source_log_rows)
    _truncate_jsonl(group_log, state.group_log_rows)
    _truncate_jsonl(window_log, state.window_log_rows)
    _truncate_jsonl(anchor_log, state.anchor_log_rows)
    _truncate_jsonl(epoch_flush_log, len(state.epoch_auxiliary_flushes))
    bundle = load_policy_model(config, device=device, policy_adapter_path=policy_path)
    grammars = _grammars(bundle, trie)
    optimizer = build_optimizer(bundle, config)
    scheduler = PolicyStepLRScheduler(optimizer, config)
    if training_state is not None:
        restore_training_state(optimizer, scheduler, training_state)
    if scheduler.completed_policy_steps != state.completed_policy_steps:
        raise RuntimeError("Recovery scheduler and policy-step cursor differ.")
    _assert_optimizer_state_step(optimizer, state.total_optimizer_steps)
    if state.pending_buffer is None:
        pending = PendingOptimizationBuffer.empty(
            state.completed_policy_steps, state.next_source_block
        )
    else:
        pending = state.pending_buffer.payload
        if not isinstance(pending, PendingOptimizationBuffer):
            raise TypeError("Recovery pending payload has the wrong type.")
        if pending.snapshot_policy_step != state.completed_policy_steps:
            raise RuntimeError("Recovery pending policy snapshot is stale.")
        if pending.next_source_block != state.next_source_block:
            raise RuntimeError("Recovery pending source cursor differs.")
    total_blocks = len(blocks)
    if total_blocks != contract.TOTAL_SOURCE_BLOCKS:
        raise RuntimeError("Two-epoch source block count differs from the contract.")
    requested_stop = total_blocks
    if stop_after_source_blocks is not None:
        if int(stop_after_source_blocks) < 0:
            raise ValueError("stop_after_source_blocks must be non-negative.")
        requested_stop = min(
            total_blocks, state.next_source_block + int(stop_after_source_blocks)
        )
    optimization_window_index = state.optimization_window_index
    auxiliary_flush_steps = state.auxiliary_flush_steps
    epoch_auxiliary_flushes = [dict(value) for value in state.epoch_auxiliary_flushes]
    last_rl_reference = state.last_rl_reference_grad_norm
    source_rows = state.source_log_rows
    group_rows = state.group_log_rows
    window_rows = state.window_log_rows
    anchor_rows = state.anchor_log_rows
    last_checkpoint: Path | None = recovery[0] if recovery is not None else None
    minimum = int(config["sampling"]["minimum_effective_groups_per_update"])

    def commit_policy_window() -> None:
        nonlocal anchor_rows, group_rows, last_rl_reference
        nonlocal optimization_window_index, pending, window_rows
        if not pending.rl_groups:
            raise RuntimeError("Policy window commit requires at least one RL group.")
        result = optimize_pending_buffer(
            bundle,
            optimizer,
            scheduler,
            pending,
            grammars,
            optimization_window_index=optimization_window_index,
            config=config,
            device=device,
        )
        anchor_output = result.execution.anchor_output
        if anchor_output is None or len(anchor_output.group_results) != len(pending.anchor_groups):
            raise RuntimeError("Every pending Anchor group must contribute once.")
        _append_jsonl(group_log, _policy_group_rows(pending, optimization_window_index))
        _append_jsonl(window_log, [_window_row(result, pending)])
        _append_jsonl(
            anchor_log,
            _anchor_group_rows(
                anchor_output, phase="merged_policy_window", unit_index=optimization_window_index
            ),
        )
        group_rows += len(pending.rl_groups)
        anchor_rows += len(anchor_output.group_results)
        window_rows += 1
        last_rl_reference = result.execution.rl_reference_gradient_norm
        optimization_window_index += 1
        pending = PendingOptimizationBuffer.empty(
            scheduler.completed_policy_steps, pending.next_source_block
        )
        _assert_optimizer_state_step(
            optimizer, scheduler.completed_policy_steps + auxiliary_flush_steps
        )
        _save_milestones(bundle, output, pending.next_source_block)

    def flush_epoch_anchor(epoch_index: int) -> None:
        nonlocal anchor_rows, auxiliary_flush_steps, epoch_auxiliary_flushes, pending
        if not pending.anchor_groups:
            return
        if last_rl_reference is None:
            raise RuntimeError(
                "Epoch-end Anchor flush requires a preceding real RL gradient reference."
            )
        reference = float(last_rl_reference)
        if not math.isfinite(reference) or reference <= 0.0:
            raise RuntimeError("Epoch-end Anchor flush lacks a positive RL gradient reference.")
        execution = flush_pending_anchor(
            bundle,
            optimizer,
            pending,
            grammars,
            last_rl_reference_gradient_norm=reference,
            config=config,
            device=device,
        )
        if execution.anchor_output is None:
            raise RuntimeError("Epoch-end Anchor flush produced no Anchor result.")
        if len(execution.anchor_output.group_results) != len(pending.anchor_groups):
            raise RuntimeError("Epoch-end Anchor coverage differs from pending groups.")
        row = _epoch_auxiliary_row(
            execution,
            pending,
            epoch_index=epoch_index,
            rl_reference_gradient_norm=reference,
        )
        if int(row["optimizer_steps"]) != 1:
            raise RuntimeError("Epoch-end Anchor coverage requires one bounded optimizer step.")
        _append_jsonl(
            anchor_log,
            _anchor_group_rows(
                execution.anchor_output,
                phase="epoch_end_anchor_flush",
                unit_index=epoch_index,
            ),
        )
        _append_jsonl(epoch_flush_log, [row])
        anchor_rows += len(execution.anchor_output.group_results)
        auxiliary_flush_steps += int(execution.optimizer_steps)
        epoch_auxiliary_flushes.append(row)
        pending = PendingOptimizationBuffer.empty(
            scheduler.completed_policy_steps, pending.next_source_block
        )
        _assert_optimizer_state_step(
            optimizer, scheduler.completed_policy_steps + auxiliary_flush_steps
        )

    while pending.next_source_block < requested_stop:
        block = blocks[pending.next_source_block]
        recommendation_order, text_to_sid_order = _orders_for_block(
            plans, block.index
        )
        sampled = sample_source_block(
            bundle,
            recommendation,
            text,
            block,
            grammars,
            recommendation_order=recommendation_order,
            text_to_sid_order=text_to_sid_order,
            policy_step=scheduler.completed_policy_steps,
            config=config,
            device=device,
            aligned_cache=aligned_cache,
        )
        pending = pending.append(sampled)
        source_batch_rows = _source_log_rows(sampled.source_audits)
        _append_jsonl(source_log, source_batch_rows)
        source_rows += len(source_batch_rows)
        if len(pending.rl_groups) >= minimum:
            commit_policy_window()
        epoch_boundary = block.epoch_block_index + 1 == SOURCE_BLOCKS
        if epoch_boundary:
            if pending.rl_groups:
                commit_policy_window()
            elif pending.anchor_groups:
                flush_epoch_anchor(block.epoch_index)
            if pending.rl_groups or pending.anchor_groups:
                raise RuntimeError("Epoch boundary left an uncommitted source group.")
        if (
            inject_fault_after_source_blocks is not None
            and pending.next_source_block == int(inject_fault_after_source_blocks)
        ):
            raise RecoveryAuditInjectedFailure(
                "Injected recovery-audit interruption after durable JSONL writes "
                "and before the next atomic recovery checkpoint."
            )
        checkpoint_due = epoch_boundary or _periodic_checkpoint_due(
            pending.next_source_block,
            requested_stop=requested_stop,
            total_blocks=total_blocks,
            interval=int(config["train"]["checkpoint_source_blocks"]),
        )
        if checkpoint_due:
            committed = _checkpoint_state(
                pending=pending,
                source_plan=plan,
                optimization_window_index=optimization_window_index,
                completed_policy_steps=scheduler.completed_policy_steps,
                auxiliary_flush_steps=auxiliary_flush_steps,
                last_rl_reference_grad_norm=last_rl_reference,
                source_log_rows=source_rows,
                group_log_rows=group_rows,
                window_log_rows=window_rows,
                anchor_log_rows=anchor_rows,
                epoch_auxiliary_flushes=epoch_auxiliary_flushes,
            )
            last_checkpoint = save_recovery_checkpoint(
                bundle,
                optimizer,
                scheduler,
                committed,
                recovery_root,
                runtime_signature,
            )
            _prune_recovery(
                recovery_root, int(config["train"]["retain_recovery_checkpoints"])
            )

    complete_source = pending.next_source_block == total_blocks
    if complete_source:
        if pending.rl_groups or pending.anchor_groups:
            raise RuntimeError("Completed training retained an uncommitted source group.")
        _assert_optimizer_state_step(
            optimizer, scheduler.completed_policy_steps + auxiliary_flush_steps
        )
        _save_milestones(bundle, output, total_blocks)
    expected_source_groups = int(config["data"]["recommendation_groups"])
    if bool(config["experiments"]["active"]["text_to_sid_enabled"]):
        expected_source_groups += int(config["data"]["text_to_sid_groups"])
    expected_source_groups *= int(config["train"]["source_epochs"])
    if complete_source and source_rows != expected_source_groups:
        raise RuntimeError(
            f"Source audit rows differ: expected={expected_source_groups}, actual={source_rows}."
        )
    summary = {
        "spec_version": contract.SPEC_VERSION,
        "arm": config["experiments"]["active"]["arm"],
        "completed_source_blocks": pending.next_source_block,
        "completed_epochs": pending.next_source_block // SOURCE_BLOCKS,
        "source_plan_sha256": plan["sha256"],
        "epoch_plan_sha256s": [item["sha256"] for item in plan["epochs"]],
        "source_groups": source_rows,
        "candidate_rollouts": source_rows * contract.GROUP_SIZE,
        "optimization_windows": optimization_window_index,
        "completed_policy_steps": scheduler.completed_policy_steps,
        "anchor_groups": anchor_rows,
        "auxiliary_flush_steps": auxiliary_flush_steps,
        "epoch_auxiliary_flushes": epoch_auxiliary_flushes,
        "total_optimizer_steps": scheduler.completed_policy_steps + auxiliary_flush_steps,
        "rl_groups": group_rows,
        "complete": complete_source,
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
    }
    _atomic_json(output / "run_summary.json", summary)
    return summary


def run_training(
    config: dict[str, Any],
    runtime_signature: dict[str, Any],
    *,
    output_dir: Path,
    device: str | torch.device = "cuda:0",
    resume: bool = True,
    stop_after_source_blocks: int | None = None,
    aligned_cache: bool = True,
    dense_scoring: bool = False,
    formal: bool = False,
    audit_mode: bool = False,
    inject_fault_after_source_blocks: int | None = None,
) -> dict[str, Any]:
    if formal and stop_after_source_blocks is not None:
        raise ValueError("Formal training cannot stop before source exhaustion.")
    if formal and dense_scoring:
        raise ValueError("Formal training cannot enable the dense-scoring test hook.")
    if formal and audit_mode:
        raise ValueError("Formal training cannot enable the recovery-audit output mode.")
    if formal and inject_fault_after_source_blocks is not None:
        raise ValueError("Formal training cannot inject a recovery-audit fault.")
    context = dense_scoring_mode() if dense_scoring else nullcontext()
    with context:
        return _run_training_impl(
            config,
            runtime_signature,
            output_dir=output_dir,
            device=device,
            resume=resume,
            stop_after_source_blocks=stop_after_source_blocks,
            aligned_cache=aligned_cache,
            audit_mode=audit_mode,
            inject_fault_after_source_blocks=inject_fault_after_source_blocks,
        )


def run_one_window_pilot(
    config: dict[str, Any],
    runtime_signature: dict[str, Any],
    *,
    output_dir: Path,
    device: str | torch.device = "cuda:0",
    aligned_cache: bool = True,
) -> dict[str, Any]:
    validate_runtime_signature(runtime_signature)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    configure_deterministic_runtime(int(config["train"]["seed"]))
    recommendation, text, trie, plans, blocks = _load_inputs(config)
    bundle = load_policy_model(config, device=device, policy_adapter_path=None)
    grammars = _grammars(bundle, trie)
    optimizer = build_optimizer(bundle, config)
    scheduler = PolicyStepLRScheduler(optimizer, config)
    before = _parameter_digest(bundle)
    pending = PendingOptimizationBuffer.empty(0, 0)
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    while len(pending.rl_groups) < int(
        config["sampling"]["minimum_effective_groups_per_update"]
    ):
        block = blocks[pending.next_source_block]
        recommendation_order, text_to_sid_order = _orders_for_block(
            plans, block.index
        )
        sampled = sample_source_block(
            bundle,
            recommendation,
            text,
            block,
            grammars,
            recommendation_order=recommendation_order,
            text_to_sid_order=text_to_sid_order,
            policy_step=0,
            config=config,
            device=device,
            aligned_cache=aligned_cache,
        )
        pending = pending.append(sampled)
    result = optimize_pending_buffer(
        bundle,
        optimizer,
        scheduler,
        pending,
        grammars,
        optimization_window_index=0,
        config=config,
        device=device,
    )
    peak = (
        torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3
        if torch.cuda.is_available() and torch.device(device).type == "cuda"
        else 0.0
    )
    report = {
        "source_blocks": pending.next_source_block,
        "source_groups": len(pending.source_audits),
        "rl_groups": len(pending.rl_groups),
        "anchor_groups": len(pending.anchor_groups),
        "policy_optimizer_steps": result.policy_optimizer_steps,
        "anchor_optimizer_steps": 0,
        "parameters_changed": before != _parameter_digest(bundle),
        "peak_reserved_gib": peak,
        "finite": all(
            math.isfinite(value)
            for value in result.execution.policy_gradient_norms
        ),
    }
    if not report["parameters_changed"] or not report["finite"]:
        raise RuntimeError("GPU pilot did not produce a finite parameter update.")
    if peak > float(config["memory"]["max_reserved_gib"]):
        raise RuntimeError("GPU pilot exceeded the 20 GiB reserved-memory gate.")
    _atomic_json(output / "pilot_report.json", report)
    return report


__all__ = [
    "PolicyStepLRScheduler",
    "configure_deterministic_runtime",
    "run_one_window_pilot",
    "run_training",
    "RecoveryAuditInjectedFailure",
]
