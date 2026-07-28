"""Formal DAPO-Anchor-Multitask V1.0 training orchestration."""

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
from statistics import median
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import contract
from ._infra.dapo_fingerprint import (
    calibration_base_signature_sha256,
    validate_runtime_signature,
)
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
from .engine import PolicyStepLRScheduler, gradient_l2_norm, partition_policy_groups
from .source_blocks import build_source_blocks
from .training import (
    backward_policy_minibatch,
    build_anchor_raw_gradient,
    build_optimizer,
    flush_pending_anchor,
    optimize_pending_buffer,
    trainable_parameters,
)
from .verify import validate_run_directory_contents


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
        return
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "resolved_config.json", dict(config))
    _atomic_json(output / "runtime_signature.json", dict(signature))


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
    blocks = build_source_blocks(
        recommendation_groups=len(recommendation),
        text_to_sid_groups=len(text),
        blocks=int(config["data"]["source_blocks"]),
    )
    return recommendation, text, trie, blocks


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


def _final_auxiliary_row(
    execution,
    pending: PendingOptimizationBuffer,
    *,
    rl_reference_gradient_norm: float,
) -> dict[str, Any]:
    """Persist the source-end Anchor-only exception in the recovery transaction."""

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


def _calibration_report(
    config: Mapping[str, Any], signature: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(config["anchor"]["calibration_path"])
    value = _read_json(path, "Anchor calibration")
    required = {
        "schema_version",
        "spec_version",
        "base_signature_sha256",
        "source_blocks",
        "policy_optimizer_steps",
        "model_unchanged",
        "rl_gradient_norms",
        "rl_reference_gradient_norm",
        "anchor_raw_gradient_norm",
        "lambda_calibrated",
    }
    if set(value) != required:
        raise ValueError("Anchor calibration report has an invalid schema.")
    if value["spec_version"] != contract.SPEC_VERSION:
        raise ValueError("Anchor calibration belongs to another spec.")
    if value["base_signature_sha256"] != calibration_base_signature_sha256(dict(signature)):
        raise RuntimeError("Anchor calibration input signature mismatch.")
    if int(value["source_blocks"]) != contract.ANCHOR_CALIBRATION_BLOCKS:
        raise RuntimeError("Anchor calibration did not consume exactly 16 source blocks.")
    if int(value["policy_optimizer_steps"]) != 0 or value["model_unchanged"] is not True:
        raise RuntimeError("Anchor calibration mutated the policy.")
    norms = value["rl_gradient_norms"]
    if not isinstance(norms, list) or not norms or any(
        not math.isfinite(float(norm)) or float(norm) <= 0.0 for norm in norms
    ):
        raise RuntimeError("Anchor calibration has no positive RL gradient reference.")
    reference = float(value["rl_reference_gradient_norm"])
    anchor_norm = float(value["anchor_raw_gradient_norm"])
    if (
        not math.isfinite(reference)
        or reference <= 0.0
        or not math.isclose(reference, float(median(float(norm) for norm in norms)), rel_tol=1.0e-6, abs_tol=1.0e-12)
        or not math.isfinite(anchor_norm)
        or anchor_norm <= 0.0
    ):
        raise RuntimeError("Anchor calibration has an invalid positive gradient reference.")
    calibrated = float(value["lambda_calibrated"])
    if not math.isfinite(calibrated) or not 0.0 < calibrated <= float(
        config["anchor"]["max_weight"]
    ):
        raise ValueError("Anchor calibration lambda is outside the frozen range.")
    return value


def load_calibrated_lambda(
    config: Mapping[str, Any], signature: Mapping[str, Any]
) -> float:
    return float(_calibration_report(config, signature)["lambda_calibrated"])


def _parameter_digest(bundle: PolicyModel) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in trainable_parameters(bundle):
            digest.update(
                parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


def run_calibration(
    config: dict[str, Any],
    base_runtime_signature: dict[str, Any],
    *,
    output_path: Path,
    device: str | torch.device = "cuda:0",
    aligned_cache: bool = True,
) -> dict[str, Any]:
    """Measure fixed first-16-block gradients without any optimizer update."""

    validate_runtime_signature(base_runtime_signature)
    if "anchor_calibration" in base_runtime_signature["inputs"]:
        raise ValueError("Calibration requires a signature without its own output.")
    configure_deterministic_runtime(int(config["train"]["seed"]))
    recommendation, text, trie, blocks = _load_inputs(config)
    bundle = load_policy_model(config, device=device, policy_adapter_path=None)
    grammars = _grammars(bundle, trie)
    optimizer = build_optimizer(bundle, config)
    scheduler = PolicyStepLRScheduler(optimizer, config)
    before = _parameter_digest(bundle)
    pending = PendingOptimizationBuffer.empty(0, 0)
    count = int(config["anchor"]["calibration_source_blocks"])
    minimum = int(config["anchor"]["min_calibration_source_blocks"])
    if count < minimum or count > len(blocks):
        raise ValueError("Anchor calibration source-block count is outside its contract.")
    for block in blocks[:count]:
        sampled = sample_source_block(
            bundle,
            recommendation,
            text,
            block,
            grammars,
            policy_step=0,
            config=config,
            device=device,
            aligned_cache=aligned_cache,
        )
        pending = pending.append(sampled)
    if not pending.rl_groups:
        raise RuntimeError("Anchor calibration found no RL-routed source group.")
    if not pending.anchor_groups:
        raise RuntimeError("Anchor calibration found no Anchor-routed source group.")
    batches = partition_policy_groups(
        pending.rl_groups, optimization_window_index=0, seed=int(config["train"]["seed"])
    )
    rl_norms: list[float] = []
    parameters = trainable_parameters(bundle)
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        backward_policy_minibatch(
            bundle,
            batch,
            grammars,
            config=config,
            device=device,
            enforce_replay_gate=True,
        )
        rl_norms.append(gradient_l2_norm(parameters))
    anchor_norm = 0.0
    if pending.anchor_groups:
        raw = build_anchor_raw_gradient(
            bundle,
            optimizer,
            pending.anchor_groups,
            grammars,
            config=config,
            device=device,
        )
        anchor_norm = raw.raw_gradient_norm
    rl_reference = float(median(rl_norms)) if rl_norms else 0.0
    if (
        not rl_norms
        or any(not math.isfinite(norm) or norm <= 0.0 for norm in rl_norms)
        or not math.isfinite(anchor_norm)
        or anchor_norm <= 0.0
    ):
        raise RuntimeError("Anchor calibration needs positive finite RL and Anchor gradients.")
    target = float(config["anchor"]["target_gradient_ratio"])
    calibrated = min(
        float(config["anchor"]["max_weight"]), target * rl_reference / anchor_norm
    )
    if not math.isfinite(calibrated) or calibrated <= 0.0:
        raise RuntimeError("Anchor calibration produced a non-positive lambda.")
    report = {
        "schema_version": 1,
        "spec_version": contract.SPEC_VERSION,
        "base_signature_sha256": base_runtime_signature["sha256"],
        "source_blocks": count,
        "policy_optimizer_steps": scheduler.completed_policy_steps,
        "model_unchanged": before == _parameter_digest(bundle),
        "rl_gradient_norms": rl_norms,
        "rl_reference_gradient_norm": rl_reference,
        "anchor_raw_gradient_norm": anchor_norm,
        "lambda_calibrated": calibrated,
    }
    if report["policy_optimizer_steps"] != 0 or not report["model_unchanged"]:
        raise RuntimeError("Read-only calibration mutated model state.")
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    _atomic_json(output_path, report)
    return report


def _checkpoint_state(
    *,
    pending: PendingOptimizationBuffer,
    optimization_window_index: int,
    completed_policy_steps: int,
    final_auxiliary_flush_steps: int,
    last_rl_reference_grad_norm: float | None,
    source_log_rows: int,
    group_log_rows: int,
    window_log_rows: int,
    final_auxiliary_flush: Mapping[str, Any] | None = None,
) -> RecoveryState:
    payload = None
    if pending.rl_groups or pending.anchor_groups:
        payload = PendingBuffer(
            snapshot_policy_step=pending.snapshot_policy_step,
            payload=pending,
        )
    return RecoveryState(
        next_source_block=pending.next_source_block,
        optimization_window_index=int(optimization_window_index),
        completed_policy_steps=int(completed_policy_steps),
        total_optimizer_steps=int(completed_policy_steps + final_auxiliary_flush_steps),
        final_auxiliary_flush_steps=int(final_auxiliary_flush_steps),
        pending_buffer=payload,
        last_rl_reference_grad_norm=last_rl_reference_grad_norm,
        source_log_rows=int(source_log_rows),
        group_log_rows=int(group_log_rows),
        window_log_rows=int(window_log_rows),
        final_auxiliary_flush=final_auxiliary_flush,
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
    for block, percentage in milestones.items():
        target = output / f"source-{percentage:03d}-adapter"
        if completed_source_blocks >= block and not target.exists():
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
) -> dict[str, Any]:
    validate_runtime_signature(runtime_signature)
    lambda_calibrated = load_calibrated_lambda(config, runtime_signature)
    output = Path(output_dir).resolve()
    if output != Path(config["output"]["run_dir"]).resolve():
        raise ValueError("Formal training output_dir differs from the frozen config.")
    _prepare_run_directory(output, config, runtime_signature, resume=resume)
    configure_deterministic_runtime(int(config["train"]["seed"]))
    recovery_root = output / "recovery"
    recovery = load_latest_recovery(recovery_root, runtime_signature) if resume else None
    if recovery is None:
        state = RecoveryState(
            next_source_block=0,
            optimization_window_index=0,
            completed_policy_steps=0,
            total_optimizer_steps=0,
            final_auxiliary_flush_steps=0,
            pending_buffer=None,
            last_rl_reference_grad_norm=None,
            source_log_rows=0,
            group_log_rows=0,
            window_log_rows=0,
            final_auxiliary_flush=None,
        )
        policy_path = None
        training_state = None
    else:
        policy_path, state, training_state = recovery
    source_log = output / "source_groups.jsonl"
    group_log = output / "groups.jsonl"
    window_log = output / "windows.jsonl"
    _truncate_jsonl(source_log, state.source_log_rows)
    _truncate_jsonl(group_log, state.group_log_rows)
    _truncate_jsonl(window_log, state.window_log_rows)
    recommendation, text, trie, blocks = _load_inputs(config)
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
    requested_stop = total_blocks
    if stop_after_source_blocks is not None:
        if int(stop_after_source_blocks) < 0:
            raise ValueError("stop_after_source_blocks must be non-negative.")
        requested_stop = min(
            total_blocks, state.next_source_block + int(stop_after_source_blocks)
        )
    optimization_window_index = state.optimization_window_index
    final_flush_steps = state.final_auxiliary_flush_steps
    final_auxiliary_flush = state.final_auxiliary_flush
    last_rl_reference = state.last_rl_reference_grad_norm
    source_rows = state.source_log_rows
    group_rows = state.group_log_rows
    window_rows = state.window_log_rows
    last_checkpoint: Path | None = recovery[0] if recovery is not None else None
    minimum = int(config["sampling"]["minimum_effective_groups_per_update"])

    while pending.next_source_block < requested_stop:
        block = blocks[pending.next_source_block]
        sampled = sample_source_block(
            bundle,
            recommendation,
            text,
            block,
            grammars,
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
            result = optimize_pending_buffer(
                bundle,
                optimizer,
                scheduler,
                pending,
                grammars,
                optimization_window_index=optimization_window_index,
                lambda_calibrated=lambda_calibrated,
                config=config,
                device=device,
            )
            _append_jsonl(
                group_log,
                _policy_group_rows(pending, optimization_window_index),
            )
            _append_jsonl(window_log, [_window_row(result, pending)])
            group_rows += len(pending.rl_groups)
            window_rows += 1
            last_rl_reference = result.execution.rl_reference_gradient_norm
            optimization_window_index += 1
            pending = PendingOptimizationBuffer.empty(
                scheduler.completed_policy_steps, pending.next_source_block
            )
            _assert_optimizer_state_step(
                optimizer, scheduler.completed_policy_steps + final_flush_steps
            )
            _save_milestones(bundle, output, pending.next_source_block)
        checkpoint_due = _periodic_checkpoint_due(
            pending.next_source_block,
            requested_stop=requested_stop,
            total_blocks=total_blocks,
            interval=int(config["train"]["checkpoint_source_blocks"]),
        )
        if checkpoint_due:
            committed = _checkpoint_state(
                pending=pending,
                optimization_window_index=optimization_window_index,
                completed_policy_steps=scheduler.completed_policy_steps,
                final_auxiliary_flush_steps=final_flush_steps,
                last_rl_reference_grad_norm=last_rl_reference,
                source_log_rows=source_rows,
                group_log_rows=group_rows,
                window_log_rows=window_rows,
                final_auxiliary_flush=final_auxiliary_flush,
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
    if complete_source and pending.rl_groups:
        result = optimize_pending_buffer(
            bundle,
            optimizer,
            scheduler,
            pending,
            grammars,
            optimization_window_index=optimization_window_index,
            lambda_calibrated=lambda_calibrated,
            config=config,
            device=device,
        )
        _append_jsonl(group_log, _policy_group_rows(pending, optimization_window_index))
        _append_jsonl(window_log, [_window_row(result, pending)])
        group_rows += len(pending.rl_groups)
        window_rows += 1
        last_rl_reference = result.execution.rl_reference_gradient_norm
        optimization_window_index += 1
        pending = PendingOptimizationBuffer.empty(
            scheduler.completed_policy_steps, total_blocks
        )
    elif complete_source and pending.anchor_groups:
        execution = flush_pending_anchor(
            bundle,
            optimizer,
            pending,
            grammars,
            lambda_calibrated=lambda_calibrated,
            last_rl_reference_gradient_norm=float(last_rl_reference or 0.0),
            config=config,
            device=device,
        )
        final_auxiliary_flush = _final_auxiliary_row(
            execution,
            pending,
            rl_reference_gradient_norm=float(last_rl_reference or 0.0),
        )
        final_flush_steps += execution.optimizer_steps
        pending = PendingOptimizationBuffer.empty(
            scheduler.completed_policy_steps, total_blocks
        )
    if complete_source:
        _assert_optimizer_state_step(
            optimizer, scheduler.completed_policy_steps + final_flush_steps
        )
        final_state = _checkpoint_state(
            pending=pending,
            optimization_window_index=optimization_window_index,
            completed_policy_steps=scheduler.completed_policy_steps,
            final_auxiliary_flush_steps=final_flush_steps,
            last_rl_reference_grad_norm=last_rl_reference,
            source_log_rows=source_rows,
            group_log_rows=group_rows,
            window_log_rows=window_rows,
            final_auxiliary_flush=final_auxiliary_flush,
        )
        checkpoint_path = recovery_root / (
            f"checkpoint-block-{final_state.next_source_block:06d}"
            f"-window-{final_state.optimization_window_index:06d}"
            f"-update-{final_state.total_optimizer_steps:06d}"
        )
        if not checkpoint_path.exists():
            last_checkpoint = save_recovery_checkpoint(
                bundle,
                optimizer,
                scheduler,
                final_state,
                recovery_root,
                runtime_signature,
            )
            _prune_recovery(
                recovery_root, int(config["train"]["retain_recovery_checkpoints"])
            )
        _save_milestones(bundle, output, total_blocks)
    expected_source_groups = int(config["data"]["recommendation_groups"])
    if bool(config["experiments"]["active"]["text_to_sid_enabled"]):
        expected_source_groups += int(config["data"]["text_to_sid_groups"])
    if complete_source and source_rows != expected_source_groups:
        raise RuntimeError(
            f"Source audit rows differ: expected={expected_source_groups}, actual={source_rows}."
        )
    summary = {
        "spec_version": contract.SPEC_VERSION,
        "arm": config["experiments"]["active"]["arm"],
        "completed_source_blocks": pending.next_source_block,
        "source_groups": source_rows,
        "candidate_rollouts": source_rows * contract.GROUP_SIZE,
        "optimization_windows": optimization_window_index,
        "completed_policy_steps": scheduler.completed_policy_steps,
        "final_auxiliary_flush_steps": final_flush_steps,
        "final_auxiliary_flush": final_auxiliary_flush,
        "total_optimizer_steps": scheduler.completed_policy_steps + final_flush_steps,
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
) -> dict[str, Any]:
    if formal and stop_after_source_blocks is not None:
        raise ValueError("Formal training cannot stop before source exhaustion.")
    if formal and dense_scoring:
        raise ValueError("Formal training cannot enable the dense-scoring test hook.")
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
    lambda_calibrated = load_calibrated_lambda(config, runtime_signature)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    configure_deterministic_runtime(int(config["train"]["seed"]))
    recommendation, text, trie, blocks = _load_inputs(config)
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
        sampled = sample_source_block(
            bundle,
            recommendation,
            text,
            blocks[pending.next_source_block],
            grammars,
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
        lambda_calibrated=lambda_calibrated,
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
    "load_calibrated_lambda",
    "run_calibration",
    "run_one_window_pilot",
    "run_training",
]
