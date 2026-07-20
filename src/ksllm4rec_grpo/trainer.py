"""Custom on-policy LoRA GRPO loop for the frozen V3.1 contract."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from ksllm4rec_orpo.data import Sid

from .checkpoint import (
    RecoveryCursor,
    capture_rng_state,
    load_latest_recovery,
    restore_rng_state,
    save_policy_atomic,
    save_recovery_checkpoint,
)
from .constraint import RecommendationGrammar
from .contract import expected_trie_leaf_count
from .data import RecommendationGroup, iter_groups
from .fingerprint import runtime_signature
from .modeling import DualAdapterModel, load_dual_adapter_model
from .objective import finalize_candidates, grpo_loss, rewards_and_advantages
from .prompt import encode_prompt
from .rollout import RolloutCandidate, rollout_group
from .scoring import CompletionScores, score_completions
from .trie import SidPrefixTrie


# Backwards-compatible values for code importing the historical constants.
# New runs derive these values from the configured group count below.
TOTAL_OPTIMIZER_STEPS = 1_596
WARMUP_STEPS = 48


def training_schedule(config: dict[str, Any], groups_per_epoch: int) -> tuple[int, int]:
    """Compute optimizer and warmup steps from the actual group stream."""

    epochs = int(config["train"]["epochs"])
    accumulation = int(config["train"]["gradient_accumulation_groups"])
    if epochs < 1 or accumulation < 1 or groups_per_epoch < 1:
        raise ValueError("epochs, accumulation, and groups must be positive")
    steps_per_epoch = math.ceil(groups_per_epoch / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = math.ceil(total_steps * float(config["train"]["warmup_ratio"]))
    return total_steps, warmup_steps


class PhaseOutOfMemory(RuntimeError):
    def __init__(self, phase: str) -> None:
        super().__init__(f"CUDA out of memory during {phase}.")
        self.phase = phase


@dataclass(frozen=True)
class RuntimeState:
    epoch_index: int
    next_group_offset: int
    global_step: int
    groups_completed: int
    rollout_chunk: int
    loss_chunk: int


@dataclass(frozen=True)
class GroupComputation:
    record: dict[str, Any]
    loss: float
    reward_mean: float
    reward_std: float
    signal: bool
    exact_live: bool
    forced: bool


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _pad_float_rows(
    rows: Sequence[Sequence[float]], width: int, device: torch.device
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=torch.float32, device=device)
    return torch.tensor(
        [list(row) + [0.0] * (width - len(row)) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _no_grad_score_chunks(
    bundle: DualAdapterModel,
    prompt_ids: Sequence[int],
    completions: Sequence[Sequence[int]],
    grammar: RecommendationGrammar,
    *,
    chunk_size: int,
    device: torch.device,
    adapter: str,
) -> list[CompletionScores]:
    model = bundle.model
    was_training = model.training
    model.train(adapter == "policy")
    results: list[CompletionScores] = []
    try:
        scope = (
            bundle.use_reference() if adapter == "reference" else bundle.use_policy()
        )
        with scope, torch.no_grad():
            for start in range(0, len(completions), chunk_size):
                scores = score_completions(
                    model,
                    prompt_ids,
                    completions[start : start + chunk_size],
                    grammar,
                    device=device,
                )
                results.append(
                    CompletionScores(
                        log_probs=scores.log_probs.detach().cpu(),
                        valid_mask=scores.valid_mask.cpu(),
                        decision_mask=scores.decision_mask.cpu(),
                    )
                )
    finally:
        model.train(was_training)
    return results


def _final_completion_rows(
    rollouts: Sequence[RolloutCandidate],
    final_sids: Sequence[Sid],
    forced_mask: Sequence[bool],
    grammar: RecommendationGrammar,
) -> list[list[int]]:
    rows = []
    for index, (sid, forced) in enumerate(zip(final_sids, forced_mask, strict=True)):
        rows.append(
            grammar.encode_sid(sid) if forced else list(rollouts[index].token_ids)
        )
    return rows


def process_group(
    bundle: DualAdapterModel,
    group: RecommendationGroup,
    grammar: RecommendationGrammar,
    prompt_ids: Sequence[int],
    *,
    epoch_index: int,
    rollout_chunk: int,
    loss_chunk: int,
    max_completion_length: int,
    window_groups: int,
    clip_epsilon: float,
    beta: float,
    std_epsilon: float,
    device: torch.device,
) -> GroupComputation:
    """Sample, score, and backpropagate one group's share of one window."""

    try:
        with bundle.use_policy():
            rollouts = rollout_group(
                bundle.model,
                prompt_ids,
                grammar,
                group_id=group.group_id,
                epoch_index=epoch_index,
                chunk_size=rollout_chunk,
                max_completion_length=max_completion_length,
                device=device,
            )
    except BaseException as exc:
        if _cuda_oom(exc):
            raise PhaseOutOfMemory("rollout") from exc
        raise

    positives = tuple(Sid.parse(value) for value in group.positive_sids)
    finalized = finalize_candidates(
        [candidate.sid for candidate in rollouts],
        positives,
        group_id=group.group_id,
        epoch_index=epoch_index,
    )
    completions = _final_completion_rows(
        rollouts,
        finalized.final_candidates,
        finalized.forced_mask,
        grammar,
    )
    reward = rewards_and_advantages(
        finalized.final_candidates, positives, epsilon=std_epsilon
    )

    try:
        old_policy_chunks = _no_grad_score_chunks(
            bundle,
            prompt_ids,
            completions,
            grammar,
            chunk_size=loss_chunk,
            device=device,
            adapter="policy",
        )
        reference_chunks = _no_grad_score_chunks(
            bundle,
            prompt_ids,
            completions,
            grammar,
            chunk_size=loss_chunk,
            device=device,
            adapter="reference",
        )
    except BaseException as exc:
        if _cuda_oom(exc):
            raise PhaseOutOfMemory("loss") from exc
        raise

    total_loss_value = 0.0
    max_old_new_delta = 0.0
    max_generation_rescore_delta = 0.0
    chunk_records: list[dict[str, Any]] = []
    bundle.activate_policy()
    bundle.model.train()
    for chunk_number, start in enumerate(range(0, 8, loss_chunk)):
        stop = min(start + loss_chunk, 8)
        try:
            current = score_completions(
                bundle.model,
                prompt_ids,
                completions[start:stop],
                grammar,
                device=device,
            )
            reference = reference_chunks[chunk_number]
            old_policy = old_policy_chunks[chunk_number]
            if not (
                torch.equal(current.decision_mask.cpu(), reference.decision_mask)
                and torch.equal(current.decision_mask.cpu(), old_policy.decision_mask)
            ):
                raise RuntimeError("Policy/old/reference decision masks differ.")
            forced = torch.tensor(
                finalized.forced_mask[start:stop], dtype=torch.bool, device=device
            )
            generation_rows = [
                rollouts[index].old_log_probs
                for index in range(start, stop)
                if not finalized.forced_mask[index]
            ]
            old = old_policy.log_probs.to(device)[~forced]
            generation = _pad_float_rows(
                generation_rows, current.log_probs.size(1), device
            )
            if generation_rows:
                live_current = current.log_probs[~forced]
                live_decision = current.decision_mask[~forced]
                rollout_decisions = torch.tensor(
                    [
                        list(rollouts[index].decision_mask)
                        + [False]
                        * (
                            current.log_probs.size(1)
                            - len(rollouts[index].decision_mask)
                        )
                        for index in range(start, stop)
                        if not finalized.forced_mask[index]
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                if not torch.equal(live_decision, rollout_decisions):
                    raise RuntimeError("Rollout/current decision masks differ.")
                delta = (live_current[live_decision] - old[live_decision]).abs()
                chunk_delta = (
                    float(delta.max().detach().item()) if delta.numel() else 0.0
                )
                if chunk_delta > 1.0e-5:
                    raise RuntimeError(
                        "Strict on-policy log-probability mismatch before backward: "
                        f"max_abs_delta={chunk_delta}."
                    )
                max_old_new_delta = max(max_old_new_delta, chunk_delta)
                generation_delta = (
                    old[live_decision] - generation[live_decision]
                ).abs()
                max_generation_rescore_delta = max(
                    max_generation_rescore_delta,
                    (
                        float(generation_delta.max().detach().item())
                        if generation_delta.numel()
                        else 0.0
                    ),
                )
            advantages = reward.advantages[start:stop].to(device)
            output = grpo_loss(
                current.log_probs,
                reference.log_probs.to(device),
                old,
                advantages,
                current.decision_mask,
                forced,
                clip_epsilon=clip_epsilon,
                beta=beta,
            )
            scaled = output.loss * ((stop - start) / 8.0) / window_groups
            scaled.backward()
            total_loss_value += float(output.loss.detach().item()) * (
                (stop - start) / 8.0
            )
            chunk_records.append(
                {
                    "start": start,
                    "stop": stop,
                    "loss": float(output.loss.detach().item()),
                    "decision_counts": output.decision_counts.detach().cpu().tolist(),
                    "reference_kl_mean": float(
                        output.reference_kl[current.decision_mask]
                        .mean()
                        .detach()
                        .item()
                    ),
                    "per_candidate_loss": output.per_candidate_loss.detach()
                    .cpu()
                    .tolist(),
                    "per_candidate_reference_kl": (
                        output.reference_kl.sum(dim=-1)
                        / output.decision_counts.to(torch.float32)
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                }
            )
            del current, output, scaled
        except BaseException as exc:
            if _cuda_oom(exc):
                raise PhaseOutOfMemory("loss") from exc
            raise

    live_sids = [candidate.sid.render() for candidate in rollouts]
    final_sids = [candidate.render() for candidate in finalized.final_candidates]
    record = {
        "epoch": epoch_index,
        "group_id": group.group_id,
        "source_lines": list(group.source_lines),
        "positive_sids": list(group.positive_sids),
        "live_sids": live_sids,
        "final_sids": final_sids,
        "has_live_gt": finalized.has_live_gt,
        "add_gt": finalized.add_gt,
        "add_gt_uniform": finalized.add_gt_uniform,
        "forced_gt": finalized.forced_gt.render() if finalized.forced_gt else None,
        "discarded_rollout": (
            finalized.discarded_rollout.render()
            if finalized.discarded_rollout
            else None
        ),
        "forced_mask": list(finalized.forced_mask),
        "rewards": reward.rewards.tolist(),
        "advantages": reward.advantages.tolist(),
        "reward_mean": float(reward.mean_reward.item()),
        "reward_std": float(reward.std_reward.item()),
        "reward_tiers": list(reward.tiers),
        "loss": total_loss_value,
        "max_old_new_logp_delta": max_old_new_delta,
        "max_generation_rescore_logp_delta": max_generation_rescore_delta,
        "chunks": chunk_records,
    }
    return GroupComputation(
        record=record,
        loss=total_loss_value,
        reward_mean=float(reward.mean_reward.item()),
        reward_std=float(reward.std_reward.item()),
        signal=bool(reward.advantages.abs().max().item() > 0.0),
        exact_live=finalized.has_live_gt,
        forced=finalized.add_gt,
    )


def _optimizer_and_scheduler(bundle: DualAdapterModel, config: dict[str, Any]):
    train = config["train"]
    parameters = [
        parameter for parameter in bundle.model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Policy has no trainable parameters.")
    groups_per_epoch = int(config["data"]["groups"])
    total_steps, warmup_steps = training_schedule(config, groups_per_epoch)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train["learning_rate"]),
        betas=(float(train["adam_beta1"]), float(train["adam_beta2"])),
        eps=float(train["adam_epsilon"]),
        weight_decay=float(train["weight_decay"]),
        fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, parameters


def _append_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _truncate_jsonl(path: Path, keep: int) -> list[dict[str, Any]]:
    """Atomically keep a verified prefix and discard work after a checkpoint."""

    if keep < 0:
        raise ValueError("JSONL keep count must be non-negative.")
    if not path.exists():
        if keep:
            raise RuntimeError(f"Recovery expects {keep} rows but {path} is missing.")
        return []
    with path.open("rb") as handle:
        lines = handle.readlines()
    if len(lines) < keep:
        raise RuntimeError(
            f"Recovery expects {keep} rows in {path}, found only {len(lines)}."
        )
    records = []
    for index, raw in enumerate(lines[:keep], start=1):
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Corrupt committed JSONL row {index} in {path}."
            ) from exc
    if len(lines) != keep:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.writelines(lines[:keep])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return records


def _reconcile_resume_outputs(
    output_dir: Path,
    groups: Sequence[RecommendationGroup],
    state: RuntimeState,
    *,
    has_recovery: bool,
    epochs: int,
) -> None:
    """Make append-only logs and epoch outputs exactly match the recovery cursor."""

    audit = _truncate_jsonl(output_dir / "train_audit.jsonl", state.groups_completed)
    for index, row in enumerate(audit):
        expected_epoch = index // len(groups) + 1
        expected_group = groups[index % len(groups)].group_id
        if row.get("epoch") != expected_epoch or row.get("group_id") != expected_group:
            raise RuntimeError(
                f"Audit prefix diverges from recovery cursor at row {index + 1}."
            )
    progress = _truncate_jsonl(output_dir / "train_progress.jsonl", state.global_step)
    for index, row in enumerate(progress, start=1):
        if row.get("optimizer_step") != index:
            raise RuntimeError(f"Progress optimizer step diverges at row {index}.")
    if progress and progress[-1].get("groups_completed") != state.groups_completed:
        raise RuntimeError("Progress log and recovery group cursor differ.")

    for epoch in range(1, epochs + 1):
        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        if epoch >= state.epoch_index and epoch_dir.exists():
            shutil.rmtree(epoch_dir)
    if not has_recovery:
        recovery_root = output_dir / "recovery"
        if recovery_root.exists():
            shutil.rmtree(recovery_root)
    if state.epoch_index <= epochs:
        summary_path = output_dir / "run_summary.json"
        if summary_path.exists():
            summary_path.unlink()


def _has_training_artifacts(output_dir: Path) -> bool:
    names = ("train_audit.jsonl", "train_progress.jsonl", "run_summary.json")
    if any((output_dir / name).exists() for name in names) or any(
        output_dir.glob("epoch_[0-9][0-9][0-9]")
    ):
        return True
    recovery = output_dir / "recovery"
    return (recovery / "latest.json").exists() or any(
        recovery.glob("checkpoint-step-*")
    )


def _validate_runtime_state(
    state: RuntimeState,
    groups_per_epoch: int,
    *,
    epochs: int,
    accumulation_groups: int,
) -> None:
    if state.rollout_chunk != state.loss_chunk:
        raise RuntimeError("Recovery cursor violates the unified on-policy chunk rule.")
    if state.epoch_index not in range(1, epochs + 2):
        raise RuntimeError(f"Invalid recovery epoch: {state.epoch_index}.")
    if state.epoch_index == epochs + 1:
        expected_groups = groups_per_epoch * epochs
        expected_steps = math.ceil(groups_per_epoch / accumulation_groups) * epochs
        if state.next_group_offset != 0:
            raise RuntimeError("Completed recovery cursor must have offset zero.")
    else:
        if not 0 <= state.next_group_offset <= groups_per_epoch:
            raise RuntimeError("Recovery group offset is outside one epoch.")
        completed_epochs = state.epoch_index - 1
        expected_groups = completed_epochs * groups_per_epoch + state.next_group_offset
        expected_steps = completed_epochs * math.ceil(
            groups_per_epoch / accumulation_groups
        ) + math.ceil(state.next_group_offset / accumulation_groups)
    if state.groups_completed != expected_groups or state.global_step != expected_steps:
        raise RuntimeError(
            "Recovery cursor counts are inconsistent: "
            f"state={asdict(state)}, expected_groups={expected_groups}, "
            f"expected_steps={expected_steps}."
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Corrupt JSONL row {line_number} in {path}."
                ) from exc
    return rows


def _aggregate_logs(audit_path: Path, progress_path: Path) -> dict[str, Any]:
    audit = _read_jsonl(audit_path)
    progress = _read_jsonl(progress_path)
    losses = [float(row["loss"]) for row in audit]
    reward_means = [float(row["reward_mean"]) for row in audit]
    reward_stds = [float(row["reward_std"]) for row in audit]
    gradient_norms = [float(row["gradient_norm"]) for row in progress]
    train_seconds = sum(float(row.get("window_seconds", 0.0)) for row in progress)
    return {
        "groups": len(audit),
        "train_seconds": train_seconds,
        "loss_mean": fmean(losses) if losses else None,
        "reward_mean": fmean(reward_means) if reward_means else None,
        "reward_std_mean": fmean(reward_stds) if reward_stds else None,
        "signal_groups": sum(
            max(abs(float(value)) for value in row["advantages"]) > 0.0 for row in audit
        ),
        "exact_live_groups": sum(bool(row["has_live_gt"]) for row in audit),
        "forced_groups": sum(bool(row["add_gt"]) for row in audit),
        "gradient_norm_min": min(gradient_norms) if gradient_norms else None,
        "gradient_norm_max": max(gradient_norms) if gradient_norms else None,
        "peak_allocated_gib": max(
            (float(row.get("peak_allocated_gib", 0.0)) for row in progress),
            default=0.0,
        ),
        "peak_reserved_gib": max(
            (float(row.get("peak_reserved_gib", 0.0)) for row in progress),
            default=0.0,
        ),
    }


def run_training(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    output_dir: Path,
    rollout_chunk: int,
    loss_chunk: int,
    max_groups: int | None = None,
    resume: bool = True,
    save_recovery: bool = True,
    save_epochs: bool = True,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the approved two-epoch loop or an isolated gate prefix."""

    if rollout_chunk not in (8, 4, 2, 1) or loss_chunk not in (8, 4, 2, 1):
        raise ValueError("Candidate chunks must be one of 8, 4, 2, 1.")
    if rollout_chunk != loss_chunk:
        raise ValueError(
            "Strict on-policy execution requires rollout_chunk == loss_chunk."
        )
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive.")
    torch_device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["train"]["tf32"])
    set_global_seed(int(config["train"]["seed"]))

    groups = list(iter_groups(groups_path))
    if len(groups) != int(config["data"]["groups"]):
        raise RuntimeError(
            f"Expected {config['data']['groups']} groups, got {len(groups)}."
        )
    trie = SidPrefixTrie.load(
        trie_dir, expected_leaf_count=expected_trie_leaf_count(config)
    )
    recovery_root = output_dir / "recovery"
    contract_signature = runtime_signature(config, groups_path, trie_dir)
    recovery = (
        load_latest_recovery(recovery_root, contract_signature) if resume else None
    )
    if resume and recovery is None and _has_training_artifacts(output_dir):
        raise RuntimeError(
            "Training artifacts exist but no valid recovery checkpoint was found; "
            "refusing to delete or restart them implicitly."
        )
    policy_path = recovery[0] if recovery else None
    if recovery:
        _, cursor, training_state = recovery
        state = RuntimeState(**asdict(cursor))
    else:
        training_state = None
        state = RuntimeState(1, 0, 0, 0, rollout_chunk, loss_chunk)
    epochs = int(config["train"]["epochs"])
    accumulation = int(config["train"]["gradient_accumulation_groups"])
    _validate_runtime_state(
        state,
        len(groups),
        epochs=epochs,
        accumulation_groups=accumulation,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _reconcile_resume_outputs(
        output_dir,
        groups,
        state,
        has_recovery=recovery is not None,
        epochs=epochs,
    )
    if state.epoch_index > epochs:
        summary_path = output_dir / "run_summary.json"
        if summary_path.is_file():
            completed = json.loads(summary_path.read_text(encoding="utf-8"))
            if completed.get("state") != asdict(state):
                raise RuntimeError("Completed summary and recovery cursor differ.")
            return completed
    started_loading = time.monotonic()
    bundle = load_dual_adapter_model(
        config, device=torch_device, policy_adapter_path=policy_path
    )
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    optimizer, scheduler, trainable_parameters = _optimizer_and_scheduler(
        bundle, config
    )

    if training_state is not None:
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        restore_rng_state(training_state["rng"])
    model_load_seconds = time.monotonic() - started_loading

    audit_path = output_dir / "train_audit.jsonl"
    progress_path = output_dir / "train_progress.jsonl"
    initial_parameter = trainable_parameters[0].detach().float().cpu().clone()
    if save_recovery and recovery is None:
        save_recovery_checkpoint(
            bundle,
            optimizer,
            scheduler,
            RecoveryCursor(**asdict(state)),
            recovery_root,
            contract_signature,
        )
    started_training = time.monotonic()
    run_groups = 0
    oom_retries: list[dict[str, Any]] = []
    for epoch_index in range(state.epoch_index, epochs + 1):
        offset = state.next_group_offset if epoch_index == state.epoch_index else 0
        while offset < len(groups):
            if max_groups is not None and run_groups >= max_groups:
                break
            remaining_gate = (
                max_groups - run_groups if max_groups is not None else accumulation
            )
            window_size = min(accumulation, len(groups) - offset, remaining_gate)
            window = groups[offset : offset + window_size]
            window_started = time.monotonic()
            window_rng = capture_rng_state()
            while True:
                optimizer.zero_grad(set_to_none=True)
                records: list[dict[str, Any]] = []
                computations: list[GroupComputation] = []
                try:
                    for group in window:
                        prompt_ids = encode_prompt(
                            bundle.tokenizer,
                            group.system,
                            group.prompt,
                            cutoff_len=int(config["data"]["cutoff_len"]),
                        )
                        computation = process_group(
                            bundle,
                            group,
                            grammar,
                            prompt_ids,
                            epoch_index=epoch_index,
                            rollout_chunk=state.rollout_chunk,
                            loss_chunk=state.loss_chunk,
                            max_completion_length=int(
                                config["rollout"]["max_completion_length"]
                            ),
                            window_groups=window_size,
                            clip_epsilon=float(config["loss"]["clip_epsilon"]),
                            beta=float(config["loss"]["reference_beta"]),
                            std_epsilon=float(config["reward"]["std_epsilon"]),
                            device=torch_device,
                        )
                        records.append(computation.record)
                        computations.append(computation)
                    break
                except PhaseOutOfMemory as exc:
                    optimizer.zero_grad(set_to_none=True)
                    restore_rng_state(window_rng)
                    del records, computations
                    torch.cuda.empty_cache()
                    previous = state.rollout_chunk
                    if previous == 1:
                        raise RuntimeError(
                            f"OOM persisted at minimum {exc.phase} chunk=1."
                        ) from exc
                    reduced = previous // 2
                    state = RuntimeState(
                        state.epoch_index,
                        state.next_group_offset,
                        state.global_step,
                        state.groups_completed,
                        reduced,
                        reduced,
                    )
                    oom_retries.append(
                        {
                            "epoch": epoch_index,
                            "offset": offset,
                            "phase": exc.phase,
                            "previous_chunk": previous,
                            "new_chunk": reduced,
                        }
                    )

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(config["train"]["max_grad_norm"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"Non-finite gradient norm: {gradient_norm}.")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            offset += window_size
            run_groups += window_size
            global_step = state.global_step + 1
            groups_completed = state.groups_completed + window_size
            state = RuntimeState(
                epoch_index,
                offset,
                global_step,
                groups_completed,
                state.rollout_chunk,
                state.loss_chunk,
            )
            for item in records:
                item["optimizer_step"] = global_step
                item["learning_rate"] = scheduler.get_last_lr()[0]
                item["gradient_norm"] = float(gradient_norm.item())
            _append_records(audit_path, records)

            progress = {
                "epoch": epoch_index,
                "next_group_offset": offset,
                "optimizer_step": global_step,
                "groups_completed": groups_completed,
                "window_groups": window_size,
                "loss_mean": fmean(item.loss for item in computations),
                "reward_mean": fmean(item.reward_mean for item in computations),
                "reward_std_mean": fmean(item.reward_std for item in computations),
                "gradient_norm": float(gradient_norm.item()),
                "learning_rate": scheduler.get_last_lr()[0],
                "rollout_chunk": state.rollout_chunk,
                "loss_chunk": state.loss_chunk,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "window_seconds": time.monotonic() - window_started,
            }
            _append_records(progress_path, [progress])

            if (
                save_recovery
                and global_step % int(config["train"]["resume_save_steps"]) == 0
            ):
                save_recovery_checkpoint(
                    bundle,
                    optimizer,
                    scheduler,
                    RecoveryCursor(**asdict(state)),
                    recovery_root,
                    contract_signature,
                )

        if max_groups is not None and run_groups >= max_groups:
            break
        if offset != len(groups):
            raise RuntimeError("Epoch stopped at an unexpected group offset.")
        if save_epochs:
            save_policy_atomic(bundle, output_dir / f"epoch_{epoch_index:03d}")
        next_state = RuntimeState(
            epoch_index + 1,
            0,
            state.global_step,
            state.groups_completed,
            state.rollout_chunk,
            state.loss_chunk,
        )
        if save_recovery:
            save_recovery_checkpoint(
                bundle,
                optimizer,
                scheduler,
                RecoveryCursor(**asdict(next_state)),
                recovery_root,
                contract_signature,
            )
        state = next_state

    train_seconds_this_invocation = time.monotonic() - started_training
    final_parameter = trainable_parameters[0].detach().float().cpu()
    parameter_max_change = float(
        (final_parameter - initial_parameter).abs().max().item()
    )
    aggregate = _aggregate_logs(audit_path, progress_path)
    result = {
        "model_load_seconds": model_load_seconds,
        "train_seconds": aggregate["train_seconds"],
        "train_seconds_this_invocation": train_seconds_this_invocation,
        "groups_run": aggregate["groups"],
        "groups_run_this_invocation": run_groups,
        "groups_per_second": (
            aggregate["groups"] / aggregate["train_seconds"]
            if aggregate["train_seconds"]
            else 0.0
        ),
        "state": asdict(state),
        "loss_mean": aggregate["loss_mean"],
        "reward_mean": aggregate["reward_mean"],
        "reward_std_mean": aggregate["reward_std_mean"],
        "signal_groups": aggregate["signal_groups"],
        "exact_live_groups": aggregate["exact_live_groups"],
        "forced_groups": aggregate["forced_groups"],
        "gradient_norm_min": aggregate["gradient_norm_min"],
        "gradient_norm_max": aggregate["gradient_norm_max"],
        "parameter_max_change_this_invocation": parameter_max_change,
        "oom_retries_this_invocation": oom_retries,
        "peak_allocated_gib": aggregate["peak_allocated_gib"],
        "peak_reserved_gib": aggregate["peak_reserved_gib"],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
