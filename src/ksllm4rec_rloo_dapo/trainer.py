"""Windowed K=1 RLOO training with dynamic effective-group collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ksllm4rec_grpo.constraint import RecommendationGrammar
from ksllm4rec_grpo.data import RecommendationGroup, iter_groups
from ksllm4rec_grpo.prompt import encode_prompt
from ksllm4rec_grpo.trie import SidPrefixTrie
from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.checkpoint import save_policy_atomic
from ksllm4rec_rloo.modeling import PolicyModel, load_policy_model

from . import contract
from .checkpoint import (
    RecoveryCursor,
    load_latest_recovery,
    recovery_checkpoint_due,
    restore_training_state,
    save_recovery_checkpoint,
)
from .fingerprint import validate_runtime_signature
from .objective import RewardOutput, clipped_rloo_token_sum, rewards_and_advantages
from .rollout import (
    CacheRolloutStats,
    CanonicalCandidate,
    PromptRequest,
    canonicalize_prompt_rollout,
    rollout_prompt_batch,
)
from .sampling import (
    DeterministicGroupStream,
    SourceCursor,
    append_effective_groups,
)
from .scoring import (
    dense_scoring_mode,
    grammar_completion_width,
    score_completions_dense_fixed,
)


@dataclass(frozen=True)
class PreparedEffectiveGroup:
    group: RecommendationGroup
    prompt_ids: tuple[int, ...]
    positives: tuple[Sid, ...]
    candidates: tuple[CanonicalCandidate, ...]
    reward: RewardOutput
    proposal_target_max_logp_difference: float
    corrected_candidate_count: int
    target_forward_calls: int


@dataclass(frozen=True)
class RawBatchAudit:
    raw_group_ids: tuple[str, ...]
    effective_group_ids: tuple[str, ...]
    filtered_group_ids: tuple[str, ...]
    overflow_group_ids: tuple[str, ...]
    cache_stats: CacheRolloutStats


@dataclass(frozen=True)
class WindowCollection:
    groups: tuple[PreparedEffectiveGroup, ...]
    source_cursor: SourceCursor
    raw_prompt_count: int
    filtered_group_count: int
    overflow_group_count: int
    audits: tuple[RawBatchAudit, ...]


@dataclass(frozen=True)
class OptimizerStepResult:
    loss: float
    learning_rate: float
    decision_tokens: int
    gradient_norm: float
    max_replay_logp_difference: float
    ratio_min: float
    ratio_max: float
    ratio_mean: float
    clipped_tokens: int
    below_low_tokens: int
    above_high_tokens: int
    group_ids: tuple[str, ...]


@dataclass(frozen=True)
class WindowTrainingResult:
    window_index: int
    learning_rate: float
    optimizer_updates: int
    group_ids: tuple[str, ...]
    steps: tuple[OptimizerStepResult, ...]


def learning_rate_for_window(config: Mapping[str, Any], window_index: int) -> float:
    train = config["train"]
    total = int(train["total_windows"])
    warmup = int(train["warmup_windows"])
    if isinstance(window_index, bool) or not 0 <= int(window_index) < total:
        raise ValueError(f"window_index must be in [0,{total - 1}].")
    if warmup <= 0:
        raise ValueError("warmup_windows must be positive.")
    return float(train["learning_rate"]) * min(int(window_index) / warmup, 1.0)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


class WindowLRScheduler:
    """Linear warmup counted once per completed logical rollout window."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
        *,
        completed_windows: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.completed_windows = int(completed_windows)
        total = int(config["train"]["total_windows"])
        if not 0 <= self.completed_windows <= total:
            raise ValueError("completed_windows is outside the training schedule.")
        if self.completed_windows < total:
            _set_optimizer_lr(
                optimizer,
                learning_rate_for_window(config, self.completed_windows),
            )

    def step(self) -> None:
        total = int(self.config["train"]["total_windows"])
        if self.completed_windows >= total:
            raise RuntimeError("Window scheduler is already complete.")
        self.completed_windows += 1
        if self.completed_windows < total:
            _set_optimizer_lr(
                self.optimizer,
                learning_rate_for_window(self.config, self.completed_windows),
            )

    def state_dict(self) -> dict[str, int]:
        return {"completed_windows": self.completed_windows}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if set(value) != {"completed_windows"}:
            raise ValueError("Window scheduler state has an invalid schema.")
        completed = value["completed_windows"]
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise TypeError("completed_windows must be an integer.")
        self.completed_windows = completed
        total = int(self.config["train"]["total_windows"])
        if not 0 <= completed <= total:
            raise ValueError("Restored scheduler position is outside the schedule.")
        if completed < total:
            _set_optimizer_lr(
                self.optimizer, learning_rate_for_window(self.config, completed)
            )


def partition_effective_groups(
    groups: Sequence[Any], *, window_index: int, seed: int
) -> tuple[tuple[Any, ...], ...]:
    values = tuple(groups)
    if len(values) != contract.EFFECTIVE_GROUPS_PER_WINDOW:
        raise ValueError("A complete window requires exactly 32 effective groups.")
    ids = [item.group.group_id for item in values]
    if len(set(ids)) != len(ids):
        raise ValueError("Effective group IDs must be unique within a window.")
    prefix = f"minibatch-order|{int(seed)}|{int(window_index)}|".encode("utf-8")
    ordered = tuple(
        sorted(
            values,
            key=lambda item: hashlib.sha256(
                prefix + item.group.group_id.encode("utf-8")
            ).digest(),
        )
    )
    batches = tuple(
        ordered[start : start + contract.MINIBATCH_GROUPS]
        for start in range(0, len(ordered), contract.MINIBATCH_GROUPS)
    )
    if len(batches) != contract.OPTIMIZER_UPDATES_PER_WINDOW or any(
        len(batch) != contract.MINIBATCH_GROUPS for batch in batches
    ):
        raise RuntimeError("Window partition did not produce four 8-group minibatches.")
    return batches


def load_groups_and_trie(
    config: Mapping[str, Any],
) -> tuple[list[RecommendationGroup], SidPrefixTrie]:
    groups_path = Path(config["output"]["groups_dir"]) / "groups.jsonl"
    groups = list(iter_groups(groups_path))
    if len(groups) != int(config["data"]["groups"]):
        raise RuntimeError(
            f"Expected {config['data']['groups']} groups, found {len(groups)}."
        )
    trie = SidPrefixTrie.load(
        Path(config["output"]["trie_dir"]),
        expected_leaf_count=int(config["trie"]["unique_sids"]),
    )
    return groups, trie


def _prepare_prompt(
    bundle: PolicyModel, group: RecommendationGroup, cutoff: int
) -> tuple[int, ...]:
    return tuple(
        encode_prompt(
            bundle.tokenizer,
            group.system,
            group.prompt,
            cutoff_len=int(cutoff),
        )
    )


def collect_effective_window(
    bundle: PolicyModel,
    stream: DeterministicGroupStream,
    grammar: RecommendationGrammar,
    *,
    window_index: int,
    config: Mapping[str, Any],
    device: str | torch.device,
    aligned_cache: bool = True,
) -> WindowCollection:
    target = int(config["sampling"]["effective_groups_per_window"])
    raw_batch_size = int(config["rollout"]["prompt_batch_size"])
    maximum_unique = int(config["sampling"]["max_unique_prompts_per_window"])
    seen_ids: set[str] = set()
    retained: tuple[PreparedEffectiveGroup, ...] = ()
    audits: list[RawBatchAudit] = []
    filtered_count = 0
    overflow_count = 0

    while len(retained) < target:
        remaining_unique = maximum_unique - len(seen_ids)
        if remaining_unique <= 0:
            raise RuntimeError(
                "A full dataset traversal did not produce 32 effective groups."
            )
        raw_groups = stream.take_unique(min(raw_batch_size, remaining_unique), seen_ids)
        requests = tuple(
            PromptRequest(
                group_id=group.group_id,
                prompt_ids=_prepare_prompt(
                    bundle, group, int(config["data"]["cutoff_len"])
                ),
            )
            for group in raw_groups
        )
        proposals, cache_stats = rollout_prompt_batch(
            bundle.model,
            requests,
            grammar,
            sampling_nonce=int(window_index),
            config=dict(config),
            device=device,
            aligned_cache=aligned_cache,
        )
        effective: list[PreparedEffectiveGroup] = []
        filtered_ids: list[str] = []
        for group, proposal in zip(raw_groups, proposals, strict=True):
            canonical = canonicalize_prompt_rollout(
                bundle.model,
                proposal,
                grammar,
                sampling_nonce=int(window_index),
                temperature=float(config["rollout"]["temperature"]),
                max_difference=float(
                    config["rollout"]["sample_canonical_max_logp_difference"]
                ),
                device=device,
            )
            positives = tuple(Sid.parse(value) for value in group.positive_sids)
            reward = rewards_and_advantages(
                [candidate.sid for candidate in canonical.candidates], positives
            )
            if not reward.effective:
                filtered_ids.append(group.group_id)
                continue
            effective.append(
                PreparedEffectiveGroup(
                    group=group,
                    prompt_ids=proposal.request.prompt_ids,
                    positives=positives,
                    candidates=canonical.candidates,
                    reward=reward,
                    proposal_target_max_logp_difference=(
                        canonical.max_proposal_canonical_logp_difference
                    ),
                    corrected_candidate_count=canonical.corrected_candidate_count,
                    target_forward_calls=canonical.target_forward_calls,
                )
            )

        appended = append_effective_groups(retained, effective, target=target)
        retained = appended.retained
        overflow_ids = tuple(item.group.group_id for item in appended.overflow)
        filtered_count += len(filtered_ids)
        overflow_count += len(overflow_ids)
        audits.append(
            RawBatchAudit(
                raw_group_ids=tuple(group.group_id for group in raw_groups),
                effective_group_ids=tuple(item.group.group_id for item in effective),
                filtered_group_ids=tuple(filtered_ids),
                overflow_group_ids=overflow_ids,
                cache_stats=cache_stats,
            )
        )

    if (
        len(retained) != target
        or len({item.group.group_id for item in retained}) != target
    ):
        raise RuntimeError("Effective window is not exactly 32 unique groups.")
    return WindowCollection(
        groups=retained,
        source_cursor=stream.cursor,
        raw_prompt_count=len(seen_ids),
        filtered_group_count=filtered_count,
        overflow_group_count=overflow_count,
        audits=tuple(audits),
    )


def _trainable_parameters(bundle: PolicyModel) -> list[torch.nn.Parameter]:
    values = [
        parameter for parameter in bundle.model.parameters() if parameter.requires_grad
    ]
    if not values:
        raise RuntimeError("Policy has no trainable LoRA parameters.")
    return values


def build_optimizer(
    bundle: PolicyModel, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    train = config["train"]
    kwargs: dict[str, Any] = {
        "lr": 0.0,
        "betas": (float(train["adam_beta1"]), float(train["adam_beta2"])),
        "eps": float(train["adam_epsilon"]),
        "weight_decay": float(train["weight_decay"]),
    }
    if next(bundle.model.parameters()).is_cuda:
        kwargs["fused"] = True
    return torch.optim.AdamW(_trainable_parameters(bundle), **kwargs)


def _candidate_old_tensors(
    candidates: Sequence[CanonicalCandidate],
    *,
    device: torch.device,
    completion_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    old = torch.zeros(
        (len(candidates), int(completion_width)),
        dtype=torch.float32,
        device=device,
    )
    decisions = torch.zeros_like(old, dtype=torch.bool)
    for row, candidate in enumerate(candidates):
        length = len(candidate.old_log_probs)
        old[row, :length] = torch.tensor(
            candidate.old_log_probs, dtype=torch.float32, device=device
        )
        decisions[row, :length] = torch.tensor(
            candidate.decision_mask, dtype=torch.bool, device=device
        )
    return old, decisions


def _gradient_l2_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total: torch.Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    result = float(total.sqrt().item())
    if not math.isfinite(result):
        raise FloatingPointError("Gradient norm is NaN or Inf.")
    return result


def train_minibatch(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    groups: Sequence[PreparedEffectiveGroup],
    grammar: RecommendationGrammar,
    *,
    config: Mapping[str, Any],
    device: str | torch.device,
    enforce_replay_gate: bool,
) -> OptimizerStepResult:
    values = tuple(groups)
    if len(values) != contract.MINIBATCH_GROUPS:
        raise ValueError("An optimizer minibatch requires exactly eight groups.")
    torch_device = torch.device(device)
    total_decisions = sum(
        sum(candidate.decision_mask)
        for group in values
        for candidate in group.candidates
    )
    if total_decisions <= 0:
        raise RuntimeError("Optimizer minibatch contains no decision token.")
    parameters = _trainable_parameters(bundle)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    maximum_delta = 0.0
    ratios: list[float] = []
    clipped_tokens = 0
    below_tokens = 0
    above_tokens = 0
    replay_limit = float(config["rollout"]["canonical_replay_max_logp_difference"])
    scoring_width = grammar_completion_width(grammar)

    for prepared in values:
        advantages = prepared.reward.advantages.to(torch_device)
        for start in (0, contract.LOSS_CHUNK_SIZE):
            chunk = prepared.candidates[start : start + contract.LOSS_CHUNK_SIZE]
            scores = score_completions_dense_fixed(
                bundle.model,
                prepared.prompt_ids,
                [candidate.token_ids for candidate in chunk],
                grammar,
                device=torch_device,
                temperature=float(config["rollout"]["temperature"]),
                batch_rows=contract.LOSS_CHUNK_SIZE,
                completion_width=scoring_width,
            )
            old, expected_mask = _candidate_old_tensors(
                chunk, device=torch_device, completion_width=scoring_width
            )
            if not torch.equal(scores.decision_mask, expected_mask):
                raise RuntimeError("Canonical old/new decision masks differ.")
            delta = (scores.log_probs - old).abs()[scores.decision_mask]
            chunk_delta = float(delta.max().detach().item()) if delta.numel() else 0.0
            maximum_delta = max(maximum_delta, chunk_delta)
            if enforce_replay_gate and chunk_delta > replay_limit:
                raise RuntimeError(
                    "Canonical replay probability gate failed before update: "
                    f"max_abs_delta={chunk_delta:.9g}, limit={replay_limit:.9g}."
                )
            output = clipped_rloo_token_sum(
                scores.log_probs,
                old,
                advantages[start : start + contract.LOSS_CHUNK_SIZE],
                scores.decision_mask,
                clip_low=float(config["loss"]["clip_ratio_low"]),
                clip_high=float(config["loss"]["clip_ratio_high"]),
            )
            (output.loss_sum / total_decisions).backward()
            loss_sum += float(output.loss_sum.detach().item())
            ratios.extend(float(value) for value in output.ratios.cpu().tolist())
            clipped_tokens += int(output.clip_active.sum().item())
            below_tokens += int(output.below_low.sum().item())
            above_tokens += int(output.above_high.sum().item())
            del scores, old, expected_mask, output

    gradient_norm = _gradient_l2_norm(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, float(config["train"]["max_grad_norm"]))
    optimizer.step()
    if not ratios or not all(math.isfinite(value) and value > 0.0 for value in ratios):
        raise FloatingPointError(
            "Optimizer minibatch ratios are not positive finite values."
        )
    return OptimizerStepResult(
        loss=loss_sum / total_decisions,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
        decision_tokens=total_decisions,
        gradient_norm=gradient_norm,
        max_replay_logp_difference=maximum_delta,
        ratio_min=min(ratios),
        ratio_max=max(ratios),
        ratio_mean=sum(ratios) / len(ratios),
        clipped_tokens=clipped_tokens,
        below_low_tokens=below_tokens,
        above_high_tokens=above_tokens,
        group_ids=tuple(group.group.group_id for group in values),
    )


def train_complete_window(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WindowLRScheduler,
    groups: Sequence[PreparedEffectiveGroup],
    grammar: RecommendationGrammar,
    *,
    window_index: int,
    config: Mapping[str, Any],
    device: str | torch.device,
) -> WindowTrainingResult:
    if scheduler.completed_windows != int(window_index):
        raise RuntimeError("Scheduler and requested window index differ.")
    expected_lr = learning_rate_for_window(config, window_index)
    if any(
        not math.isclose(float(group["lr"]), expected_lr, rel_tol=0.0, abs_tol=1e-15)
        for group in optimizer.param_groups
    ):
        raise RuntimeError("Optimizer LR differs from the window schedule.")
    minibatches = partition_effective_groups(
        groups,
        window_index=int(window_index),
        seed=int(config["train"]["seed"]),
    )
    steps: list[OptimizerStepResult] = []
    for minibatch_index, minibatch in enumerate(minibatches):
        step = train_minibatch(
            bundle,
            optimizer,
            minibatch,
            grammar,
            config=config,
            device=device,
            enforce_replay_gate=minibatch_index == 0,
        )
        steps.append(step)
        print(
            json.dumps(
                {
                    "event": "optimizer_step_complete",
                    "window_index": int(window_index),
                    "minibatch_index": minibatch_index,
                    "loss": step.loss,
                    "learning_rate": step.learning_rate,
                    "ratio_min": step.ratio_min,
                    "ratio_max": step.ratio_max,
                    "clipped_tokens": step.clipped_tokens,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    scheduler.step()
    ids = tuple(group_id for step in steps for group_id in step.group_ids)
    if len(ids) != contract.EFFECTIVE_GROUPS_PER_WINDOW or len(set(ids)) != len(ids):
        raise RuntimeError("K=1 invariant failed: a group was lost or reused.")
    return WindowTrainingResult(
        window_index=int(window_index),
        learning_rate=expected_lr,
        optimizer_updates=len(steps),
        group_ids=ids,
        steps=tuple(steps),
    )


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
            raise RuntimeError(f"Recovery expects {rows} rows but {path} is missing.")
        return
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) < rows:
        raise RuntimeError(
            f"Recovery expects {rows} rows but {path} contains {len(raw_lines)}."
        )
    for line_number, raw in enumerate(raw_lines[:rows], start=1):
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Corrupt JSONL row {line_number}: {path}") from exc
    if len(raw_lines) != rows or (
        path.stat().st_size and not raw_lines[-1].endswith(b"\n")
    ):
        temporary = path.with_name(f".{path.name}.truncate")
        temporary.write_bytes(b"".join(raw_lines[:rows]))
        os.replace(temporary, path)


def _group_log_rows(
    collection: WindowCollection, *, window_index: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prepared in collection.groups:
        rows.append(
            {
                "schema_version": 1,
                "window_index": int(window_index),
                "effective_epoch": int(window_index)
                // contract.WINDOWS_PER_EFFECTIVE_EPOCH,
                "group_id": prepared.group.group_id,
                "source_lines": list(prepared.group.source_lines),
                "positive_sids": list(prepared.group.positive_sids),
                "candidate_indices": [
                    candidate.candidate_index for candidate in prepared.candidates
                ],
                "candidate_sids": [
                    candidate.sid.render() for candidate in prepared.candidates
                ],
                "candidate_token_ids": [
                    list(candidate.token_ids) for candidate in prepared.candidates
                ],
                "rewards": prepared.reward.rewards.tolist(),
                "reward_tiers": list(prepared.reward.tiers),
                "advantages": prepared.reward.advantages.tolist(),
                "effective": prepared.reward.effective,
                "proposal_target_max_logp_difference": (
                    prepared.proposal_target_max_logp_difference
                ),
                "corrected_candidate_count": prepared.corrected_candidate_count,
                "target_forward_calls": prepared.target_forward_calls,
                "anchor_groups": 0,
                "gt_injection_count": 0,
                "k": contract.NUM_ITERATIONS,
            }
        )
    return rows


def _window_log_row(
    collection: WindowCollection,
    training: WindowTrainingResult,
    *,
    collection_seconds: float,
    training_seconds: float,
    peak_reserved_gib: float,
    collection_peak_reserved_gib: float,
    training_peak_reserved_gib: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "window_index": training.window_index,
        "effective_epoch": (
            training.window_index // contract.WINDOWS_PER_EFFECTIVE_EPOCH
        ),
        "learning_rate": training.learning_rate,
        "effective_groups": len(collection.groups),
        "rollouts": len(collection.groups) * contract.GROUP_SIZE,
        "raw_prompt_count": collection.raw_prompt_count,
        "filtered_group_count": collection.filtered_group_count,
        "overflow_group_count": collection.overflow_group_count,
        "group_ids": list(training.group_ids),
        "optimizer_updates": training.optimizer_updates,
        "optimizer_update_step": (
            (training.window_index + 1) * contract.OPTIMIZER_UPDATES_PER_WINDOW
        ),
        "steps": [asdict(step) for step in training.steps],
        "source_cursor": asdict(collection.source_cursor),
        "sampling_batches": [
            {
                **asdict(audit),
                "cache_stats": asdict(audit.cache_stats),
            }
            for audit in collection.audits
        ],
        "collection_seconds": float(collection_seconds),
        "training_seconds": float(training_seconds),
        "window_seconds": float(collection_seconds + training_seconds),
        "peak_reserved_gib": float(peak_reserved_gib),
        "collection_peak_reserved_gib": float(collection_peak_reserved_gib),
        "training_peak_reserved_gib": float(training_peak_reserved_gib),
        "anchor_groups": 0,
        "gt_injection_count": 0,
        "k": contract.NUM_ITERATIONS,
    }


def _write_run_contract(
    output_dir: Path,
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> None:
    signature_path = output_dir / "runtime_signature.json"
    config_path = output_dir / "resolved_config.json"
    if signature_path.exists() or config_path.exists():
        raise RuntimeError("A new run directory already contains contract files.")
    _atomic_json(signature_path, signature)
    _atomic_json(config_path, config)


def _read_run_contract(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Existing run directory is missing {label}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Existing run directory contains unreadable {label}."
        ) from exc


def _prepare_run_directory(
    output_dir: Path,
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    """Claim a new directory or validate an existing directory without rewriting it."""

    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path is not a directory: {output_dir}")
        is_non_empty = next(output_dir.iterdir(), None) is not None
        if is_non_empty:
            if not resume:
                raise FileExistsError(f"Non-empty output directory: {output_dir}")
            existing_signature = _read_run_contract(
                output_dir / "runtime_signature.json",
                label="runtime_signature.json",
            )
            existing_config = _read_run_contract(
                output_dir / "resolved_config.json",
                label="resolved_config.json",
            )
            if existing_signature != signature or existing_config != config:
                raise RuntimeError(
                    "Existing run contract differs from this invocation."
                )
            return
    else:
        output_dir.mkdir(parents=True)

    _write_run_contract(output_dir, config, signature)


def _ensure_epoch_adapters(
    bundle: PolicyModel,
    output: Path,
    *,
    completed_windows: int,
) -> None:
    completed_epochs = completed_windows // contract.WINDOWS_PER_EFFECTIVE_EPOCH
    for epoch_number in range(1, completed_epochs + 1):
        epoch_dir = output / f"epoch-{epoch_number:02d}-adapter"
        if epoch_dir.exists():
            continue
        boundary = epoch_number * contract.WINDOWS_PER_EFFECTIVE_EPOCH
        if completed_windows != boundary:
            raise RuntimeError(
                f"Missing epoch-{epoch_number:02d} adapter cannot be reconstructed "
                f"from window {completed_windows}."
            )
        save_policy_atomic(bundle, epoch_dir)


def _run_training_impl(
    config: dict[str, Any],
    runtime_signature: dict[str, Any],
    *,
    output_dir: Path,
    device: str | torch.device = "cuda:0",
    resume: bool = True,
    stop_after_windows: int | None = None,
    aligned_cache: bool = True,
) -> dict[str, Any]:
    """Run formal training or a complete-window pilot with exact recovery."""

    validate_runtime_signature(runtime_signature)
    output = Path(output_dir).expanduser().resolve()
    _prepare_run_directory(output, config, runtime_signature, resume=resume)
    configure_deterministic_runtime(int(config["train"]["seed"]))
    recovery_root = output / "recovery"
    recovery = (
        load_latest_recovery(recovery_root, runtime_signature) if resume else None
    )
    if recovery is None:
        cursor = RecoveryCursor(
            next_window_index=0,
            optimizer_update_step=0,
            source_cursor=SourceCursor(cycle_index=0, offset=0),
            window_log_rows=0,
            group_log_rows=0,
        )
        policy_path = None
        training_state = None
    else:
        checkpoint_path, cursor, training_state = recovery
        policy_path = checkpoint_path

    window_log = output / "windows.jsonl"
    group_log = output / "groups.jsonl"
    _truncate_jsonl(window_log, cursor.window_log_rows)
    _truncate_jsonl(group_log, cursor.group_log_rows)

    groups, trie = load_groups_and_trie(config)
    bundle = load_policy_model(config, device=device, policy_adapter_path=policy_path)
    _ensure_epoch_adapters(
        bundle,
        output,
        completed_windows=cursor.next_window_index,
    )
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    optimizer = build_optimizer(bundle, config)
    scheduler = WindowLRScheduler(optimizer, config)
    if training_state is not None:
        restore_training_state(optimizer, scheduler, training_state)
    if scheduler.completed_windows != cursor.next_window_index:
        raise RuntimeError("Recovery scheduler and cursor window differ.")
    stream = DeterministicGroupStream(
        groups,
        seed=int(config["train"]["seed"]),
        cursor=cursor.source_cursor,
    )

    total_windows = int(config["train"]["total_windows"])
    requested_stop = (
        total_windows
        if stop_after_windows is None
        else min(total_windows, cursor.next_window_index + int(stop_after_windows))
    )
    if requested_stop < cursor.next_window_index:
        raise ValueError("stop_after_windows must be non-negative.")
    last_checkpoint: Path | None = recovery[0] if recovery is not None else None
    windows_run = 0
    for window_index in range(cursor.next_window_index, requested_stop):
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        start = time.perf_counter()
        collection = collect_effective_window(
            bundle,
            stream,
            grammar,
            window_index=window_index,
            config=config,
            device=device,
            aligned_cache=aligned_cache,
        )
        collection_seconds = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "event": "collection_complete",
                    "window_index": window_index,
                    "raw_prompt_count": collection.raw_prompt_count,
                    "effective_groups": len(collection.groups),
                    "filtered_groups": collection.filtered_group_count,
                    "seconds": collection_seconds,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        collection_peak_reserved = (
            torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3
            if torch.cuda.is_available() and torch.device(device).type == "cuda"
            else 0.0
        )
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            # Cached rollout objects are no longer live after collection.  Return
            # their allocator blocks before the differently shaped autograd phase.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        start = time.perf_counter()
        training = train_complete_window(
            bundle,
            optimizer,
            scheduler,
            collection.groups,
            grammar,
            window_index=window_index,
            config=config,
            device=device,
        )
        training_seconds = time.perf_counter() - start
        training_peak_reserved = (
            torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3
            if torch.cuda.is_available() and torch.device(device).type == "cuda"
            else 0.0
        )
        peak_reserved = max(collection_peak_reserved, training_peak_reserved)
        if peak_reserved > float(config["memory"]["max_reserved_gib"]):
            raise RuntimeError(
                f"Window reserved memory {peak_reserved:.3f} GiB exceeds the limit."
            )
        _append_jsonl(group_log, _group_log_rows(collection, window_index=window_index))
        _append_jsonl(
            window_log,
            [
                _window_log_row(
                    collection,
                    training,
                    collection_seconds=collection_seconds,
                    training_seconds=training_seconds,
                    peak_reserved_gib=peak_reserved,
                    collection_peak_reserved_gib=collection_peak_reserved,
                    training_peak_reserved_gib=training_peak_reserved,
                )
            ],
        )
        next_window = window_index + 1
        cursor = RecoveryCursor(
            next_window_index=next_window,
            optimizer_update_step=(next_window * contract.OPTIMIZER_UPDATES_PER_WINDOW),
            source_cursor=collection.source_cursor,
            window_log_rows=next_window,
            group_log_rows=next_window * contract.EFFECTIVE_GROUPS_PER_WINDOW,
        )
        epoch_finished = next_window % contract.WINDOWS_PER_EFFECTIVE_EPOCH == 0
        should_checkpoint = (
            recovery_checkpoint_due(
                cursor,
                epoch_finished=epoch_finished,
                interval_updates=int(config["train"]["resume_save_updates"]),
            )
            or next_window == requested_stop
        )
        if should_checkpoint:
            checkpoint_path = recovery_root / (
                f"checkpoint-window-{cursor.next_window_index:06d}"
                f"-update-{cursor.optimizer_update_step:06d}"
            )
            if checkpoint_path.exists():
                raise FileExistsError(checkpoint_path)
            last_checkpoint = save_recovery_checkpoint(
                bundle,
                optimizer,
                scheduler,
                cursor,
                recovery_root,
                runtime_signature,
            )
        if epoch_finished:
            epoch_number = next_window // contract.WINDOWS_PER_EFFECTIVE_EPOCH
            epoch_dir = output / f"epoch-{epoch_number:02d}-adapter"
            if not epoch_dir.exists():
                save_policy_atomic(bundle, epoch_dir)
        print(
            json.dumps(
                {
                    "event": "window_complete",
                    "completed_windows": next_window,
                    "optimizer_update_step": cursor.optimizer_update_step,
                    "window_seconds": collection_seconds + training_seconds,
                    "peak_reserved_gib": peak_reserved,
                    "checkpoint_saved": bool(should_checkpoint),
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        windows_run += 1

    if cursor.next_window_index == total_windows:
        final_dir = output / "final-adapter"
        if not final_dir.exists():
            save_policy_atomic(bundle, final_dir)
    summary = {
        "schema_version": 1,
        "completed_windows": cursor.next_window_index,
        "optimizer_update_step": cursor.optimizer_update_step,
        "windows_run_this_invocation": windows_run,
        "source_cursor": asdict(cursor.source_cursor),
        "last_checkpoint": (
            last_checkpoint.relative_to(output).as_posix() if last_checkpoint else None
        ),
        "complete": cursor.next_window_index == total_windows,
        "anchor_groups": 0,
        "gt_injection_count": 0,
        "k": contract.NUM_ITERATIONS,
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
    stop_after_windows: int | None = None,
    aligned_cache: bool = True,
    dense_scoring: bool = False,
    formal: bool = False,
) -> dict[str, Any]:
    """Run formal training or a gate-only implementation baseline."""

    if formal:
        expected_output = Path(config["output"]["run_dir"]).expanduser().resolve()
        actual_output = Path(output_dir).expanduser().resolve()
        if actual_output != expected_output:
            raise ValueError(
                "Formal training output_dir must equal config.output.run_dir."
            )
        if stop_after_windows is not None:
            raise ValueError("Formal training cannot set stop_after_windows.")
        if dense_scoring:
            raise ValueError("Formal training cannot enable dense_scoring.")
        if not aligned_cache:
            raise ValueError("Formal training must enable aligned_cache.")
        if torch.device(device) != torch.device(contract.EXECUTION_DEVICE):
            raise ValueError(
                f"Formal training device must be {contract.EXECUTION_DEVICE}."
            )

    kwargs = {
        "output_dir": output_dir,
        "device": device,
        "resume": resume,
        "stop_after_windows": stop_after_windows,
        "aligned_cache": aligned_cache,
    }
    if dense_scoring:
        with dense_scoring_mode():
            return _run_training_impl(config, runtime_signature, **kwargs)
    return _run_training_impl(config, runtime_signature, **kwargs)


__all__ = [
    "OptimizerStepResult",
    "PreparedEffectiveGroup",
    "RawBatchAudit",
    "WindowCollection",
    "WindowLRScheduler",
    "WindowTrainingResult",
    "build_optimizer",
    "collect_effective_window",
    "configure_deterministic_runtime",
    "learning_rate_for_window",
    "load_groups_and_trie",
    "partition_effective_groups",
    "run_training",
    "set_global_seed",
    "train_complete_window",
    "train_minibatch",
]
