"""Windowed online training and calibration for the approved RLOO run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ksllm4rec_grpo.constraint import RecommendationGrammar
from ksllm4rec_grpo.data import RecommendationGroup, iter_groups
from ksllm4rec_grpo.prompt import encode_prompt
from ksllm4rec_grpo.trie import SidPrefixTrie
from ksllm4rec_orpo.data import Sid

from . import contract
from .checkpoint import (
    RecoveryCursor,
    load_latest_recovery,
    recovery_checkpoint_due,
    restore_training_state,
    save_policy_atomic,
    save_recovery_checkpoint,
)
from .integrity import canonical_sha256, file_record
from .modeling import PolicyModel, load_policy_model
from .objective import (
    ObjectiveBranch,
    RewardOutput,
    calibrate_anchor_lambda,
    gt_sequence_logps,
    gt_set_anchor_loss,
    rewards_and_advantages,
    rloo_chunk_loss,
)
from .rollout import RolloutCandidate, rollout_group
from .scoring import score_completions, score_completions_chunked


@dataclass(frozen=True)
class PreparedGroup:
    """One live rollout retained until its source window is optimized."""

    group: RecommendationGroup
    prompt_ids: tuple[int, ...]
    positives: tuple[Sid, ...]
    candidates: tuple[RolloutCandidate, ...]
    reward: RewardOutput


@dataclass(frozen=True)
class BackwardResult:
    loss: float
    max_logp_difference: float
    decision_tokens: int
    per_candidate_losses: tuple[float, ...]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_for_window(config: dict[str, Any], window_index: int) -> float:
    """Return the frozen LR for zero-based source window ``w``."""

    train = config["train"]
    total = int(train["total_windows"])
    warmup = int(train["warmup_windows"])
    maximum = float(train["learning_rate"])
    if isinstance(window_index, bool) or not 0 <= int(window_index) < total:
        raise ValueError(f"window_index must be in [0, {total - 1}].")
    w = int(window_index)
    if w < warmup:
        return maximum * (w + 1) / warmup
    denominator = (total - 1) - warmup
    return maximum * 0.5 * (
        1.0 + math.cos(math.pi * (w - warmup) / denominator)
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _trainable_parameters(bundle: PolicyModel) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in bundle.model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Policy has no trainable LoRA parameters.")
    return parameters


def build_optimizer(bundle: PolicyModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    train = config["train"]
    kwargs: dict[str, Any] = {
        "lr": float(train["learning_rate"]),
        "betas": (float(train["adam_beta1"]), float(train["adam_beta2"])),
        "eps": float(train["adam_epsilon"]),
        "weight_decay": float(train["weight_decay"]),
    }
    if next(bundle.model.parameters()).is_cuda:
        kwargs["fused"] = True
    return torch.optim.AdamW(_trainable_parameters(bundle), **kwargs)


def gradient_l2_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    """Compute the global pre-clipping L2 norm with FP32 gradient values."""

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
        raise FloatingPointError("Global gradient norm is NaN or Inf.")
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Corrupt JSONL row {line_number}: {path}") from exc
    return rows


def _truncate_jsonl(path: Path, keep: int) -> list[dict[str, Any]]:
    if keep < 0:
        raise ValueError("JSONL keep count must be non-negative.")
    if not path.exists():
        if keep:
            raise RuntimeError(f"Recovery expects {keep} rows but {path} is missing.")
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) < keep:
        raise RuntimeError(f"Recovery expects {keep} rows but found {len(lines)}: {path}")
    for line_number, line in enumerate(lines[:keep], start=1):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Corrupt committed JSONL row {line_number}: {path}"
            ) from exc
    if len(lines) != keep or (raw and not raw.endswith(b"\n")):
        temporary = path.with_name(f".{path.name}.truncate")
        temporary.write_bytes(b"".join(lines[:keep]))
        os.replace(temporary, path)
    return _load_jsonl(path)


def calibration_rank(group_id: str) -> str:
    payload = f"anchor-calibration|42|{group_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_calibration_ids(
    groups: Sequence[RecommendationGroup], count: int = contract.CALIBRATION_GROUPS
) -> list[str]:
    if len(groups) < count:
        raise ValueError("Not enough recommendation groups for calibration.")
    ranked = sorted((calibration_rank(group.group_id), group.group_id) for group in groups)
    return [group_id for _, group_id in ranked[:count]]


def write_calibration_ids(
    groups_path: Path,
    output_path: Path,
    *,
    count: int = contract.CALIBRATION_GROUPS,
) -> dict[str, Any]:
    groups = list(iter_groups(groups_path))
    ids = expected_calibration_ids(groups, count)
    value = {
        "schema_version": 1,
        "selection": "first_after_sha256(anchor-calibration|42|group_id)",
        "count": count,
        "groups": file_record(groups_path),
        "group_ids": ids,
        "group_ids_sha256": canonical_sha256(ids),
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError("Existing calibration IDs differ from the frozen rule.")
    else:
        _atomic_json(output_path, value)
    return value


def load_calibration_ids(
    path: Path, groups: Sequence[RecommendationGroup]
) -> tuple[str, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "selection",
        "count",
        "groups",
        "group_ids",
        "group_ids_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Calibration ID file has an invalid schema.")
    expected = expected_calibration_ids(groups, contract.CALIBRATION_GROUPS)
    if (
        value["schema_version"] != 1
        or value["selection"]
        != "first_after_sha256(anchor-calibration|42|group_id)"
        or value["count"] != contract.CALIBRATION_GROUPS
        or value["group_ids"] != expected
        or value["group_ids_sha256"] != canonical_sha256(expected)
    ):
        raise RuntimeError("Calibration IDs differ from the frozen selection rule.")
    return tuple(expected)


def _load_groups_and_trie(
    config: dict[str, Any], groups_path: Path, trie_dir: Path
) -> tuple[list[RecommendationGroup], SidPrefixTrie]:
    groups = list(iter_groups(groups_path))
    if len(groups) != int(config["data"]["groups"]):
        raise RuntimeError(f"Expected {config['data']['groups']} groups, got {len(groups)}.")
    if file_record(groups_path, contract.GROUPS_SHA256)["sha256"] != contract.GROUPS_SHA256:
        raise AssertionError("unreachable")
    trie = SidPrefixTrie.load(
        trie_dir, expected_leaf_count=int(config["trie"]["unique_sids"])
    )
    manifest = file_record(trie_dir / "manifest.json", contract.TRIE_MANIFEST_SHA256)
    if manifest["sha256"] != contract.TRIE_MANIFEST_SHA256:
        raise AssertionError("unreachable")
    return groups, trie


def prepare_group(
    bundle: PolicyModel,
    group: RecommendationGroup,
    grammar: RecommendationGrammar,
    *,
    epoch_index: int,
    config: dict[str, Any],
    device: str | torch.device,
) -> PreparedGroup:
    prompt_ids = tuple(
        encode_prompt(
            bundle.tokenizer,
            group.system,
            group.prompt,
            cutoff_len=int(config["data"]["cutoff_len"]),
        )
    )
    candidates = rollout_group(
        bundle.model,
        prompt_ids,
        grammar,
        group_id=group.group_id,
        epoch_index=epoch_index,
        chunk_size=int(config["rollout"]["candidate_chunk_size"]),
        max_completion_length=int(config["rollout"]["max_completion_length"]),
        device=device,
        temperature=float(config["rollout"]["temperature"]),
    )
    positives = tuple(Sid.parse(value) for value in group.positive_sids)
    reward = rewards_and_advantages(
        [candidate.sid for candidate in candidates], positives
    )
    return PreparedGroup(
        group=group,
        prompt_ids=prompt_ids,
        positives=positives,
        candidates=candidates,
        reward=reward,
    )


def _candidate_old_tensors(
    candidates: Sequence[RolloutCandidate], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(candidate.old_log_probs) for candidate in candidates)
    old = torch.zeros((len(candidates), maximum), dtype=torch.float32, device=device)
    decisions = torch.zeros((len(candidates), maximum), dtype=torch.bool, device=device)
    for index, candidate in enumerate(candidates):
        length = len(candidate.old_log_probs)
        old[index, :length] = torch.tensor(
            candidate.old_log_probs, dtype=torch.float32, device=device
        )
        decisions[index, :length] = torch.tensor(
            candidate.decision_mask, dtype=torch.bool, device=device
        )
    return old, decisions


def backward_rloo_group(
    bundle: PolicyModel,
    prepared: PreparedGroup,
    grammar: RecommendationGrammar,
    *,
    scale: float,
    device: str | torch.device,
    max_logp_difference: float,
) -> BackwardResult:
    if prepared.reward.branch != ObjectiveBranch.RLOO:
        raise ValueError("backward_rloo_group requires the RLOO branch.")
    torch_device = torch.device(device)
    total_decisions = sum(
        sum(candidate.decision_mask) for candidate in prepared.candidates
    )
    if total_decisions <= 0:
        raise RuntimeError("RLOO group has no decision token.")
    advantages = prepared.reward.advantages.to(torch_device)
    losses: list[float] = []
    per_candidate: list[float] = []
    maximum_delta = 0.0
    completions = [candidate.token_ids for candidate in prepared.candidates]
    for start in (0, 8):
        stop = start + 8
        scores = score_completions(
            bundle.model,
            prepared.prompt_ids,
            completions[start:stop],
            grammar,
            device=torch_device,
            temperature=1.0,
        )
        old, expected_mask = _candidate_old_tensors(
            prepared.candidates[start:stop], torch_device
        )
        width = scores.log_probs.shape[1]
        old = old[:, :width]
        expected_mask = expected_mask[:, :width]
        if not torch.equal(scores.decision_mask, expected_mask):
            raise RuntimeError("Sampling and replay decision masks differ.")
        delta = (scores.log_probs - old).abs()[scores.decision_mask]
        chunk_delta = float(delta.max().detach().item()) if delta.numel() else 0.0
        maximum_delta = max(maximum_delta, chunk_delta)
        if chunk_delta > max_logp_difference:
            raise RuntimeError(
                "Sampling/replay probability gate failed before backward: "
                f"max_abs_delta={chunk_delta:.9g}."
            )
        output = rloo_chunk_loss(
            scores.log_probs,
            advantages[start:stop],
            scores.decision_mask,
            total_group_decisions=total_decisions,
        )
        (output.loss * float(scale)).backward()
        losses.append(float(output.loss.detach().item()))
        per_candidate.extend(
            float(value) for value in output.per_candidate_loss.detach().cpu().tolist()
        )
        del scores, output
    return BackwardResult(
        loss=sum(losses),
        max_logp_difference=maximum_delta,
        decision_tokens=total_decisions,
        per_candidate_losses=tuple(per_candidate),
    )


def _anchor_no_grad_scores(
    bundle: PolicyModel,
    prepared: PreparedGroup,
    grammar: RecommendationGrammar,
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    list[tuple[int, int]],
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    completions = [grammar.encode_sid(sid) for sid in prepared.positives]
    values: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []
    replay_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    cursor = 0
    with torch.no_grad():
        for scores in score_completions_chunked(
            bundle.model,
            prepared.prompt_ids,
            completions,
            grammar,
            chunk_size=8,
            device=device,
            temperature=1.0,
        ):
            sequence, _ = gt_sequence_logps(scores.log_probs, scores.decision_mask)
            values.append(sequence.detach())
            replay_rows.append(
                (scores.log_probs.detach(), scores.decision_mask.detach())
            )
            spans.append((cursor, cursor + sequence.numel()))
            cursor += sequence.numel()
    return torch.cat(values), spans, replay_rows


def backward_anchor_group(
    bundle: PolicyModel,
    prepared: PreparedGroup,
    grammar: RecommendationGrammar,
    *,
    scale: float,
    device: str | torch.device,
    max_logp_difference: float,
) -> BackwardResult:
    if prepared.reward.branch != ObjectiveBranch.GT_SET_ANCHOR:
        raise ValueError("backward_anchor_group requires the anchor branch.")
    torch_device = torch.device(device)
    completions = [grammar.encode_sid(sid) for sid in prepared.positives]
    detached_sequences, spans, first_pass_rows = _anchor_no_grad_scores(
        bundle, prepared, grammar, device=torch_device
    )
    raw_loss = gt_set_anchor_loss(detached_sequences)
    weights = detached_sequences.softmax(dim=0)
    maximum_delta = 0.0
    total_decisions = 0
    for chunk_index, (start, stop) in enumerate(spans):
        # Build and backpropagate one GT chunk at a time.  Calling the chunked
        # helper here would materialize its whole tuple before this loop and
        # retain up to three 16k-context autograd graphs simultaneously.
        scores = score_completions(
            bundle.model,
            prepared.prompt_ids,
            completions[start:stop],
            grammar,
            device=torch_device,
            temperature=1.0,
        )
        sequence, counts = gt_sequence_logps(scores.log_probs, scores.decision_mask)
        first_logps, first_mask = first_pass_rows[chunk_index]
        if not torch.equal(scores.decision_mask, first_mask):
            raise RuntimeError("Anchor no-grad/grad decision masks differ.")
        replay_delta = (scores.log_probs.detach() - first_logps).abs()[first_mask]
        if replay_delta.numel():
            maximum_delta = max(maximum_delta, float(replay_delta.max().item()))
        if maximum_delta > max_logp_difference:
            raise RuntimeError(
                "Anchor no-grad/grad probability gate failed before backward: "
                f"max_abs_delta={maximum_delta:.9g}."
            )
        total_decisions += int(counts.sum().item())
        surrogate = -(weights[start:stop] * sequence).sum() * float(scale)
        if surrogate.requires_grad:
            surrogate.backward()
        del scores, sequence, surrogate
    return BackwardResult(
        loss=float(raw_loss.item()),
        max_logp_difference=maximum_delta,
        decision_tokens=total_decisions,
        per_candidate_losses=tuple(),
    )


def _group_audit(
    prepared: PreparedGroup,
    *,
    epoch_index: int,
    window_step: int,
    optimizer_update_step: int,
    backward: BackwardResult | None,
) -> dict[str, Any]:
    rewards = prepared.reward.rewards.tolist()
    exact_count = sum(tier == "exact" for tier in prepared.reward.tiers)
    return {
        "schema_version": 1,
        "epoch": epoch_index,
        "window_step": window_step,
        "optimizer_update_step": optimizer_update_step,
        "group_id": prepared.group.group_id,
        "source_lines": list(prepared.group.source_lines),
        "positive_sids": list(prepared.group.positive_sids),
        "candidate_sids": [candidate.sid.render() for candidate in prepared.candidates],
        "candidate_indices": [candidate.candidate_index for candidate in prepared.candidates],
        "rewards": rewards,
        "reward_tiers": list(prepared.reward.tiers),
        "reward_mean_live": float(prepared.reward.mean_reward.item()),
        "reward_mean_final": float(prepared.reward.mean_reward.item()),
        "candidate_exact_rate_live": exact_count / contract.GROUP_SIZE,
        "candidate_exact_rate_final": exact_count / contract.GROUP_SIZE,
        "gt_injection_count": 0,
        "advantages": prepared.reward.advantages.tolist(),
        "divide_by_std": False,
        "branch": prepared.reward.branch.value,
        "loss": backward.loss if backward is not None else None,
        "decision_tokens": backward.decision_tokens if backward is not None else 0,
        "max_sample_replay_logp_difference": (
            backward.max_logp_difference if backward is not None else 0.0
        ),
        "per_candidate_losses": (
            list(backward.per_candidate_losses) if backward is not None else []
        ),
    }


def _rollout_groups(
    bundle: PolicyModel,
    groups: Sequence[RecommendationGroup],
    grammar: RecommendationGrammar,
    *,
    epoch_index: int,
    config: dict[str, Any],
    device: str | torch.device,
) -> list[PreparedGroup]:
    return [
        prepare_group(
            bundle,
            group,
            grammar,
            epoch_index=epoch_index,
            config=config,
            device=device,
        )
        for group in groups
    ]


def _branch_counts(prepared: Sequence[PreparedGroup]) -> Counter[str]:
    return Counter(item.reward.branch.value for item in prepared)


def run_calibration(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    runtime_signature: dict[str, Any],
    output_path: Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Resolve lambda0 from the frozen 512-group initial-policy sample."""

    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("gate") == "calibration":
            # Import lazily to avoid a module cycle during normal startup.
            from .gates import validate_calibration_report

            validate_calibration_report(config, existing, runtime_signature)
            return existing
        # A process may have stopped after the raw result was atomically
        # written but before the gate wrapper bound/validated it.  Recompute
        # instead of trusting that incomplete cache.
        output_path.unlink()
    groups, trie = _load_groups_and_trie(config, groups_path, trie_dir)
    selected_ids = load_calibration_ids(calibration_ids_path, groups)
    by_id = {group.group_id: group for group in groups}
    selected = [by_id[group_id] for group_id in selected_ids]
    set_global_seed(int(config["train"]["seed"]))
    bundle = load_policy_model(config, device=device)
    bundle.model.train()
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    started = time.monotonic()
    prepared = _rollout_groups(
        bundle,
        selected,
        grammar,
        epoch_index=0,
        config=config,
        device=device,
    )
    counts = _branch_counts(prepared)
    rloo_groups = [item for item in prepared if item.reward.branch == ObjectiveBranch.RLOO]
    anchor_groups = [
        item for item in prepared if item.reward.branch == ObjectiveBranch.GT_SET_ANCHOR
    ]
    if not rloo_groups or not anchor_groups:
        raise RuntimeError(
            f"Calibration requires both branches, got {dict(counts)}."
        )
    parameters = _trainable_parameters(bundle)
    bundle.model.zero_grad(set_to_none=True)
    maximum_delta = 0.0
    for item in rloo_groups:
        result = backward_rloo_group(
            bundle,
            item,
            grammar,
            scale=1.0 / len(rloo_groups),
            device=device,
            max_logp_difference=float(config["gates"]["max_logp_difference"]),
        )
        maximum_delta = max(maximum_delta, result.max_logp_difference)
    rloo_norm = gradient_l2_norm(parameters)
    bundle.model.zero_grad(set_to_none=True)
    anchor_losses: list[float] = []
    for item in anchor_groups:
        result = backward_anchor_group(
            bundle,
            item,
            grammar,
            scale=1.0 / len(anchor_groups),
            device=device,
            max_logp_difference=float(config["gates"]["max_logp_difference"]),
        )
        anchor_losses.append(result.loss)
        maximum_delta = max(maximum_delta, result.max_logp_difference)
    anchor_norm = gradient_l2_norm(parameters)
    bundle.model.zero_grad(set_to_none=True)
    lambda0 = calibrate_anchor_lambda(rloo_norm, anchor_norm)
    resolved_contract = {
        "schema_version": 1,
        "anchor": {
            "lambda0": lambda0,
            "group_ids_sha256": canonical_sha256(list(selected_ids)),
            "groups": len(selected_ids),
            "rloo_groups": len(rloo_groups),
            "anchor_groups": len(anchor_groups),
            "skip_groups": counts[ObjectiveBranch.SKIP.value],
            "rloo_gradient_l2_fp32": rloo_norm,
            "anchor_gradient_l2_fp32": anchor_norm,
            "target_gradient_ratio": float(config["anchor"]["target_gradient_ratio"]),
            "max_weight": float(config["anchor"]["max_weight"]),
        },
    }
    tier_counts = Counter(
        tier for item in prepared for tier in item.reward.tiers
    )
    report = {
        "schema_version": 1,
        "runtime_signature": runtime_signature,
        "resolved_contract": resolved_contract,
        "resolved_contract_sha256": canonical_sha256(resolved_contract),
        "lambda0": lambda0,
        "groups_run": len(prepared),
        "candidates_run": len(prepared) * contract.GROUP_SIZE,
        "all_legal": True,
        "all_finite": True,
        "rloo_groups": len(rloo_groups),
        "anchor_groups": len(anchor_groups),
        "skip_groups": counts[ObjectiveBranch.SKIP.value],
        "rloo_group_rate": len(rloo_groups) / len(prepared),
        "rloo_grad_norm": rloo_norm,
        "anchor_grad_norm": anchor_norm,
        "optimizer_updates": 0,
        "parameter_max_change": 0.0,
        "reward_mean": fmean(float(item.reward.mean_reward) for item in prepared),
        "reward_tier_counts": dict(tier_counts),
        "max_sample_replay_logp_difference": maximum_delta,
        "anchor_loss_mean": fmean(anchor_losses),
        "seconds": time.monotonic() - started,
    }
    minimum_rate = float(config["gates"]["min_rloo_group_rate"])
    if report["rloo_group_rate"] < minimum_rate:
        raise RuntimeError(f"Calibration RLOO rate gate failed: {report}")
    _atomic_json(output_path, report)
    return report


def _validate_cursor(
    cursor: RecoveryCursor, *, groups_per_epoch: int, epochs: int
) -> None:
    if cursor.rollout_chunk != 8 or cursor.loss_chunk != 8:
        raise RuntimeError("Recovery violates the fixed 8+8 chunk contract.")
    if not 1 <= cursor.epoch_index <= epochs + 1:
        raise RuntimeError("Recovery epoch is outside the run.")
    if cursor.epoch_index == epochs + 1:
        expected_groups = groups_per_epoch * epochs
        expected_windows = int(math.ceil(groups_per_epoch / 8)) * epochs
        if cursor.next_group_offset != 0:
            raise RuntimeError("Completed recovery offset must be zero.")
    else:
        if cursor.next_group_offset % 8 != 0 or not 0 <= cursor.next_group_offset <= groups_per_epoch:
            raise RuntimeError("Recovery offset is not a complete source window.")
        expected_groups = (cursor.epoch_index - 1) * groups_per_epoch + cursor.next_group_offset
        expected_windows = (cursor.epoch_index - 1) * (groups_per_epoch // 8) + cursor.next_group_offset // 8
    if cursor.groups_completed != expected_groups or cursor.window_step != expected_windows:
        raise RuntimeError("Recovery cursor counters are inconsistent.")


def _reconcile_outputs(
    output_dir: Path,
    cursor: RecoveryCursor,
    *,
    groups_per_epoch: int,
    epochs: int,
) -> None:
    _truncate_jsonl(output_dir / "train_audit.jsonl", cursor.groups_completed)
    _truncate_jsonl(output_dir / "train_progress.jsonl", cursor.window_step)
    for epoch in range(cursor.epoch_index, epochs + 1):
        path = output_dir / f"epoch_{epoch:03d}"
        if path.exists():
            shutil.rmtree(path)
    if cursor.epoch_index <= epochs:
        summary = output_dir / "run_summary.json"
        if summary.exists():
            summary.unlink()


def _has_training_artifacts(output_dir: Path) -> bool:
    return any(
        (output_dir / name).exists()
        for name in (
            "train_audit.jsonl",
            "train_progress.jsonl",
            "run_summary.json",
            "recovery",
        )
    ) or any(output_dir.glob("epoch_[0-9][0-9][0-9]"))


def _summarize_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    branches = Counter(row["branch"] for row in rows)
    tiers = Counter(tier for row in rows for tier in row["reward_tiers"])
    by_epoch: dict[str, dict[str, Any]] = {}
    for epoch in sorted({int(row["epoch"]) for row in rows}):
        selected = [row for row in rows if int(row["epoch"]) == epoch]
        epoch_tiers = Counter(tier for row in selected for tier in row["reward_tiers"])
        by_epoch[str(epoch)] = {
            "groups": len(selected),
            "reward_mean": fmean(float(row["reward_mean_live"]) for row in selected),
            "branch_counts": dict(Counter(row["branch"] for row in selected)),
            "reward_tier_counts": dict(epoch_tiers),
            "candidate_exact_rate": epoch_tiers["exact"] / (len(selected) * 16),
            "gt_injection_count": sum(int(row["gt_injection_count"]) for row in selected),
        }
    return {
        "groups": len(rows),
        "reward_mean": (
            fmean(float(row["reward_mean_live"]) for row in rows) if rows else None
        ),
        "branch_counts": dict(branches),
        "reward_tier_counts": dict(tiers),
        "candidate_exact_rate": tiers["exact"] / (len(rows) * 16) if rows else None,
        "gt_injection_count": sum(int(row["gt_injection_count"]) for row in rows),
        "by_epoch": by_epoch,
    }


def run_training(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    output_dir: Path,
    resolved_contract: dict[str, Any],
    runtime_signature: dict[str, Any],
    max_groups: int | None = None,
    resume: bool = True,
    save_recovery: bool = True,
    save_epochs: bool = True,
    device: str = "cuda:0",
    gate_mode: str | None = None,
) -> dict[str, Any]:
    """Train in immutable eight-source-group windows with online G=16 rollout."""

    invocation_started = time.monotonic()
    groups, trie = _load_groups_and_trie(config, groups_path, trie_dir)
    epochs = int(config["train"]["epochs"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume and _has_training_artifacts(output_dir):
        raise RuntimeError(
            "resume=False requires an output directory with no training artifacts."
        )
    lambda0 = float(resolved_contract["anchor"]["lambda0"])
    recovery = None
    if resume:
        recovery = load_latest_recovery(
            output_dir / "recovery", runtime_signature, resolved_contract
        )
    if recovery is None:
        if resume and any(
            (output_dir / name).exists()
            for name in ("train_audit.jsonl", "train_progress.jsonl", "run_summary.json")
        ):
            raise RuntimeError("Training outputs exist without a recovery checkpoint.")
        cursor = RecoveryCursor(
            epoch_index=1,
            next_group_offset=0,
            window_step=0,
            optimizer_update_step=0,
            groups_completed=0,
            rollout_chunk=8,
            loss_chunk=8,
            lambda0=lambda0,
        )
        policy_path = None
        training_state = None
    else:
        checkpoint_path, cursor, training_state = recovery
        policy_path = checkpoint_path
        if training_state["resolved_contract"] != resolved_contract:
            raise RuntimeError("Recovery resolved contract differs.")
    _validate_cursor(cursor, groups_per_epoch=len(groups), epochs=epochs)
    _reconcile_outputs(
        output_dir, cursor, groups_per_epoch=len(groups), epochs=epochs
    )
    if cursor.epoch_index == epochs + 1:
        summary_path = output_dir / "run_summary.json"
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                existing.get("runtime_signature") != runtime_signature
                or existing.get("resolved_contract") != resolved_contract
                or existing.get("state") != asdict(cursor)
                or existing.get("completed") is not True
            ):
                raise RuntimeError("Completed run summary differs from recovery state.")
            return existing

    set_global_seed(int(config["train"]["seed"]))
    bundle = load_policy_model(
        config, device=device, policy_adapter_path=policy_path
    )
    bundle.model.train()
    optimizer = build_optimizer(bundle, config)
    if training_state is not None:
        restore_training_state(optimizer, None, training_state)
    parameters = _trainable_parameters(bundle)
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    groups_this_invocation = 0
    gradient_norm_max = 0.0
    parameter_change_max = 0.0
    initial_parameters = (
        [parameter.detach().clone() for parameter in parameters]
        if gate_mode is not None
        else []
    )

    epoch_index = cursor.epoch_index
    next_offset = cursor.next_group_offset
    window_step = cursor.window_step
    optimizer_update_step = cursor.optimizer_update_step
    initial_optimizer_update_step = optimizer_update_step
    groups_completed = cursor.groups_completed
    stop_requested = False
    while epoch_index <= epochs and not stop_requested:
        offset = next_offset
        while offset < len(groups):
            if max_groups is not None and groups_this_invocation >= max_groups:
                stop_requested = True
                break
            window = groups[offset : offset + 8]
            if len(window) != 8:
                raise RuntimeError("The frozen dataset must divide into 8-group windows.")
            if max_groups is not None and groups_this_invocation + 8 > max_groups:
                stop_requested = True
                break
            lr = learning_rate_for_window(config, window_step)
            _set_optimizer_lr(optimizer, lr)
            prepared = _rollout_groups(
                bundle,
                window,
                grammar,
                epoch_index=epoch_index,
                config=config,
                device=device,
            )
            counts = _branch_counts(prepared)
            rloo_count = counts[ObjectiveBranch.RLOO.value]
            anchor_count = counts[ObjectiveBranch.GT_SET_ANCHOR.value]
            optimizer.zero_grad(set_to_none=True)
            backward_by_group: dict[str, BackwardResult] = {}
            maximum_delta = 0.0
            for item in prepared:
                if item.reward.branch == ObjectiveBranch.RLOO:
                    result = backward_rloo_group(
                        bundle,
                        item,
                        grammar,
                        scale=1.0 / rloo_count,
                        device=device,
                        max_logp_difference=float(
                            config["gates"]["max_logp_difference"]
                        ),
                    )
                    backward_by_group[item.group.group_id] = result
                    maximum_delta = max(maximum_delta, result.max_logp_difference)
            for item in prepared:
                if item.reward.branch == ObjectiveBranch.GT_SET_ANCHOR:
                    result = backward_anchor_group(
                        bundle,
                        item,
                        grammar,
                        scale=lambda0 / anchor_count,
                        device=device,
                        max_logp_difference=float(
                            config["gates"]["max_logp_difference"]
                        ),
                    )
                    backward_by_group[item.group.group_id] = result
                    maximum_delta = max(
                        maximum_delta, result.max_logp_difference
                    )
            updated = bool(rloo_count or anchor_count)
            gradient_norm = gradient_l2_norm(parameters) if updated else 0.0
            gradient_norm_max = max(gradient_norm_max, gradient_norm)
            if updated:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(config["train"]["max_grad_norm"])
                )
                optimizer.step()
                optimizer_update_step += 1
            optimizer.zero_grad(set_to_none=True)

            offset += 8
            groups_this_invocation += 8
            groups_completed += 8
            window_step += 1
            epoch_finished = offset == len(groups)
            next_epoch = epoch_index + 1 if epoch_finished else epoch_index
            next_group_offset = 0 if epoch_finished else offset
            cursor = RecoveryCursor(
                epoch_index=next_epoch,
                next_group_offset=next_group_offset,
                window_step=window_step,
                optimizer_update_step=optimizer_update_step,
                groups_completed=groups_completed,
                rollout_chunk=8,
                loss_chunk=8,
                lambda0=lambda0,
            )
            audit_rows = [
                _group_audit(
                    item,
                    epoch_index=epoch_index,
                    window_step=window_step,
                    optimizer_update_step=optimizer_update_step,
                    backward=backward_by_group.get(item.group.group_id),
                )
                for item in prepared
            ]
            progress = {
                "schema_version": 1,
                "epoch": epoch_index,
                "window_step": window_step,
                "optimizer_update_step": optimizer_update_step,
                "groups_completed": groups_completed,
                "group_offset_after": next_group_offset,
                "learning_rate": lr,
                "optimizer_updated": updated,
                "branch_counts": dict(counts),
                "reward_mean": fmean(
                    float(item.reward.mean_reward) for item in prepared
                ),
                "gradient_l2_pre_clip_fp32": gradient_norm,
                "max_sample_replay_logp_difference": maximum_delta,
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
            _append_jsonl(output_dir / "train_audit.jsonl", audit_rows)
            _append_jsonl(output_dir / "train_progress.jsonl", [progress])
            if save_epochs and epoch_finished:
                save_policy_atomic(bundle, output_dir / f"epoch_{epoch_index:03d}")
            if save_recovery and recovery_checkpoint_due(
                cursor,
                optimizer_stepped=updated,
                epoch_finished=epoch_finished,
                interval=int(config["train"]["resume_save_updates"]),
            ):
                save_recovery_checkpoint(
                    bundle,
                    optimizer,
                    None,
                    cursor,
                    output_dir / "recovery",
                    runtime_signature,
                    resolved_contract,
                )
            if epoch_finished:
                epoch_index += 1
                next_offset = 0
                break

    for initial, parameter in zip(initial_parameters, parameters):
        if initial.shape == parameter.shape:
            parameter_change_max = max(
                parameter_change_max,
                float((parameter.detach() - initial).abs().max().item()),
            )
    if gate_mode == "timing":
        # Include one real adapter+Adam+RNG recovery write.  One save per 256
        # measured groups is more frequent than the formal every-100-update
        # cadence, so the frozen 1.2x projection remains conservative for I/O.
        save_recovery_checkpoint(
            bundle,
            optimizer,
            None,
            cursor,
            output_dir / "timing_recovery",
            runtime_signature,
            resolved_contract,
        )
    audit = _load_jsonl(output_dir / "train_audit.jsonl")
    aggregate = _summarize_audit(audit)
    completed = cursor.epoch_index == epochs + 1
    elapsed = time.monotonic() - invocation_started
    branch_counts = Counter(row["branch"] for row in audit)
    losses = [float(row["loss"]) for row in audit if row["loss"] is not None]
    all_finite = all(
        math.isfinite(float(row["reward_mean_live"]))
        and (row["loss"] is None or math.isfinite(float(row["loss"])))
        and all(math.isfinite(float(value)) for value in row["rewards"])
        and all(math.isfinite(float(value)) for value in row["advantages"])
        for row in audit
    )
    formal_history = gate_mode is None
    reported_groups = len(audit) if formal_history else groups_this_invocation
    reported_updates = (
        optimizer_update_step
        if formal_history
        else optimizer_update_step - initial_optimizer_update_step
    )
    reported_gradient_max = (
        max(
            (
                float(row.get("gradient_l2_pre_clip_fp32", 0.0))
                for row in _load_jsonl(output_dir / "train_progress.jsonl")
            ),
            default=0.0,
        )
        if formal_history
        else gradient_norm_max
    )
    summary = {
        "schema_version": 1,
        "runtime_signature": runtime_signature,
        "resolved_contract": resolved_contract,
        "gate_mode": gate_mode,
        "completed": completed,
        "state": asdict(cursor),
        "groups_run_this_invocation": groups_this_invocation,
        "groups_in_audit": len(audit),
        "train_seconds_this_invocation": elapsed,
        "gradient_norm_max_this_invocation": gradient_norm_max,
        "parameter_max_change_this_invocation": parameter_change_max,
        "metrics": aggregate,
        "groups_run": reported_groups,
        "candidates_run": reported_groups * contract.GROUP_SIZE,
        "all_legal": all(
            len(row["candidate_sids"]) == contract.GROUP_SIZE for row in audit
        ),
        "all_finite": all_finite,
        "rloo_groups": branch_counts[ObjectiveBranch.RLOO.value],
        "anchor_groups": branch_counts[ObjectiveBranch.GT_SET_ANCHOR.value],
        "skip_groups": branch_counts[ObjectiveBranch.SKIP.value],
        "rloo_group_rate": (
            branch_counts[ObjectiveBranch.RLOO.value] / len(audit) if audit else 0.0
        ),
        "rollout_chunk": 8,
        "loss_chunk": 8,
        "gt_chunk": 8,
        "optimizer_updates": reported_updates,
        "gradient_norm_max": reported_gradient_max,
        "loss_mean": fmean(losses) if losses else 0.0,
    }
    if gate_mode is not None:
        summary["parameter_max_change"] = parameter_change_max
    if gate_mode == "timing":
        projected = 1.2 * (elapsed / groups_this_invocation * 34_032) / 3600.0
        summary.update(
            {
                "end_to_end": True,
                "seconds": elapsed,
                "projected_two_epoch_hours": projected,
                "checkpoint_io_included": True,
                "checkpoint_io_measurement": "one_full_recovery_save",
            }
        )
    if completed and max_groups is None:
        _atomic_json(output_dir / "run_summary.json", summary)
    return summary


def run_probability_check(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    resolved_contract: dict[str, Any],
    runtime_signature: dict[str, Any],
    output_path: Path,
    groups_to_check: int | None = None,
    max_groups: int | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Replay live samples before any update and enforce the 1e-5 hard gate."""

    if groups_to_check is not None and max_groups is not None:
        raise ValueError("Pass only one probability group-count argument.")
    count = int(
        max_groups if max_groups is not None else groups_to_check or 1
    )
    groups, trie = _load_groups_and_trie(config, groups_path, trie_dir)
    bundle = load_policy_model(config, device=device)
    bundle.model.train()
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    checked = _rollout_groups(
        bundle,
        groups[:count],
        grammar,
        epoch_index=0,
        config=config,
        device=device,
    )
    maximum = 0.0
    decision_tokens = 0
    for item in checked:
        for start in (0, 8):
            candidates = item.candidates[start : start + 8]
            with torch.no_grad():
                scores = score_completions(
                    bundle.model,
                    item.prompt_ids,
                    [candidate.token_ids for candidate in candidates],
                    grammar,
                    device=device,
                    temperature=1.0,
                )
            old, expected_mask = _candidate_old_tensors(
                candidates, torch.device(device)
            )
            width = scores.log_probs.shape[1]
            old, expected_mask = old[:, :width], expected_mask[:, :width]
            if not torch.equal(scores.decision_mask, expected_mask):
                raise RuntimeError("Probability gate decision masks differ.")
            delta = (scores.log_probs - old).abs()[scores.decision_mask]
            maximum = max(
                maximum, float(delta.max().item()) if delta.numel() else 0.0
            )
            decision_tokens += int(scores.decision_mask.sum().item())
    threshold = float(config["gates"]["max_logp_difference"])
    report = {
        "groups_run": len(checked),
        "candidates_run": len(checked) * 16,
        "all_legal": True,
        "all_finite": math.isfinite(maximum),
        "pre_update": True,
        "comparisons": decision_tokens,
        "max_abs_logp_difference": maximum,
    }
    if maximum > threshold:
        raise RuntimeError(f"Probability alignment gate failed: {report}")
    return report


def run_memory_check(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    resolved_contract: dict[str, Any],
    runtime_signature: dict[str, Any],
    max_groups: int = 32,
    output_path: Path | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run four real windows over the deterministic 32 longest prompts."""

    if max_groups != 32:
        raise ValueError("Memory gate requires exactly 32 source groups.")
    groups, trie = _load_groups_and_trie(config, groups_path, trie_dir)
    bundle = load_policy_model(config, device=device)
    bundle.model.train()
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    cutoff = int(config["data"]["cutoff_len"])
    ranked: list[tuple[int, str, RecommendationGroup]] = []
    for group in groups:
        length = len(
            encode_prompt(
                bundle.tokenizer,
                group.system,
                group.prompt,
                cutoff_len=cutoff,
            )
        )
        ranked.append((length, group.group_id, group))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[:32]
    optimizer = build_optimizer(bundle, config)
    parameters = _trainable_parameters(bundle)
    lambda0 = float(resolved_contract["anchor"]["lambda0"])
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("Memory gate requires CUDA.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch_device)
    all_prepared: list[PreparedGroup] = []
    all_results: list[BackwardResult] = []
    backward_passes = 0
    optimizer_updates = 0
    for start in range(0, 32, 8):
        prepared = _rollout_groups(
            bundle,
            [item[2] for item in selected[start : start + 8]],
            grammar,
            epoch_index=0,
            config=config,
            device=device,
        )
        all_prepared.extend(prepared)
        counts = _branch_counts(prepared)
        optimizer.zero_grad(set_to_none=True)
        for item in prepared:
            if item.reward.branch == ObjectiveBranch.RLOO:
                result = backward_rloo_group(
                        bundle,
                        item,
                        grammar,
                        scale=1.0 / counts[ObjectiveBranch.RLOO.value],
                        device=device,
                        max_logp_difference=float(
                            config["gates"]["max_logp_difference"]
                        ),
                    )
                all_results.append(result)
                backward_passes += 1
        for item in prepared:
            if item.reward.branch == ObjectiveBranch.GT_SET_ANCHOR:
                result = backward_anchor_group(
                        bundle,
                        item,
                        grammar,
                        scale=lambda0
                        / counts[ObjectiveBranch.GT_SET_ANCHOR.value],
                        device=device,
                        max_logp_difference=float(
                            config["gates"]["max_logp_difference"]
                        ),
                    )
                all_results.append(result)
                if result.decision_tokens > 0:
                    backward_passes += 1
        if counts[ObjectiveBranch.RLOO.value] or counts[
            ObjectiveBranch.GT_SET_ANCHOR.value
        ]:
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["train"]["max_grad_norm"])
            )
            optimizer.step()
            optimizer_updates += 1
        optimizer.zero_grad(set_to_none=True)
    peak_allocated = torch.cuda.max_memory_allocated(torch_device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(torch_device) / 1024**3
    finite = all(
        math.isfinite(float(item.reward.mean_reward)) for item in all_prepared
    ) and all(math.isfinite(result.loss) for result in all_results)
    return {
        "groups_run": len(all_prepared),
        "candidates_run": len(all_prepared) * contract.GROUP_SIZE,
        "all_legal": all(
            len(item.candidates) == contract.GROUP_SIZE for item in all_prepared
        ),
        "all_finite": finite,
        "selection": "longest_prompt_tokens_desc_group_id_tiebreak",
        "group_ids": [item[1] for item in selected],
        "prompt_token_lengths": [item[0] for item in selected],
        "gpu_name": torch.cuda.get_device_name(torch_device),
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "backward_passes": backward_passes,
        "optimizer_updates": optimizer_updates,
        "rollout_chunk": 8,
        "loss_chunk": 8,
        "gt_chunk": 8,
    }


__all__ = [
    "BackwardResult",
    "PreparedGroup",
    "backward_anchor_group",
    "backward_rloo_group",
    "build_optimizer",
    "calibration_rank",
    "expected_calibration_ids",
    "gradient_l2_norm",
    "learning_rate_for_window",
    "load_calibration_ids",
    "prepare_group",
    "run_calibration",
    "run_memory_check",
    "run_probability_check",
    "run_training",
    "set_global_seed",
    "write_calibration_ids",
]
