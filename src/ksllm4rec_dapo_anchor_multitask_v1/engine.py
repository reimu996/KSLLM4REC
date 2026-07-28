"""Core state transitions for the Multitask V1 policy optimizer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import torch

from . import contract
from .objective import anchor_effective_lambda


def learning_rate_for_policy_step(
    config: Mapping[str, Any], policy_step: int
) -> float:
    """Return the LR used by a one-based policy optimizer step."""

    if isinstance(policy_step, bool) or not isinstance(policy_step, int):
        raise TypeError("policy_step must be an integer.")
    if policy_step < 1:
        raise ValueError("policy_step must be one-based and positive.")
    train = config["train"]
    per_level = int(train["warmup_steps_per_level"])
    warmup = int(train["warmup_policy_steps"])
    if per_level <= 0 or warmup <= 0 or warmup % per_level:
        raise ValueError("The policy-step warmup contract is invalid.")
    levels = warmup // per_level
    level = min((policy_step - 1) // per_level + 1, levels)
    return float(train["learning_rate"]) * float(level) / float(levels)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


class PolicyStepLRScheduler:
    """Advance only when a minibatch performs a policy optimizer update."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
        *,
        completed_policy_steps: int = 0,
    ) -> None:
        if completed_policy_steps < 0:
            raise ValueError("completed_policy_steps must be non-negative.")
        self.optimizer = optimizer
        self.config = config
        self.completed_policy_steps = int(completed_policy_steps)
        _set_optimizer_lr(
            self.optimizer,
            learning_rate_for_policy_step(config, self.completed_policy_steps + 1),
        )

    def step(self) -> None:
        self.completed_policy_steps += 1
        _set_optimizer_lr(
            self.optimizer,
            learning_rate_for_policy_step(
                self.config, self.completed_policy_steps + 1
            ),
        )

    def state_dict(self) -> dict[str, int]:
        return {"completed_policy_steps": self.completed_policy_steps}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if set(value) != {"completed_policy_steps"}:
            raise ValueError("Policy scheduler state has an invalid schema.")
        completed = value["completed_policy_steps"]
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise TypeError("completed_policy_steps must be an integer.")
        if completed < 0:
            raise ValueError("completed_policy_steps must be non-negative.")
        self.completed_policy_steps = completed
        _set_optimizer_lr(
            self.optimizer,
            learning_rate_for_policy_step(self.config, completed + 1),
        )


def _task_and_group_id(item: Any) -> tuple[str, str]:
    group = item.group
    task = group.task.value if hasattr(group.task, "value") else str(group.task)
    group_id = str(group.group_id)
    if not task or not group_id:
        raise ValueError("Every policy group needs task and group_id.")
    return task, group_id


def partition_policy_groups(
    groups: Sequence[Any], *, optimization_window_index: int, seed: int
) -> tuple[tuple[Any, ...], ...]:
    """Stable K=1 shuffle into batches of at most eight groups."""

    values = tuple(groups)
    if not values:
        raise ValueError("A policy optimization window needs at least one group.")
    identities = [_task_and_group_id(item) for item in values]
    if len(set(identities)) != len(identities):
        raise ValueError("Policy group identities must be unique within a window.")
    prefix = (
        f"multitask-minibatch|{int(seed)}|{int(optimization_window_index)}|"
    ).encode("utf-8")
    ordered = tuple(
        sorted(
            values,
            key=lambda item: hashlib.sha256(
                prefix
                + "|".join(_task_and_group_id(item)).encode("utf-8")
            ).digest(),
        )
    )
    batches = tuple(
        ordered[start : start + contract.MINIBATCH_GROUPS]
        for start in range(0, len(ordered), contract.MINIBATCH_GROUPS)
    )
    flattened = [_task_and_group_id(item) for batch in batches for item in batch]
    if len(flattened) != len(values) or set(flattened) != set(identities):
        raise RuntimeError("K=1 partition lost or reused a policy group.")
    if any(not 1 <= len(batch) <= contract.MINIBATCH_GROUPS for batch in batches):
        raise RuntimeError("A policy minibatch is outside size 1..8.")
    return batches


def gradient_l2_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
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


def clone_gradients(
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )


@dataclass(frozen=True)
class GradientMergeResult:
    anchor_raw_gradient_norm: float
    anchor_scaled_gradient_norm: float
    lambda_cap: float
    lambda_effective: float
    anchor_to_rl_gradient_ratio: float
    combined_gradient_norm: float
    anchor_optimizer_steps: int = 0


@dataclass(frozen=True)
class PolicyUpdateExecution:
    policy_optimizer_steps: int
    anchor_optimizer_steps: int
    policy_gradient_norms: tuple[float, ...]
    rl_reference_gradient_norm: float
    policy_outputs: tuple[Any, ...]
    anchor_output: Any
    gradient_merge: GradientMergeResult


@dataclass(frozen=True)
class FinalAuxiliaryExecution:
    optimizer_steps: int
    anchor_output: Any
    gradient_merge: GradientMergeResult


def combine_last_policy_and_anchor_gradients(
    parameters: Sequence[torch.nn.Parameter],
    last_policy_gradients: Sequence[torch.Tensor | None],
    *,
    lambda_calibrated: float,
    rl_reference_gradient_norm: float,
    target_gradient_ratio: float,
    epsilon: float,
) -> GradientMergeResult:
    """Replace ``.grad`` with last-policy + bounded Anchor gradient."""

    params = tuple(parameters)
    last = tuple(last_policy_gradients)
    if len(params) != len(last):
        raise ValueError("Gradient buffer does not match the trainable parameters.")
    anchor_norm = gradient_l2_norm(params)
    budget = anchor_effective_lambda(
        lambda_calibrated=float(lambda_calibrated),
        rl_reference_grad_norm=float(rl_reference_gradient_norm),
        anchor_raw_grad_norm=float(anchor_norm),
        target_gradient_ratio=float(target_gradient_ratio),
        epsilon=float(epsilon),
    )
    for parameter, policy_gradient in zip(params, last, strict=True):
        anchor_gradient = parameter.grad
        if policy_gradient is None and anchor_gradient is None:
            parameter.grad = None
            continue
        if policy_gradient is None:
            combined = torch.zeros_like(anchor_gradient)
        else:
            if policy_gradient.shape != parameter.shape:
                raise ValueError("Saved policy gradient has the wrong shape.")
            combined = policy_gradient.to(parameter.device).clone()
        if anchor_gradient is not None and budget.lambda_effective > 0.0:
            combined.add_(anchor_gradient, alpha=float(budget.lambda_effective))
        parameter.grad = combined
    scaled = float(budget.lambda_effective) * float(anchor_norm)
    tolerance = max(1.0e-12, target_gradient_ratio * rl_reference_gradient_norm * 1.0e-6)
    if scaled > target_gradient_ratio * rl_reference_gradient_norm + tolerance:
        raise RuntimeError("Scaled Anchor gradient exceeds the RL gradient budget.")
    return GradientMergeResult(
        anchor_raw_gradient_norm=anchor_norm,
        anchor_scaled_gradient_norm=scaled,
        lambda_cap=budget.lambda_cap,
        lambda_effective=budget.lambda_effective,
        anchor_to_rl_gradient_ratio=budget.anchor_to_rl_grad_ratio,
        combined_gradient_norm=gradient_l2_norm(params),
    )


def execute_policy_updates(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: PolicyStepLRScheduler,
    batches: Sequence[Any],
    *,
    backward_policy: Callable[[int, Any], Any],
    backward_anchor: Callable[[], Any] | None,
    lambda_calibrated: float,
    target_gradient_ratio: float,
    epsilon: float,
    max_grad_norm: float,
) -> PolicyUpdateExecution:
    """Commit all policy minibatches; merge Anchor into the final step."""

    params = tuple(parameters)
    values = tuple(batches)
    if not params or not values:
        raise ValueError("Policy execution needs parameters and minibatches.")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive.")
    outputs: list[Any] = []
    norms: list[float] = []
    last_gradients: tuple[torch.Tensor | None, ...] | None = None
    for index, batch in enumerate(values):
        optimizer.zero_grad(set_to_none=True)
        outputs.append(backward_policy(index, batch))
        norm = gradient_l2_norm(params)
        norms.append(norm)
        if index + 1 < len(values):
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
            optimizer.step()
            scheduler.step()
        else:
            last_gradients = clone_gradients(params)
    if last_gradients is None:
        raise RuntimeError("The final policy gradient was not captured.")
    rl_reference = float(median(norms))
    optimizer.zero_grad(set_to_none=True)
    anchor_output = backward_anchor() if backward_anchor is not None else None
    merge = combine_last_policy_and_anchor_gradients(
        params,
        last_gradients,
        lambda_calibrated=float(lambda_calibrated),
        rl_reference_gradient_norm=rl_reference,
        target_gradient_ratio=float(target_gradient_ratio),
        epsilon=float(epsilon),
    )
    torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
    optimizer.step()
    scheduler.step()
    return PolicyUpdateExecution(
        policy_optimizer_steps=len(values),
        anchor_optimizer_steps=0,
        policy_gradient_norms=tuple(norms),
        rl_reference_gradient_norm=rl_reference,
        policy_outputs=tuple(outputs),
        anchor_output=anchor_output,
        gradient_merge=merge,
    )


def execute_final_auxiliary_flush(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    backward_anchor: Callable[[], Any],
    lambda_calibrated: float,
    rl_reference_gradient_norm: float,
    target_gradient_ratio: float,
    epsilon: float,
    max_grad_norm: float,
) -> FinalAuxiliaryExecution:
    """Commit the one source-end Anchor-only exception without advancing LR."""

    params = tuple(parameters)
    if not params:
        raise ValueError("Final auxiliary flush needs trainable parameters.")
    optimizer.zero_grad(set_to_none=True)
    output = backward_anchor()
    merge = combine_last_policy_and_anchor_gradients(
        params,
        (None,) * len(params),
        lambda_calibrated=float(lambda_calibrated),
        rl_reference_gradient_norm=float(rl_reference_gradient_norm),
        target_gradient_ratio=float(target_gradient_ratio),
        epsilon=float(epsilon),
    )
    steps = 0
    if merge.lambda_effective > 0.0:
        torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        optimizer.step()
        steps = 1
    else:
        optimizer.zero_grad(set_to_none=True)
    return FinalAuxiliaryExecution(
        optimizer_steps=steps,
        anchor_output=output,
        gradient_merge=merge,
    )


__all__ = [
    "GradientMergeResult",
    "FinalAuxiliaryExecution",
    "PolicyUpdateExecution",
    "PolicyStepLRScheduler",
    "clone_gradients",
    "combine_last_policy_and_anchor_gradients",
    "execute_policy_updates",
    "execute_final_auxiliary_flush",
    "gradient_l2_norm",
    "learning_rate_for_policy_step",
    "partition_policy_groups",
]
