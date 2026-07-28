from __future__ import annotations

from types import SimpleNamespace
import math
import unittest

import torch

from ksllm4rec_dapo_anchor_multitask_v1.config import build_config
from ksllm4rec_dapo_anchor_multitask_v1.engine import (
    PolicyStepLRScheduler,
    combine_last_policy_and_anchor_gradients,
    execute_final_auxiliary_flush,
    execute_policy_updates,
    learning_rate_for_policy_step,
    partition_policy_groups,
)


def group(task: str, group_id: str):
    return SimpleNamespace(
        group=SimpleNamespace(task=SimpleNamespace(value=task), group_id=group_id)
    )


class PolicyStepScheduleTest(unittest.TestCase):
    def test_lr_is_piecewise_constant_in_four_step_levels(self) -> None:
        config = build_config()
        expected = {
            1: 0.1e-6,
            4: 0.1e-6,
            5: 0.2e-6,
            8: 0.2e-6,
            9: 0.3e-6,
            36: 0.9e-6,
            37: 1.0e-6,
            40: 1.0e-6,
            41: 1.0e-6,
        }
        for step, value in expected.items():
            self.assertTrue(
                math.isclose(
                    learning_rate_for_policy_step(config, step),
                    value,
                    rel_tol=0.0,
                    abs_tol=1.0e-18,
                )
            )

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=99.0)
        scheduler = PolicyStepLRScheduler(optimizer, config)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1e-6)
        for _ in range(4):
            scheduler.step()
        self.assertEqual(scheduler.completed_policy_steps, 4)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.2e-6)


class VariableMinibatchTest(unittest.TestCase):
    def test_33_groups_partition_8_8_8_8_1_without_loss_or_reuse(self) -> None:
        groups = tuple(group("recommendation", f"g-{index}") for index in range(33))
        batches = partition_policy_groups(
            groups, optimization_window_index=7, seed=42
        )
        self.assertEqual([len(batch) for batch in batches], [8, 8, 8, 8, 1])
        observed = [item.group.group_id for batch in batches for item in batch]
        self.assertEqual(set(observed), {f"g-{index}" for index in range(33)})
        self.assertEqual(len(observed), len(set(observed)))


class GradientMergeTest(unittest.TestCase):
    def test_anchor_is_scaled_to_ten_percent_then_added_to_last_rl_gradient(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        last_rl = (torch.tensor([3.0, 4.0]),)
        parameter.grad = torch.tensor([0.0, 4.0])

        result = combine_last_policy_and_anchor_gradients(
            [parameter],
            last_rl,
            lambda_calibrated=1.0,
            rl_reference_gradient_norm=2.0,
            target_gradient_ratio=0.10,
            epsilon=1.0e-12,
        )

        self.assertAlmostEqual(result.anchor_raw_gradient_norm, 4.0)
        self.assertAlmostEqual(result.lambda_effective, 0.05)
        self.assertAlmostEqual(result.anchor_scaled_gradient_norm, 0.2)
        torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.2]))
        self.assertEqual(result.anchor_optimizer_steps, 0)

    def test_anchor_is_merged_into_last_policy_step_without_extra_step(self) -> None:
        config = build_config()
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

        class CountingSGD(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=99.0)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        optimizer = CountingSGD([parameter])
        scheduler = PolicyStepLRScheduler(optimizer, config)
        policy_gradients = (torch.tensor([1.0, 0.0]), torch.tensor([3.0, 4.0]))

        def backward_policy(index, _batch):
            parameter.grad = policy_gradients[index].clone()
            return f"policy-{index}"

        def backward_anchor():
            parameter.grad = torch.tensor([0.0, 4.0])
            return "anchor"

        result = execute_policy_updates(
            [parameter],
            optimizer,
            scheduler,
            (("a",), ("b",)),
            backward_policy=backward_policy,
            backward_anchor=backward_anchor,
            lambda_calibrated=1.0,
            target_gradient_ratio=0.10,
            epsilon=1.0e-12,
            max_grad_norm=100.0,
        )

        self.assertEqual(optimizer.step_calls, 2)
        self.assertEqual(scheduler.completed_policy_steps, 2)
        self.assertEqual(result.policy_optimizer_steps, 2)
        self.assertEqual(result.anchor_optimizer_steps, 0)
        self.assertEqual(result.policy_gradient_norms, (1.0, 5.0))
        self.assertAlmostEqual(result.rl_reference_gradient_norm, 3.0)
        self.assertAlmostEqual(result.gradient_merge.anchor_scaled_gradient_norm, 0.3)

    def test_source_end_anchor_flush_performs_at_most_one_bounded_step(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

        class CountingSGD(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=0.1)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        optimizer = CountingSGD([parameter])

        def backward_anchor():
            parameter.grad = torch.tensor([3.0, 4.0])
            return "anchor-tail"

        result = execute_final_auxiliary_flush(
            [parameter],
            optimizer,
            backward_anchor=backward_anchor,
            lambda_calibrated=1.0,
            rl_reference_gradient_norm=2.0,
            target_gradient_ratio=0.10,
            epsilon=1.0e-12,
            max_grad_norm=100.0,
        )

        self.assertEqual(result.optimizer_steps, 1)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertAlmostEqual(result.gradient_merge.anchor_raw_gradient_norm, 5.0)
        self.assertAlmostEqual(result.gradient_merge.anchor_scaled_gradient_norm, 0.2)

    def test_source_end_anchor_flush_skips_when_no_rl_reference_exists(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))

        class CountingSGD(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=0.1)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        optimizer = CountingSGD([parameter])

        def backward_anchor():
            parameter.grad = torch.tensor([9.0])
            return "anchor-tail"

        result = execute_final_auxiliary_flush(
            [parameter],
            optimizer,
            backward_anchor=backward_anchor,
            lambda_calibrated=0.05,
            rl_reference_gradient_norm=0.0,
            target_gradient_ratio=0.10,
            epsilon=1.0e-12,
            max_grad_norm=1.0,
        )

        self.assertEqual(result.optimizer_steps, 0)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(result.gradient_merge.lambda_effective, 0.0)


if __name__ == "__main__":
    unittest.main()
