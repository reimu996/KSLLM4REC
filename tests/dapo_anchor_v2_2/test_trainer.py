from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ksllm4rec_dapo_anchor_v2_2 import trainer
from ksllm4rec_dapo_anchor_v2_2._infra.orpo_data import Sid
from ksllm4rec_dapo_anchor_v2_2.anchor_scoring import (
    score_prompt_completion_pairs_dense,
)
from ksllm4rec_dapo_anchor_v2_2.config import build_config
from ksllm4rec_dapo_anchor_v2_2.objective import RewardOutput
from ksllm4rec_dapo_anchor_v2_2.objective import (
    gt_sequence_logps,
    gt_set_anchor_loss,
)


def _reward(*, effective: bool, tier: str) -> RewardOutput:
    values = torch.zeros(16)
    return RewardOutput(
        rewards=values,
        advantages=values,
        mean_reward=values.mean(),
        std_reward=values.std(correction=0),
        tiers=(tier,) * 16,
        effective=effective,
    )


class AnchorCandidateRuleTest(unittest.TestCase):
    def test_reward_flat_without_exact_is_anchor_candidate(self) -> None:
        self.assertTrue(
            trainer.is_anchor_candidate(_reward(effective=False, tier="same_domain"))
        )

    def test_effective_or_all_exact_group_is_not_anchor_candidate(self) -> None:
        self.assertFalse(
            trainer.is_anchor_candidate(_reward(effective=True, tier="same_domain"))
        )
        self.assertFalse(
            trainer.is_anchor_candidate(_reward(effective=False, tier="exact"))
        )


class W0AnchorEnablementTest(unittest.TestCase):
    @staticmethod
    def _groups() -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(group=SimpleNamespace(group_id=f"rl-{index}"))
            for index in range(32)
        )

    @staticmethod
    def _rl_step(groups, optimizer) -> trainer.OptimizerStepResult:
        return trainer.OptimizerStepResult(
            loss=0.0,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            decision_tokens=128,
            gradient_norm=0.05,
            max_replay_logp_difference=0.0,
            ratio_min=1.0,
            ratio_max=1.0,
            ratio_mean=1.0,
            clipped_tokens=0,
            below_low_tokens=0,
            above_high_tokens=0,
            group_ids=tuple(item.group.group_id for item in groups),
        )

    @staticmethod
    def _anchor_result(candidates) -> trainer.AnchorStepResult:
        return trainer.AnchorStepResult(
            candidate_count=len(candidates),
            groups_used=len(candidates),
            group_ids=tuple(item.group.group_id for item in candidates),
            gt_sid_count=len(candidates),
            decision_tokens=3 * len(candidates),
            set_nll_mean=2.0,
            all_gt_decision_nll=1.0,
            phase_seconds=0.1,
            raw_gradient_norm=4.0,
            scaled_gradient_norm=0.005,
            rl_reference_gradient_norm=0.05,
            lambda_calibrated=0.01,
            lambda_cap=0.00125,
            lambda_effective=0.00125,
            anchor_to_rl_gradient_ratio=0.10,
            max_replay_logp_difference=0.0,
            optimizer_steps=1,
            skip_reason=None,
        )

    def test_w0_with_candidates_runs_four_rl_steps_then_one_anchor_step(self) -> None:
        config = build_config()
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=99.0)
        scheduler = trainer.WindowLRScheduler(
            optimizer, config, completed_windows=0
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1e-6)
        candidates = tuple(
            SimpleNamespace(group=SimpleNamespace(group_id=f"anchor-{index}"))
            for index in range(2)
        )

        def fake_rl(bundle, current_optimizer, groups, grammar, **kwargs):
            del bundle, grammar, kwargs
            return self._rl_step(groups, current_optimizer)

        anchor_result = self._anchor_result(candidates)
        with (
            patch.object(trainer, "train_minibatch", side_effect=fake_rl) as rl_mock,
            patch.object(
                trainer, "run_anchor_phase", return_value=anchor_result
            ) as anchor_mock,
        ):
            result = trainer.train_complete_window(
                SimpleNamespace(),
                optimizer,
                scheduler,
                self._groups(),
                candidates,
                SimpleNamespace(),
                window_index=0,
                lambda_calibrated=0.01,
                config=config,
                device="cpu",
            )

        self.assertEqual(rl_mock.call_count, 4)
        anchor_mock.assert_called_once()
        self.assertEqual(result.rl_optimizer_steps, 4)
        self.assertEqual(result.anchor_optimizer_steps, 1)
        self.assertEqual(result.optimizer_updates, 5)
        self.assertEqual(scheduler.completed_windows, 1)

    def test_w0_without_candidates_runs_only_four_rl_steps(self) -> None:
        config = build_config()
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=99.0)
        scheduler = trainer.WindowLRScheduler(
            optimizer, config, completed_windows=0
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1e-6)

        def fake_rl(bundle, current_optimizer, groups, grammar, **kwargs):
            del bundle, grammar, kwargs
            return self._rl_step(groups, current_optimizer)

        with (
            patch.object(trainer, "train_minibatch", side_effect=fake_rl) as rl_mock,
            patch.object(trainer, "run_anchor_phase") as anchor_mock,
        ):
            result = trainer.train_complete_window(
                SimpleNamespace(),
                optimizer,
                scheduler,
                self._groups(),
                (),
                SimpleNamespace(),
                window_index=0,
                lambda_calibrated=0.01,
                config=config,
                device="cpu",
            )

        self.assertEqual(rl_mock.call_count, 4)
        anchor_mock.assert_not_called()
        self.assertEqual(result.rl_optimizer_steps, 4)
        self.assertEqual(result.anchor_optimizer_steps, 0)
        self.assertEqual(result.optimizer_updates, 4)
        self.assertEqual(scheduler.completed_windows, 1)


class AllCandidateAnchorStepTest(unittest.TestCase):
    def test_129_candidates_produce_exactly_one_bounded_optimizer_step(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        bundle = SimpleNamespace(model=model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        candidates = tuple(
            SimpleNamespace(group=SimpleNamespace(group_id=f"g{index}"))
            for index in range(129)
        )
        group_result = trainer.AnchorGroupBackwardResult(
            set_nll=2.0,
            all_gt_decision_nll_sum=6.0,
            gt_sid_count=1,
            decision_tokens=3,
            max_replay_logp_difference=0.0,
        )

        def fake_raw(*args, **kwargs):
            del args, kwargs
            model.weight.grad = torch.tensor([[4.0]])
            return trainer.AnchorRawGradientResult(
                group_results=(group_result,) * len(candidates),
                raw_gradient_norm=4.0,
                max_replay_logp_difference=0.0,
            )

        config = {
            "anchor": {
                "max_weight": 0.05,
                "target_gradient_ratio": 0.10,
                "gradient_epsilon": 1.0e-12,
            },
            "train": {"max_grad_norm": 1.0},
        }
        with patch.object(trainer, "build_anchor_raw_gradient", side_effect=fake_raw):
            result = trainer.run_anchor_phase(
                bundle,
                optimizer,
                candidates,
                SimpleNamespace(),
                rl_reference_gradient_norm=0.05,
                lambda_calibrated=0.01,
                config=config,
                device="cpu",
            )

        self.assertEqual(result.candidate_count, 129)
        self.assertEqual(result.groups_used, 129)
        self.assertEqual(len(result.group_ids), 129)
        self.assertEqual(result.optimizer_steps, 1)
        self.assertAlmostEqual(result.lambda_effective, 0.00125, places=12)
        self.assertAlmostEqual(result.anchor_to_rl_gradient_ratio, 0.10, places=6)
        trainer._assert_optimizer_state_step(optimizer, 1)


class _Output:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> _Output:
        del kwargs
        return _Output(self.embedding(input_ids))


class _ToyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(4, 128, bias=False)


class _Grammar:
    eos_token_id = 99

    def allowed_next(self, prefix):
        if not prefix:
            return [10, 11]
        if prefix == [10]:
            return [12]
        if prefix == [11]:
            return [13, 14]
        return [99]

    def parse(self, tokens):
        if tokens not in ([10, 12, 99], [11, 13, 99]):
            raise ValueError("invalid")
        return tokens

    def encode_sid(self, sid: Sid):
        return [10, 12, 99] if sid.c == 0 else [11, 13, 99]


class HeterogeneousAnchorGradientTest(unittest.TestCase):
    def test_cross_prompt_microbatch_matches_direct_gt_set_gradient(self) -> None:
        torch.manual_seed(9)
        actual_model = _ToyLM()
        expected_model = _ToyLM()
        expected_model.load_state_dict(actual_model.state_dict())
        grammar = _Grammar()
        sid0 = Sid("video", 1, 1, 0)
        sid1 = Sid("video", 1, 1, 1)
        candidates = (
            SimpleNamespace(
                group=SimpleNamespace(group_id="g0"),
                prompt_ids=(1, 2),
                positives=(sid0, sid1),
            ),
            SimpleNamespace(
                group=SimpleNamespace(group_id="g1"),
                prompt_ids=(3, 4, 5),
                positives=(sid1,),
            ),
        )
        config = {
            "anchor": {
                "gt_completion_microbatch": 8,
                "teacher_forcing_temperature": 1.0,
                "replay_max_logp_difference": 1.0e-5,
            }
        }
        actual_optimizer = torch.optim.AdamW(actual_model.parameters(), lr=0.0)
        result = trainer.build_anchor_raw_gradient(
            SimpleNamespace(model=actual_model),
            actual_optimizer,
            candidates,
            grammar,
            config=config,
            device="cpu",
        )

        expected_losses = []
        for candidate in candidates:
            completions = [grammar.encode_sid(sid) for sid in candidate.positives]
            scores = score_prompt_completion_pairs_dense(
                expected_model,
                [candidate.prompt_ids] * len(completions),
                completions,
                grammar,
                device="cpu",
                temperature=1.0,
                completion_width=32,
            )
            sequences, _ = gt_sequence_logps(
                scores.log_probs, scores.decision_mask
            )
            expected_losses.append(gt_set_anchor_loss(sequences))
        torch.stack(expected_losses).mean().backward()

        self.assertEqual(len(result.group_results), 2)
        self.assertLessEqual(result.max_replay_logp_difference, 1.0e-5)
        for actual, expected in zip(
            actual_model.parameters(), expected_model.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
