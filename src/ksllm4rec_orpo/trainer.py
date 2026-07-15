"""LLaMA-Factory ORPO trainer without full-sequence vocabulary logits."""

from __future__ import annotations

from typing import Any, Literal

import torch

from llamafactory.train.dpo.trainer import CustomDPOTrainer

from ksllm4rec_sft.trainer import (
    _forward_with_captured_hidden,
    _unwrap_causal_lm,
)

from .loss import chunked_sequence_logps, stable_orpo_loss


class ChunkedORPOTrainer(CustomDPOTrainer):
    """Compute canonical ORPO from sequence log-probs in bounded LM-Head chunks."""

    def __init__(self, *, lm_chunk_size: int = 512, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.loss_type != "orpo" or self.finetuning_args.use_ref_model:
            raise ValueError("ChunkedORPOTrainer requires reference-free ORPO.")
        if lm_chunk_size <= 0:
            raise ValueError("lm_chunk_size must be positive.")
        self.lm_chunk_size = int(lm_chunk_size)
        self.micro_step = 0
        self.min_sequence_length: int | None = None
        self.max_sequence_length: int | None = None
        self.min_chosen_tokens: int | None = None
        self.max_chosen_tokens: int | None = None
        self.min_rejected_tokens: int | None = None
        self.max_rejected_tokens: int | None = None

    @staticmethod
    def _bounded(value: int, minimum: int | None, maximum: int | None):
        return (
            value if minimum is None else min(minimum, value),
            value if maximum is None else max(maximum, value),
        )

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch.get("labels")
        input_ids = batch.get("input_ids")
        if labels is None or input_ids is None:
            raise KeyError("ORPO batches require input_ids and labels.")
        row_count, sequence_length = labels.shape
        if row_count < 2 or row_count % 2 != 0:
            raise ValueError(
                f"Pair collator must produce an even row count, got {row_count}."
            )
        self.min_sequence_length, self.max_sequence_length = self._bounded(
            int(sequence_length), self.min_sequence_length, self.max_sequence_length
        )

        causal_lm = _unwrap_causal_lm(self, model)
        lm_head_weight = causal_lm.lm_head.weight
        if lm_head_weight.requires_grad:
            raise RuntimeError("Chunked ORPO requires a frozen LM Head.")

        model_inputs: dict[str, Any] = {
            key: batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "position_ids",
                "inputs_embeds",
            )
            if key in batch and batch[key] is not None
        }
        model_inputs.update(use_cache=False, return_dict=True, logits_to_keep=1)
        hidden_states, _ = _forward_with_captured_hidden(
            model, causal_lm, model_inputs
        )
        all_logps, token_counts = chunked_sequence_logps(
            hidden_states,
            lm_head_weight,
            labels,
            chunk_size=self.lm_chunk_size,
        )
        pair_batch_size = row_count // 2
        chosen_logps, rejected_logps = all_logps.split(pair_batch_size, dim=0)
        chosen_counts, rejected_counts = token_counts.split(pair_batch_size, dim=0)
        output = stable_orpo_loss(
            chosen_logps, rejected_logps, beta=float(self.beta)
        )

        chosen_min = int(chosen_counts.min().item())
        chosen_max = int(chosen_counts.max().item())
        rejected_min = int(rejected_counts.min().item())
        rejected_max = int(rejected_counts.max().item())
        self.min_chosen_tokens, self.max_chosen_tokens = self._bounded(
            chosen_min, self.min_chosen_tokens, self.max_chosen_tokens
        )
        self.min_chosen_tokens, self.max_chosen_tokens = self._bounded(
            chosen_max, self.min_chosen_tokens, self.max_chosen_tokens
        )
        self.min_rejected_tokens, self.max_rejected_tokens = self._bounded(
            rejected_min, self.min_rejected_tokens, self.max_rejected_tokens
        )
        self.min_rejected_tokens, self.max_rejected_tokens = self._bounded(
            rejected_max, self.min_rejected_tokens, self.max_rejected_tokens
        )
        self.micro_step += 1

        prefix = "eval_" if train_eval == "eval" else ""
        chosen_rewards = float(self.beta) * chosen_logps.detach()
        rejected_rewards = float(self.beta) * rejected_logps.detach()
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().item(),
            f"{prefix}rewards/accuracies": (
                chosen_logps > rejected_logps
            ).float().mean().item(),
            f"{prefix}rewards/margins": (
                chosen_rewards - rejected_rewards
            ).mean().item(),
            f"{prefix}logps/chosen": chosen_logps.detach().mean().item(),
            f"{prefix}logps/rejected": rejected_logps.detach().mean().item(),
            f"{prefix}sft_loss": output.sft_loss.detach().mean().item(),
            f"{prefix}odds_ratio_loss": output.odds_ratio_loss.detach()
            .mean()
            .item(),
            f"{prefix}log_odds": output.log_odds.detach().mean().item(),
            f"{prefix}tokens/chosen": chosen_counts.float().mean().item(),
            f"{prefix}tokens/rejected": rejected_counts.float().mean().item(),
        }
        if torch.cuda.is_available():
            metrics[f"{prefix}peak_memory_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            metrics[f"{prefix}peak_memory_reserved_gib"] = (
                torch.cuda.max_memory_reserved() / 1024**3
            )
        return output.loss, metrics
