"""LLaMA-Factory trainer extension for focal and item-weighted SFT."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean
from typing import Any

import torch

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

from .data import DATASET_NAME
from .loss import chunked_focal_loss, summarize_metrics


def dataset_loss_metric_name(dataset_name: str) -> str:
    if not dataset_name:
        raise ValueError("dataset_name must not be empty.")
    return f"loss_ds_{dataset_name}"


def _unwrap_causal_lm(trainer: "FocalItemTrainer", model):
    unwrapped = trainer.accelerator.unwrap_model(model)
    if hasattr(unwrapped, "get_base_model"):
        unwrapped = unwrapped.get_base_model()
    if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "lm_head"):
        raise TypeError(f"Unsupported causal LM wrapper: {type(unwrapped)!r}")
    return unwrapped


def _forward_with_captured_hidden(model, causal_lm, model_inputs):
    captured_hidden: list[torch.Tensor] = []

    def capture_backbone_output(_module, _args, output) -> None:
        captured_hidden.append(output.last_hidden_state)

    hook = causal_lm.model.register_forward_hook(capture_backbone_output)
    try:
        model_outputs = model(**model_inputs)
    finally:
        hook.remove()
    if len(captured_hidden) != 1:
        raise RuntimeError(
            f"Expected one backbone forward output, captured {len(captured_hidden)}."
        )
    return captured_hidden[0], model_outputs


class FocalItemTrainer(CustomSeq2SeqTrainer):
    """Compute the approved custom loss from hidden states instead of full logits."""

    def __init__(
        self,
        *,
        item_token_ids: list[int],
        focal_gamma: float = 2.0,
        item_weight: float = 3.0,
        lm_chunk_size: int = 512,
        dataset_name: str = DATASET_NAME,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not item_token_ids:
            raise ValueError("item_token_ids must not be empty.")
        if not dataset_name:
            raise ValueError("dataset_name must not be empty.")
        self._item_token_ids_cpu = torch.tensor(item_token_ids, dtype=torch.long)
        self._item_lookup_by_device: dict[torch.device, torch.Tensor] = {}
        self.focal_gamma = float(focal_gamma)
        self.item_weight = float(item_weight)
        self.lm_chunk_size = int(lm_chunk_size)
        self.dataset_name = dataset_name
        self.micro_step = 0
        self.custom_metric_buffer: dict[str, list[float]] = defaultdict(list)
        self.fallback_count = 0
        self.min_sequence_length: int | None = None
        self.max_sequence_length: int | None = None

    def _item_lookup(self, device: torch.device, vocab_size: int) -> torch.Tensor:
        lookup = self._item_lookup_by_device.get(device)
        if lookup is None or lookup.numel() != vocab_size:
            lookup = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            lookup[self._item_token_ids_cpu.to(device)] = True
            self._item_lookup_by_device[device] = lookup
        return lookup

    def _record_metrics(
        self, ce_values: torch.Tensor, item_flags: torch.Tensor
    ) -> None:
        metrics = summarize_metrics(ce_values, item_flags)
        values = {
            "item_ratio": metrics.item_ratio,
            "item_loss": metrics.item_loss,
            "text_loss": metrics.text_loss,
            dataset_loss_metric_name(self.dataset_name): metrics.dataset_loss,
            "valid_tokens": float(metrics.valid_tokens),
            "item_tokens": float(metrics.item_tokens),
        }
        for key, value in values.items():
            if value is not None:
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite custom metric {key}: {value}")
                self.custom_metric_buffer[key].append(float(value))

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            raise KeyError("labels are required for focal SFT.")
        sequence_length = int(labels.size(-1))
        self.min_sequence_length = (
            sequence_length
            if self.min_sequence_length is None
            else min(self.min_sequence_length, sequence_length)
        )
        self.max_sequence_length = (
            sequence_length
            if self.max_sequence_length is None
            else max(self.max_sequence_length, sequence_length)
        )

        causal_lm = _unwrap_causal_lm(self, model)
        lm_head_weight = causal_lm.lm_head.weight
        if lm_head_weight.requires_grad:
            raise RuntimeError(
                "LM head is unexpectedly trainable; the approved loss assumes it is frozen."
            )

        model_inputs: dict[str, Any] = {
            key: inputs[key]
            for key in ("input_ids", "attention_mask", "position_ids", "inputs_embeds")
            if key in inputs and inputs[key] is not None
        }
        model_inputs.update(use_cache=False, return_dict=True, logits_to_keep=1)
        hidden_states, model_outputs = _forward_with_captured_hidden(
            model, causal_lm, model_inputs
        )

        lookup = self._item_lookup(labels.device, lm_head_weight.size(0))
        safe_labels = labels.clamp_min(0)
        item_mask = lookup[safe_labels] & labels.ne(-100)

        loss, ce_values, item_flags = chunked_focal_loss(
            hidden_states,
            lm_head_weight,
            labels,
            item_mask,
            gamma=self.focal_gamma,
            item_weight=self.item_weight,
            chunk_size=self.lm_chunk_size,
        )
        if not bool(torch.isfinite(loss)):
            self.fallback_count += 1
            loss, ce_values, item_flags = chunked_focal_loss(
                hidden_states,
                lm_head_weight,
                labels,
                item_mask,
                gamma=0.0,
                item_weight=1.0,
                chunk_size=self.lm_chunk_size,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    "Custom focal loss and standard CE fallback are both non-finite."
                )

        self.micro_step += 1
        self._record_metrics(ce_values, item_flags)
        if torch.cuda.is_available():
            gib = 1024**3
            self.custom_metric_buffer["peak_memory_allocated_gib"].append(
                torch.cuda.max_memory_allocated() / gib
            )
            self.custom_metric_buffer["peak_memory_reserved_gib"].append(
                torch.cuda.max_memory_reserved() / gib
            )

        if return_outputs:
            return loss, model_outputs
        return loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self.custom_metric_buffer:
            for key, values in self.custom_metric_buffer.items():
                if values:
                    logs[key] = fmean(values)
            logs["micro_step"] = self.micro_step
            logs["optimizer_step"] = self.state.global_step
            logs["ce_fallback_count"] = self.fallback_count
            self.custom_metric_buffer.clear()
        return super().log(logs, *args, **kwargs)
