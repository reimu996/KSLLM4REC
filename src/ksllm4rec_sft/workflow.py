"""Project-local SFT workflow built on the pinned LLaMA-Factory APIs."""

from __future__ import annotations

from typing import Any

from llamafactory.data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.callbacks import LogCallback
from llamafactory.train.sft.workflow import calculate_tps, plot_loss

from .data import DATASET_NAME
from .checkpoint import CheckpointBindingCallback
from .item_tokens import build_item_token_ids
from .trainer import FocalItemTrainer


def run_focal_sft(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    generating_args,
    *,
    focal_gamma: float,
    item_weight: float,
    lm_chunk_size: int,
    dataset_name: str = DATASET_NAME,
    checkpoint_binding: dict[str, Any] | None = None,
    checkpoint_origin_manifest=None,
) -> dict[str, Any]:
    """Run SFT while replacing only LLaMA-Factory's standard loss path."""

    if training_args.do_eval or training_args.do_predict:
        raise ValueError(
            "This reproducibility run does not accept eval or predict datasets."
        )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    item_token_ids = build_item_token_ids(tokenizer)

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    callbacks = [LogCallback()]
    if checkpoint_binding is not None:
        if checkpoint_origin_manifest is None:
            raise ValueError(
                "checkpoint_origin_manifest is required with checkpoint_binding."
            )
        callbacks.append(
            CheckpointBindingCallback(
                run_identity=checkpoint_binding,
                origin_manifest=checkpoint_origin_manifest,
            )
        )

    trainer = FocalItemTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        model_args=model_args,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
        item_token_ids=item_token_ids,
        focal_gamma=focal_gamma,
        item_weight=item_weight,
        lm_chunk_size=lm_chunk_size,
        dataset_name=dataset_name,
        **dataset_module,
    )

    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()
    if finetuning_args.include_effective_tokens_per_second:
        train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
            dataset_module["train_dataset"], train_result.metrics, stage="sft"
        )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss"])

    return {
        "metrics": train_result.metrics,
        "micro_steps": trainer.micro_step,
        "optimizer_steps": trainer.state.global_step,
        "fallback_count": trainer.fallback_count,
        "item_token_count": len(item_token_ids),
        "min_sequence_length": trainer.min_sequence_length,
        "max_sequence_length": trainer.max_sequence_length,
    }
