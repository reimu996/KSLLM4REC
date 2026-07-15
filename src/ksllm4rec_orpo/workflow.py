"""Project-local ORPO workflow built on pinned LLaMA-Factory APIs."""

from __future__ import annotations

from typing import Any

from llamafactory.data import (
    PairwiseDataCollatorWithPadding,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import calculate_tps
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.callbacks import LogCallback

from .trainer import ChunkedORPOTrainer


def run_chunked_orpo(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    *,
    lm_chunk_size: int,
) -> dict[str, Any]:
    if training_args.do_eval or training_args.do_predict:
        raise ValueError("The controlled full run does not accept eval or predict.")
    if finetuning_args.use_ref_model or finetuning_args.pref_loss != "orpo":
        raise ValueError("The controlled workflow accepts reference-free ORPO only.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="rm",
        **tokenizer_module,
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        **tokenizer_module,
    )
    trainer = ChunkedORPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        lm_chunk_size=lm_chunk_size,
        **dataset_module,
        **tokenizer_module,
    )
    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()
    if finetuning_args.include_effective_tokens_per_second:
        train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
            dataset_module["train_dataset"], train_result.metrics, stage="rm"
        )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    return {
        "metrics": train_result.metrics,
        "micro_steps": trainer.micro_step,
        "optimizer_steps": trainer.state.global_step,
        "completed_epoch": trainer.state.epoch,
        "min_sequence_length": trainer.min_sequence_length,
        "max_sequence_length": trainer.max_sequence_length,
        "min_chosen_tokens": trainer.min_chosen_tokens,
        "max_chosen_tokens": trainer.max_chosen_tokens,
        "min_rejected_tokens": trainer.min_rejected_tokens,
        "max_rejected_tokens": trainer.max_rejected_tokens,
        "reference_model_used": False,
        "lm_chunk_size": lm_chunk_size,
    }
