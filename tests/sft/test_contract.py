from __future__ import annotations

import unittest

from ksllm4rec_sft.contract import validate_training_contract


def approved_config() -> dict:
    return {
        "model_name_or_path": "/home/lyc/models/OneReason-0.8B-pretrain-competition",
        "trust_remote_code": True,
        "flash_attn": "fa2",
        "disable_gradient_checkpointing": False,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "additional_target": None,
        "pure_bf16": False,
        "dataset": "hf_kuaishou_llmrec_sft_baseline_0_91",
        "dataset_dir": "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/sft/data/hf_baseline_091",
        "template": "qwen3_nothink",
        "cutoff_len": 16384,
        "packing": True,
        "neat_packing": True,
        "train_on_prompt": False,
        "mask_history": False,
        "overwrite_cache": False,
        "preprocessing_num_workers": 8,
        "preprocessing_batch_size": 1000,
        "dataloader_num_workers": 2,
        "logging_strategy": "steps",
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": 256,
        "save_total_limit": 2,
        "save_only_model": False,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "report_to": "none",
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.001,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "ddp_find_unused_parameters": False,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
    }


class TrainingContractTest(unittest.TestCase):
    def test_accepts_the_approved_full_configuration(self) -> None:
        report = validate_training_contract(
            approved_config(),
            {"gamma": 2.0, "item_weight": 3.0, "chunk_size": 512},
            "full_epoch_001",
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["requested_cutoff_len"], 16384)
        self.assertEqual(report["gradient_accumulation_steps"], 8)

    def test_rejects_learning_rate_drift(self) -> None:
        config = approved_config()
        config["learning_rate"] = 1.0e-4
        with self.assertRaisesRegex(RuntimeError, "learning_rate"):
            validate_training_contract(
                config,
                {"gamma": 2.0, "item_weight": 3.0, "chunk_size": 512},
                "full_epoch_001",
            )

    def test_rejects_the_old_accumulation_value(self) -> None:
        config = approved_config()
        config["gradient_accumulation_steps"] = 4
        with self.assertRaisesRegex(RuntimeError, "gradient_accumulation_steps"):
            validate_training_contract(
                config,
                {"gamma": 2.0, "item_weight": 3.0, "chunk_size": 512},
                "full_epoch_001",
            )

    def test_rejects_a_gate_with_smaller_chunk_override(self) -> None:
        config = approved_config()
        config["cutoff_len"] = 512
        config["max_steps"] = 1
        with self.assertRaisesRegex(RuntimeError, "custom_loss.chunk_size"):
            validate_training_contract(
                config,
                {"gamma": 2.0, "item_weight": 3.0, "chunk_size": 128},
                "gate_00512",
            )


if __name__ == "__main__":
    unittest.main()
