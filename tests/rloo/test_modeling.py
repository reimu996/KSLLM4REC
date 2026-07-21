from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from ksllm4rec_rloo.config import approved_config
from ksllm4rec_rloo.modeling import (
    PolicyModel,
    configure_torch_runtime,
    disable_and_record_dropout,
    enforce_policy_trainability,
    load_policy_model,
    save_policy_adapter,
    validate_adapter_contract,
)


TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class TorchRuntimeTest(unittest.TestCase):
    def test_tf32_is_applied_explicitly(self) -> None:
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        old_precision = torch.get_float32_matmul_precision()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            configure_torch_runtime(approved_config())
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
            torch.set_float32_matmul_precision(old_precision)


def adapter_config(dropout: float = 0.05) -> dict[str, object]:
    return {
        "r": 64,
        "lora_alpha": 64,
        "lora_dropout": dropout,
        "target_modules": sorted(TARGETS),
    }


class ToyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.dropout = nn.Dropout(0.25)
        self.attention_dropout = 0.15
        self.config = types.SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


class ToyPeftModel(nn.Module):
    def __init__(self, base: ToyBase) -> None:
        super().__init__()
        self.base_model = base
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.05)})
        self.peft_config = {
            "default": types.SimpleNamespace(**adapter_config())
        }
        self.input_grads_enabled = False
        self.save_calls: list[tuple[Path, dict[str, object]]] = []

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True

    def save_pretrained(self, directory: str, **kwargs) -> None:
        output = Path(directory)
        self.save_calls.append((output, kwargs))
        (output / "adapter_model.safetensors").write_bytes(b"toy-weights")
        config = adapter_config(self.peft_config["default"].lora_dropout)
        (output / "adapter_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )


def write_adapter(directory: Path, *, rank: int = 64, dropout: float = 0.05) -> None:
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    value = adapter_config(dropout)
    value["r"] = rank
    (directory / "adapter_config.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


class TrainabilityTest(unittest.TestCase):
    def test_only_default_lora_tensors_are_trainable(self) -> None:
        model = ToyPeftModel(ToyBase())
        for parameter in model.parameters():
            parameter.requires_grad_(True)

        report = enforce_policy_trainability(model)

        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {"lora_A.default.weight", "lora_B.default.weight"},
        )
        self.assertEqual(report.lora_tensor_count, 2)
        self.assertEqual(report.trainable_tensor_count, 2)
        self.assertGreater(report.frozen_numel, 0)

    def test_any_second_adapter_is_rejected(self) -> None:
        model = ToyPeftModel(ToyBase())
        model.peft_config["reference"] = types.SimpleNamespace(**adapter_config())
        with self.assertRaisesRegex(ValueError, "exactly one adapter"):
            enforce_policy_trainability(model)

    def test_runtime_and_serialized_dropout_are_zeroed(self) -> None:
        model = ToyPeftModel(ToyBase())
        report = disable_and_record_dropout(model)
        self.assertGreaterEqual(report.module_count, 2)
        self.assertGreaterEqual(report.scalar_attribute_count, 1)
        self.assertEqual(report.peft_config_count, 1)
        self.assertEqual(model.base_model.dropout.p, 0.0)
        self.assertEqual(model.base_model.attention_dropout, 0.0)
        self.assertEqual(model.lora_dropout["default"].p, 0.0)
        self.assertEqual(model.peft_config["default"].lora_dropout, 0.0)


class AdapterFilesTest(unittest.TestCase):
    def test_source_dropout_point_zero_five_and_resume_zero_are_both_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            resume = Path(directory) / "resume"
            write_adapter(source, dropout=0.05)
            write_adapter(resume, dropout=0.0)
            self.assertEqual(validate_adapter_contract(source).source_dropout, 0.05)
            self.assertEqual(validate_adapter_contract(resume).source_dropout, 0.0)

    def test_wrong_rank_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            write_adapter(adapter, rank=32)
            with self.assertRaisesRegex(ValueError, "frozen r64"):
                validate_adapter_contract(adapter)

    def test_save_has_only_default_adapter_and_dropout_zero(self) -> None:
        model = ToyPeftModel(ToyBase())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "epoch_001"
            weights, config = save_policy_adapter(model, output)
            saved = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(weights, output.resolve() / "adapter_model.safetensors")
            self.assertEqual(saved["lora_dropout"], 0.0)
            self.assertFalse((output / "reference").exists())
            _, kwargs = model.save_calls[-1]
            self.assertEqual(kwargs["selected_adapters"], ["default"])
            self.assertFalse(kwargs["save_embedding_layers"])


class LoaderTest(unittest.TestCase):
    def test_loader_continues_one_existing_adapter_without_reference_copy(self) -> None:
        tokenizer_object = object()
        calls: dict[str, object] = {}

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["tokenizer"] = (Path(path), kwargs)
                return tokenizer_object

        class FakeAutoModel:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["base"] = (Path(path), kwargs)
                return ToyBase()

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, base, path, **kwargs):
                calls["policy"] = (base, Path(path), kwargs)
                return ToyPeftModel(base)

        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = FakeAutoTokenizer
        transformers_module.AutoModelForCausalLM = FakeAutoModel
        peft_module = types.ModuleType("peft")
        peft_module.PeftModel = FakePeftModel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base"
            adapter_path = root / "adapter"
            base_path.mkdir()
            write_adapter(adapter_path)
            config = {
                "model": {
                    "base_model": str(base_path),
                    "sft_adapter": str(adapter_path),
                    "tokenizer": str(adapter_path),
                    "policy_adapter": "default",
                    "attention": "flash_attention_2",
                    "dtype": "bfloat16",
                    "disable_dropout": True,
                    "lora_rank": 64,
                    "lora_alpha": 64,
                    "lora_dropout": 0.0,
                },
                "train": {
                    "gradient_checkpointing": True,
                    "bf16": True,
                    "tf32": True,
                },
            }
            with patch.dict(
                sys.modules,
                {"transformers": transformers_module, "peft": peft_module},
            ):
                bundle = load_policy_model(config, device="cpu")

        self.assertIsInstance(bundle, PolicyModel)
        self.assertIs(bundle.tokenizer, tokenizer_object)
        _, policy_path, policy_kwargs = calls["policy"]
        self.assertEqual(policy_path, adapter_path.resolve())
        self.assertEqual(policy_kwargs["adapter_name"], "default")
        self.assertTrue(policy_kwargs["is_trainable"])
        self.assertNotIn("reference", bundle.model.peft_config)
        self.assertTrue(bundle.model.input_grads_enabled)
        self.assertFalse(calls["policy"][0].config.use_cache)
        self.assertEqual(bundle.model.lora_dropout["default"].p, 0.0)


if __name__ == "__main__":
    unittest.main()
