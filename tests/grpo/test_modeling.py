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

from ksllm4rec_grpo.modeling import (
    DualAdapterModel,
    activate_adapter,
    adapter_scope,
    disable_all_dropout,
    enforce_policy_trainability,
    load_dual_adapter_model,
    save_policy_adapter,
    zero_peft_config_dropout,
)


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
        self.peft_config = {"default": object()}
        self.active_adapter = "default"
        self.input_grads_enabled = False
        self.load_calls = []
        self.save_calls = []

    def load_adapter(self, path, *, adapter_name, is_trainable):
        self.lora_A[adapter_name] = nn.Linear(3, 2, bias=False)
        self.lora_B[adapter_name] = nn.Linear(2, 3, bias=False)
        self.lora_dropout[adapter_name] = nn.Dropout(0.05)
        self.peft_config[adapter_name] = object()
        self.load_calls.append((Path(path), adapter_name, is_trainable))

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name
        # PEFT 0.18 makes the selected adapter trainable; production code must undo it.
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(f".{adapter_name}." in name)

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True

    def save_pretrained(self, directory: str, **kwargs) -> None:
        output = Path(directory)
        self.save_calls.append((output, kwargs))
        (output / "adapter_model.safetensors").write_bytes(b"toy-weights")
        (output / "adapter_config.json").write_text(
            json.dumps({"adapter": "default"}), encoding="utf-8"
        )


def make_dual_model() -> ToyPeftModel:
    model = ToyPeftModel(ToyBase())
    model.load_adapter(Path("/unused"), adapter_name="reference", is_trainable=False)
    return model


class AdapterTrainabilityTest(unittest.TestCase):
    def test_only_policy_lora_parameters_are_trainable(self) -> None:
        model = make_dual_model()
        for parameter in model.parameters():
            parameter.requires_grad_(True)

        report = enforce_policy_trainability(model)

        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {
                "lora_A.default.weight",
                "lora_B.default.weight",
            },
        )
        self.assertEqual(report.policy_tensor_count, 2)
        self.assertEqual(report.reference_tensor_count, 2)
        self.assertEqual(report.trainable_tensor_count, 2)

    def test_reference_activation_cannot_unfreeze_reference(self) -> None:
        model = make_dual_model()

        activate_adapter(model, "reference")

        self.assertEqual(model.active_adapter, "reference")
        for name, parameter in model.named_parameters():
            if ".reference." in name:
                self.assertFalse(parameter.requires_grad, name)
            elif ".default." in name and "lora_" in name:
                self.assertTrue(parameter.requires_grad, name)
            else:
                self.assertFalse(parameter.requires_grad, name)

    def test_scope_restores_previous_adapter_even_after_exception(self) -> None:
        model = make_dual_model()
        activate_adapter(model, "default")

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with adapter_scope(model, "reference"):
                self.assertEqual(model.active_adapter, "reference")
                raise RuntimeError("forward failed")

        self.assertEqual(model.active_adapter, "default")
        self.assertEqual(
            {
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            {"lora_A.default.weight", "lora_B.default.weight"},
        )


class DropoutAndSaveTest(unittest.TestCase):
    def test_all_runtime_dropout_is_zeroed(self) -> None:
        model = make_dual_model()

        report = disable_all_dropout(model)

        self.assertGreaterEqual(report.module_count, 3)
        self.assertGreaterEqual(report.scalar_attribute_count, 1)
        self.assertEqual(model.base_model.dropout.p, 0.0)
        self.assertEqual(model.base_model.attention_dropout, 0.0)
        self.assertEqual(model.lora_dropout["default"].p, 0.0)
        self.assertEqual(model.lora_dropout["reference"].p, 0.0)

    def test_save_writes_only_default_adapter_at_root(self) -> None:
        model = make_dual_model()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "epoch_001"
            weights, config = save_policy_adapter(model, output)

            self.assertEqual(weights, output.resolve() / "adapter_model.safetensors")
            self.assertEqual(config, output.resolve() / "adapter_config.json")
            self.assertFalse((output / "reference").exists())
            _, kwargs = model.save_calls[-1]
            self.assertEqual(kwargs["selected_adapters"], ["default"])
            self.assertTrue(kwargs["safe_serialization"])
        self.assertFalse(kwargs["save_embedding_layers"])

    def test_peft_config_dropout_is_zeroed(self) -> None:
        model = make_dual_model()
        model.peft_config = {
            "default": types.SimpleNamespace(lora_dropout=0.05),
            "reference": types.SimpleNamespace(lora_dropout=0.05),
        }
        self.assertEqual(zero_peft_config_dropout(model), 2)
        self.assertEqual(model.peft_config["default"].lora_dropout, 0.0)
        self.assertEqual(model.peft_config["reference"].lora_dropout, 0.0)


class LoaderTest(unittest.TestCase):
    def test_loader_uses_local_bf16_base_and_builds_dual_adapter(self) -> None:
        tokenizer_object = object()
        calls = {}

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
            adapter_path.mkdir()
            (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
            config = {
                "model": {
                    "base_model": str(base_path),
                    "sft_adapter": str(adapter_path),
                    "tokenizer": str(adapter_path),
                    "policy_adapter": "default",
                    "reference_adapter": "reference",
                    "attention": "flash_attention_2",
                    "dtype": "bfloat16",
                    "disable_dropout": True,
                },
                "train": {"gradient_checkpointing": True},
            }
            with patch.dict(
                sys.modules,
                {"transformers": transformers_module, "peft": peft_module},
            ):
                bundle = load_dual_adapter_model(config, device="cpu")

        self.assertIsInstance(bundle, DualAdapterModel)
        self.assertIs(bundle.tokenizer, tokenizer_object)
        _, tokenizer_kwargs = calls["tokenizer"]
        self.assertTrue(tokenizer_kwargs["local_files_only"])
        _, base_kwargs = calls["base"]
        self.assertIs(base_kwargs["torch_dtype"], torch.bfloat16)
        self.assertEqual(base_kwargs["attn_implementation"], "flash_attention_2")
        self.assertTrue(base_kwargs["local_files_only"])
        base = calls["policy"][0]
        self.assertFalse(base.config.use_cache)
        self.assertEqual(base.gradient_checkpointing_kwargs, {"use_reentrant": False})
        self.assertEqual(
            bundle.model.load_calls,
            [(adapter_path.resolve(), "reference", False)],
        )
        self.assertTrue(bundle.model.input_grads_enabled)
        self.assertTrue(bundle.model.training)
        self.assertEqual(bundle.model.active_adapter, "default")
        self.assertEqual(bundle.model.base_model.dropout.p, 0.0)
        self.assertEqual(bundle.model.lora_dropout["default"].p, 0.0)
        self.assertEqual(bundle.model.lora_dropout["reference"].p, 0.0)


if __name__ == "__main__":
    unittest.main()
