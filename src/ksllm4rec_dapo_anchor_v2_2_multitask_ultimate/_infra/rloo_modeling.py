"""Single-policy-adapter model lifecycle for RLOO Spec V2.0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .rloo_contract import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    POLICY_ADAPTER,
)


@dataclass(frozen=True)
class TrainabilityReport:
    """Evidence that only the existing default LoRA tensors are trainable."""

    lora_tensor_count: int
    trainable_tensor_count: int
    trainable_numel: int
    frozen_numel: int


@dataclass(frozen=True)
class DropoutReport:
    """Runtime dropout values changed to zero."""

    module_count: int
    scalar_attribute_count: int
    peft_config_count: int


@dataclass(frozen=True)
class AdapterContractReport:
    """Frozen properties read from an adapter_config.json file."""

    rank: int
    alpha: int
    source_dropout: float
    target_modules: frozenset[str]


@dataclass
class PolicyModel:
    """One frozen base plus one continued, trainable ``default`` LoRA adapter."""

    model: nn.Module
    tokenizer: Any
    adapter: str = POLICY_ADAPTER
    trainability: TrainabilityReport | None = None
    dropout: DropoutReport | None = None

    def save_policy(self, output_dir: Path) -> tuple[Path, Path]:
        return save_policy_adapter(self.model, output_dir, adapter_name=self.adapter)


def configure_torch_runtime(config: Mapping[str, Any]) -> None:
    """Apply the frozen BF16/TF32 runtime flags instead of trusting defaults."""

    train = config.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("Missing train configuration mapping.")
    if train.get("bf16") is not True or train.get("tf32") is not True:
        raise ValueError("RLOO Spec V2.0 requires train.bf16=true and tf32=true.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _adapter_configs(model: nn.Module) -> Mapping[str, Any]:
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping):
        raise ValueError("Model has no PEFT adapter configuration mapping.")
    names = set(configs)
    if names != {POLICY_ADAPTER}:
        raise ValueError(
            "RLOO Spec V2.0 requires exactly one adapter named 'default'; "
            f"loaded={sorted(names)}."
        )
    return configs


def _is_policy_lora_parameter(name: str, adapter_name: str = POLICY_ADAPTER) -> bool:
    parts = name.split(".")
    return adapter_name in parts and any(part.startswith("lora_") for part in parts)


def enforce_policy_trainability(
    model: nn.Module,
    *,
    adapter_name: str = POLICY_ADAPTER,
) -> TrainabilityReport:
    """Freeze base/embedding/head/bias and train only existing LoRA A/B tensors."""

    if adapter_name != POLICY_ADAPTER:
        raise ValueError("RLOO Spec V2.0 requires adapter_name='default'.")
    _adapter_configs(model)

    lora_parameters: list[tuple[str, nn.Parameter]] = []
    frozen_numel = 0
    for name, parameter in model.named_parameters():
        is_policy_lora = _is_policy_lora_parameter(name, adapter_name)
        parameter.requires_grad_(is_policy_lora)
        if is_policy_lora:
            lora_parameters.append((name, parameter))
        else:
            frozen_numel += parameter.numel()

    if not lora_parameters:
        raise RuntimeError("No default-adapter LoRA parameters were found.")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name for name, _ in trainable if not _is_policy_lora_parameter(name)
    ]
    if unexpected or len(trainable) != len(lora_parameters):
        raise RuntimeError(
            "Single-adapter trainability invariant failed: "
            f"unexpected_trainable={unexpected}."
        )
    return TrainabilityReport(
        lora_tensor_count=len(lora_parameters),
        trainable_tensor_count=len(trainable),
        trainable_numel=sum(parameter.numel() for _, parameter in trainable),
        frozen_numel=frozen_numel,
    )


def disable_all_dropout(model: nn.Module) -> tuple[int, int]:
    """Set module and scalar runtime dropout fields to zero."""

    changed_modules = 0
    changed_attributes = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            changed_modules += 1
        for name, value in vars(module).items():
            if "dropout" in name.lower() and isinstance(value, float) and value != 0.0:
                setattr(module, name, 0.0)
                changed_attributes += 1
    return changed_modules, changed_attributes


def zero_peft_config_dropout(model: nn.Module) -> int:
    """Make the one serialized PEFT configuration record dropout=0."""

    configs = _adapter_configs(model)
    config = configs[POLICY_ADAPTER]
    if not hasattr(config, "lora_dropout"):
        return 0
    if float(config.lora_dropout) == LORA_DROPOUT:
        return 0
    config.lora_dropout = LORA_DROPOUT
    return 1


def disable_and_record_dropout(model: nn.Module) -> DropoutReport:
    modules, attributes = disable_all_dropout(model)
    configs = zero_peft_config_dropout(model)
    return DropoutReport(
        module_count=modules,
        scalar_attribute_count=attributes,
        peft_config_count=configs,
    )


def validate_adapter_contract(path: Path) -> AdapterContractReport:
    """Validate rank/alpha/targets without changing the downloaded adapter."""

    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {directory}")
    weights_path = directory / "adapter_model.safetensors"
    config_path = directory / "adapter_config.json"
    for required in (weights_path, config_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty adapter file: {required}")

    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot parse adapter config: {config_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("adapter_config.json must contain one JSON object.")
    required_keys = {"r", "lora_alpha", "lora_dropout", "target_modules"}
    missing = sorted(required_keys - set(value))
    if missing:
        raise ValueError(f"Adapter config is missing keys: {missing}")

    targets = frozenset(str(name) for name in value["target_modules"])
    rank = int(value["r"])
    alpha = int(value["lora_alpha"])
    dropout = float(value["lora_dropout"])
    mismatches: dict[str, object] = {}
    if rank != LORA_RANK:
        mismatches["r"] = rank
    if alpha != LORA_ALPHA:
        mismatches["lora_alpha"] = alpha
    if targets != LORA_TARGET_MODULES:
        mismatches["target_modules"] = sorted(targets)
    if dropout not in (0.0, 0.05):
        mismatches["lora_dropout"] = dropout
    if mismatches:
        raise ValueError(f"Adapter differs from frozen r64 contract: {mismatches}")
    return AdapterContractReport(
        rank=rank,
        alpha=alpha,
        source_dropout=dropout,
        target_modules=targets,
    )


def _existing_directory(value: Any, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def _model_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    section = config.get("model")
    if not isinstance(section, Mapping):
        raise ValueError("Missing model configuration mapping.")
    required = {
        "base_model",
        "sft_adapter",
        "tokenizer",
        "policy_adapter",
        "attention",
        "dtype",
        "disable_dropout",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
    }
    missing = sorted(required - set(section))
    if missing:
        raise ValueError(f"Missing model configuration keys: {missing}")
    forbidden = sorted(key for key in section if "reference" in str(key).lower())
    if forbidden:
        raise ValueError(f"Reference model fields are forbidden: {forbidden}")
    return section


def load_policy_model(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "cuda:0",
    policy_adapter_path: Path | None = None,
) -> PolicyModel:
    """Load the local base and continue the one existing r64 SFT adapter."""

    configure_torch_runtime(config)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = _model_section(config)
    train_config = config.get("train")
    if not isinstance(train_config, Mapping):
        raise ValueError("Missing train configuration mapping.")
    expected_model_values = {
        "policy_adapter": POLICY_ADAPTER,
        "dtype": "bfloat16",
        "disable_dropout": True,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
    }
    mismatches = {
        key: value
        for key, value in expected_model_values.items()
        if model_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Model config differs from r64 policy contract: {mismatches}")

    base_path = _existing_directory(model_config["base_model"], "Base model")
    initial_adapter = _existing_directory(model_config["sft_adapter"], "SFT adapter")
    policy_path = (
        _existing_directory(policy_adapter_path, "Recovery policy adapter")
        if policy_adapter_path is not None
        else initial_adapter
    )
    tokenizer_path = _existing_directory(model_config["tokenizer"], "Tokenizer")
    validate_adapter_contract(policy_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention"]),
    )
    gradient_checkpointing = train_config.get("gradient_checkpointing")
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("train.gradient_checkpointing must be a boolean.")
    if gradient_checkpointing:
        base.config.use_cache = False
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model = PeftModel.from_pretrained(
        base,
        policy_path,
        adapter_name=POLICY_ADAPTER,
        is_trainable=True,
    )
    model.to(device)
    if gradient_checkpointing:
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if not callable(enable_input_grads):
            raise RuntimeError(
                "Gradient checkpointing requires enable_input_require_grads()."
            )
        enable_input_grads()
    model.train()
    dropout = disable_and_record_dropout(model)
    trainability = enforce_policy_trainability(model)
    return PolicyModel(
        model=model,
        tokenizer=tokenizer,
        trainability=trainability,
        dropout=dropout,
    )


def save_policy_adapter(
    model: nn.Module,
    output_dir: Path,
    *,
    adapter_name: str = POLICY_ADAPTER,
) -> tuple[Path, Path]:
    """Save only the default adapter and prove its serialized dropout is zero."""

    if adapter_name != POLICY_ADAPTER:
        raise ValueError("Root adapter output requires adapter_name='default'.")
    disable_and_record_dropout(model)
    enforce_policy_trainability(model, adapter_name=adapter_name)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    saver = getattr(model, "save_pretrained", None)
    if not callable(saver):
        raise TypeError("Model does not provide save_pretrained().")
    saver(
        str(output),
        safe_serialization=True,
        selected_adapters=[adapter_name],
        save_embedding_layers=False,
    )

    weights_path = output / "adapter_model.safetensors"
    config_path = output / "adapter_config.json"
    for path in (weights_path, config_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Adapter save did not create a non-empty {path}.")
    saved_contract = validate_adapter_contract(output)
    if saved_contract.source_dropout != LORA_DROPOUT:
        raise RuntimeError("Saved adapter_config.json must contain lora_dropout=0.0.")
    if any(path.name == "reference" for path in output.iterdir()):
        raise RuntimeError("Reference adapter output is forbidden.")
    return weights_path, config_path
