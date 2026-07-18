"""Single-base, dual-adapter model lifecycle for GRPO Spec V3.1."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import nn


POLICY_ADAPTER = "default"
REFERENCE_ADAPTER = "reference"
_LORA_PARAMETER_PREFIX = "lora_"


@dataclass(frozen=True)
class TrainabilityReport:
    """Result of enforcing the policy-only LoRA trainability invariant."""

    policy_tensor_count: int
    reference_tensor_count: int
    trainable_tensor_count: int
    trainable_numel: int


@dataclass(frozen=True)
class DropoutReport:
    """Runtime dropout values that were changed to zero."""

    module_count: int
    scalar_attribute_count: int


@dataclass
class DualAdapterModel:
    """One base model with a trainable policy and a frozen reference adapter."""

    model: nn.Module
    tokenizer: Any
    policy_adapter: str = POLICY_ADAPTER
    reference_adapter: str = REFERENCE_ADAPTER
    trainability: TrainabilityReport | None = None
    dropout: DropoutReport | None = None

    def activate_policy(self) -> TrainabilityReport:
        self.trainability = activate_adapter(
            self.model,
            self.policy_adapter,
            policy_adapter=self.policy_adapter,
            reference_adapter=self.reference_adapter,
        )
        return self.trainability

    def activate_reference(self) -> TrainabilityReport:
        self.trainability = activate_adapter(
            self.model,
            self.reference_adapter,
            policy_adapter=self.policy_adapter,
            reference_adapter=self.reference_adapter,
        )
        return self.trainability

    @contextmanager
    def use_policy(self) -> Iterator[nn.Module]:
        with adapter_scope(
            self.model,
            self.policy_adapter,
            policy_adapter=self.policy_adapter,
            reference_adapter=self.reference_adapter,
        ) as active_model:
            yield active_model

    @contextmanager
    def use_reference(self) -> Iterator[nn.Module]:
        with adapter_scope(
            self.model,
            self.reference_adapter,
            policy_adapter=self.policy_adapter,
            reference_adapter=self.reference_adapter,
        ) as active_model:
            yield active_model

    def save_policy(self, output_dir: Path) -> tuple[Path, Path]:
        return save_policy_adapter(
            self.model,
            output_dir,
            policy_adapter=self.policy_adapter,
            reference_adapter=self.reference_adapter,
        )


def _adapter_parameter(name: str, adapter_name: str) -> bool:
    parts = name.split(".")
    return adapter_name in parts and any(
        part.startswith(_LORA_PARAMETER_PREFIX) for part in parts
    )


def _require_adapter(model: nn.Module, adapter_name: str) -> None:
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or adapter_name not in configs:
        raise ValueError(f"Adapter {adapter_name!r} is not loaded.")


def enforce_policy_trainability(
    model: nn.Module,
    *,
    policy_adapter: str = POLICY_ADAPTER,
    reference_adapter: str = REFERENCE_ADAPTER,
) -> TrainabilityReport:
    """Freeze everything except LoRA tensors belonging to the policy adapter."""

    if policy_adapter == reference_adapter:
        raise ValueError("Policy and reference adapter names must differ.")
    _require_adapter(model, policy_adapter)
    _require_adapter(model, reference_adapter)

    policy_parameters: list[tuple[str, nn.Parameter]] = []
    reference_parameters: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        is_policy = _adapter_parameter(name, policy_adapter)
        is_reference = _adapter_parameter(name, reference_adapter)
        parameter.requires_grad_(is_policy)
        if is_policy:
            policy_parameters.append((name, parameter))
        if is_reference:
            reference_parameters.append((name, parameter))

    if not policy_parameters:
        raise RuntimeError(
            f"No LoRA parameters found for policy adapter {policy_adapter!r}."
        )
    if not reference_parameters:
        raise RuntimeError(
            f"No LoRA parameters found for reference adapter {reference_adapter!r}."
        )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name for name, _ in trainable if not _adapter_parameter(name, policy_adapter)
    ]
    frozen_policy = [
        name for name, parameter in policy_parameters if not parameter.requires_grad
    ]
    trainable_reference = [
        name for name, parameter in reference_parameters if parameter.requires_grad
    ]
    if unexpected or frozen_policy or trainable_reference:
        raise RuntimeError(
            "Policy-only trainability invariant failed: "
            f"unexpected={unexpected}, frozen_policy={frozen_policy}, "
            f"trainable_reference={trainable_reference}"
        )

    return TrainabilityReport(
        policy_tensor_count=len(policy_parameters),
        reference_tensor_count=len(reference_parameters),
        trainable_tensor_count=len(trainable),
        trainable_numel=sum(parameter.numel() for _, parameter in trainable),
    )


def _active_adapter(model: nn.Module) -> str:
    active = getattr(model, "active_adapter", None)
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple)) and len(active) == 1:
        return str(active[0])
    raise RuntimeError(f"Expected exactly one active adapter, got {active!r}.")


def activate_adapter(
    model: nn.Module,
    adapter_name: str,
    *,
    policy_adapter: str = POLICY_ADAPTER,
    reference_adapter: str = REFERENCE_ADAPTER,
) -> TrainabilityReport:
    """Activate one adapter, then undo PEFT's implicit trainability changes."""

    if adapter_name not in (policy_adapter, reference_adapter):
        raise ValueError(f"Unsupported adapter {adapter_name!r}.")
    _require_adapter(model, adapter_name)
    setter = getattr(model, "set_adapter", None)
    if not callable(setter):
        raise TypeError("Model does not provide set_adapter().")
    setter(adapter_name)
    if _active_adapter(model) != adapter_name:
        raise RuntimeError(f"Adapter activation did not select {adapter_name!r}.")
    return enforce_policy_trainability(
        model,
        policy_adapter=policy_adapter,
        reference_adapter=reference_adapter,
    )


@contextmanager
def adapter_scope(
    model: nn.Module,
    adapter_name: str,
    *,
    policy_adapter: str = POLICY_ADAPTER,
    reference_adapter: str = REFERENCE_ADAPTER,
) -> Iterator[nn.Module]:
    """Temporarily activate an adapter and restore the previous one on exit."""

    previous = _active_adapter(model)
    activate_adapter(
        model,
        adapter_name,
        policy_adapter=policy_adapter,
        reference_adapter=reference_adapter,
    )
    try:
        yield model
    finally:
        activate_adapter(
            model,
            previous,
            policy_adapter=policy_adapter,
            reference_adapter=reference_adapter,
        )


def disable_all_dropout(model: nn.Module) -> DropoutReport:
    """Set module dropout and numeric runtime dropout attributes to zero."""

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
    return DropoutReport(
        module_count=changed_modules,
        scalar_attribute_count=changed_attributes,
    )


def zero_peft_config_dropout(model: nn.Module) -> int:
    """Keep serialized PEFT configs consistent with zero runtime dropout."""

    changed = 0
    configs = getattr(model, "peft_config", {})
    for config in configs.values():
        if hasattr(config, "lora_dropout") and config.lora_dropout != 0.0:
            config.lora_dropout = 0.0
            changed += 1
    return changed


def _model_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("Missing model configuration mapping.")
    required = (
        "base_model",
        "sft_adapter",
        "tokenizer",
        "policy_adapter",
        "reference_adapter",
        "attention",
        "dtype",
        "disable_dropout",
    )
    missing = [key for key in required if key not in model_config]
    if missing:
        raise ValueError(f"Missing model configuration keys: {missing}")
    return model_config


def _existing_directory(value: Any, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def load_dual_adapter_model(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "cuda:0",
    policy_adapter_path: Path | None = None,
) -> DualAdapterModel:
    """Load one local base and two copies of the approved SFT LoRA adapter."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = _model_section(config)
    train_config = config.get("train")
    if not isinstance(train_config, Mapping):
        raise ValueError("Missing train configuration mapping.")

    base_path = _existing_directory(model_config["base_model"], "Base model")
    adapter_path = _existing_directory(model_config["sft_adapter"], "SFT adapter")
    policy_path = (
        _existing_directory(policy_adapter_path, "Recovery policy adapter")
        if policy_adapter_path is not None
        else adapter_path
    )
    tokenizer_path = _existing_directory(model_config["tokenizer"], "Tokenizer")
    for source_path in {adapter_path, policy_path}:
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            if not (source_path / filename).is_file():
                raise FileNotFoundError(
                    f"Missing adapter file: {source_path / filename}"
                )

    policy_adapter = str(model_config["policy_adapter"])
    reference_adapter = str(model_config["reference_adapter"])
    if policy_adapter != POLICY_ADAPTER or reference_adapter != REFERENCE_ADAPTER:
        raise ValueError(
            "Spec V3.1 requires policy adapter 'default' and reference adapter "
            "'reference'."
        )
    if model_config["dtype"] != "bfloat16":
        raise ValueError("Spec V3.1 requires model.dtype=bfloat16.")
    if model_config["disable_dropout"] is not True:
        raise ValueError("Spec V3.1 requires model.disable_dropout=true.")
    gradient_checkpointing = train_config.get("gradient_checkpointing")
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("train.gradient_checkpointing must be a boolean.")

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
    if gradient_checkpointing:
        base.config.use_cache = False
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model = PeftModel.from_pretrained(
        base,
        policy_path,
        adapter_name=policy_adapter,
        is_trainable=True,
    )
    model.load_adapter(
        adapter_path,
        adapter_name=reference_adapter,
        is_trainable=False,
    )
    model.to(device)
    if gradient_checkpointing:
        input_grad_enabler = getattr(model, "enable_input_require_grads", None)
        if not callable(input_grad_enabler):
            raise RuntimeError(
                "Gradient checkpointing requires enable_input_require_grads()."
            )
        input_grad_enabler()
    model.train()

    dropout = disable_all_dropout(model)
    zero_peft_config_dropout(model)
    trainability = activate_adapter(
        model,
        policy_adapter,
        policy_adapter=policy_adapter,
        reference_adapter=reference_adapter,
    )
    return DualAdapterModel(
        model=model,
        tokenizer=tokenizer,
        policy_adapter=policy_adapter,
        reference_adapter=reference_adapter,
        trainability=trainability,
        dropout=dropout,
    )


def save_policy_adapter(
    model: nn.Module,
    output_dir: Path,
    *,
    policy_adapter: str = POLICY_ADAPTER,
    reference_adapter: str = REFERENCE_ADAPTER,
) -> tuple[Path, Path]:
    """Save only the policy adapter in submission-compatible root filenames."""

    if policy_adapter != POLICY_ADAPTER:
        raise ValueError(
            "Root-level PEFT output requires policy adapter name 'default'."
        )
    enforce_policy_trainability(
        model,
        policy_adapter=policy_adapter,
        reference_adapter=reference_adapter,
    )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = getattr(model, "save_pretrained", None)
    if not callable(saver):
        raise TypeError("Model does not provide save_pretrained().")
    saver(
        str(output_dir),
        safe_serialization=True,
        selected_adapters=[policy_adapter],
        save_embedding_layers=False,
    )

    weights_path = output_dir / "adapter_model.safetensors"
    config_path = output_dir / "adapter_config.json"
    for path in (weights_path, config_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Policy adapter save did not create {path}.")
    if (output_dir / reference_adapter).exists():
        raise RuntimeError("Reference adapter must not be written to policy output.")
    return weights_path, config_path
