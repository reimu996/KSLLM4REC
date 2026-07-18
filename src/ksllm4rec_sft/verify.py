"""Post-training structural and runtime verification for the LoRA adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .contract import validate_training_contract
from .data import sha256_file
from .fingerprint import implementation_fingerprint
from .integrity import (
    artifact_identity,
    require_matching_artifact_identity,
    validate_profile_artifact_identity,
    verify_output_artifacts,
)
from .profiles import BASELINE_PROFILE, SFTProfile, get_profile


EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
EXPECTED_LAYERS = 28


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required training artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_full_manifest(
    log_root: Path,
    output_dir: Path,
    profile: str | SFTProfile = BASELINE_PROFILE,
) -> tuple[Path, dict[str, Any]]:
    selected = get_profile(profile)
    expected_output = output_dir.resolve()
    matches: list[tuple[str, Path, dict[str, Any]]] = []
    for path in log_root.glob("*/manifest.json"):
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        manifest_profile = manifest.get("profile")
        profile_matches = manifest_profile == selected.name or (
            selected is BASELINE_PROFILE and manifest_profile is None
        )
        if (
            manifest.get("stage") == selected.full_stage
            and profile_matches
            and manifest.get("status") == "passed"
            and Path(
                manifest.get("resolved_config", {}).get("output_dir", "")
            ).resolve()
            == expected_output
        ):
            matches.append((str(manifest.get("updated_at", "")), path, manifest))
    if not matches:
        raise RuntimeError(
            f"No passed {selected.full_stage} manifest for profile "
            f"{selected.name!r} bound to {expected_output} was found."
        )
    _, path, manifest = max(matches, key=lambda item: item[0])
    return path, manifest


def _verify_manifest_fingerprint(
    manifest: dict[str, Any],
    project_root: Path,
    profile: str | SFTProfile = BASELINE_PROFILE,
) -> dict[str, Any]:
    current_fingerprint = implementation_fingerprint(project_root, profile)
    manifest_fingerprint = manifest.get("implementation_fingerprint", {})
    if manifest_fingerprint.get("sha256") != current_fingerprint["sha256"]:
        raise RuntimeError(
            "Full-run implementation fingerprint does not match the current controlled implementation: "
            f"manifest={manifest_fingerprint.get('sha256')!r}, "
            f"current={current_fingerprint['sha256']!r}"
        )
    return current_fingerprint


def verify_adapter_files(
    output_dir: Path,
    log_root: Path,
    project_root: Path,
    max_reserved_gib: float = 20.0,
    *,
    profile: str | SFTProfile = BASELINE_PROFILE,
    expected_input_integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    selected = get_profile(profile)
    if selected is not BASELINE_PROFILE and expected_input_integrity is None:
        raise RuntimeError(
            f"Profile {selected.name!r} requires an artifact lock identity for verification."
        )
    if expected_input_integrity is not None:
        validate_profile_artifact_identity(expected_input_integrity, selected)
    output_dir = output_dir.resolve()
    manifest_path, manifest = _latest_full_manifest(log_root, output_dir, selected)
    manifest_integrity = manifest.get("input_integrity", {})
    if expected_input_integrity is not None:
        require_matching_artifact_identity(manifest_integrity, expected_input_integrity)
    if selected.name != "baseline":
        from .checkpoint import validate_run_binding

        run_identity = manifest.get("run_identity")
        if not isinstance(run_identity, dict):
            raise RuntimeError("Frontier full-run manifest has no run identity.")
        if expected_input_integrity is None:
            raise RuntimeError(
                "Frontier verification requires an artifact lock identity."
            )
        if run_identity.get("input_artifact_identity") != artifact_identity(
            expected_input_integrity
        ):
            raise RuntimeError(
                "Frontier run identity does not match the artifact lock."
            )
        if (
            run_identity.get("profile") != selected.name
            or run_identity.get("stage") != selected.full_stage
            or run_identity.get("output_dir") != str(output_dir)
        ):
            raise RuntimeError("Frontier run identity profile/stage/output is invalid.")
        validate_run_binding(output_dir, expected_run_identity=run_identity)
    validate_training_contract(
        manifest.get("resolved_config", {}),
        manifest.get("custom_loss", {}),
        selected.full_stage,
        profile=selected,
    )
    current_fingerprint = _verify_manifest_fingerprint(manifest, project_root, selected)
    verified_outputs = verify_output_artifacts(
        output_dir, manifest.get("output_artifacts", {})
    )
    adapter_config_path = output_dir / "adapter_config.json"
    adapter_weights_path = output_dir / "adapter_model.safetensors"
    trainer_state_path = output_dir / "trainer_state.json"
    train_results_path = output_dir / "train_results.json"
    for path in (
        adapter_config_path,
        adapter_weights_path,
        trainer_state_path,
        train_results_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required training artifact is missing: {path}")

    adapter_config = _read_json(adapter_config_path)
    if int(adapter_config.get("r", -1)) != 32:
        raise RuntimeError(f"Unexpected LoRA rank: {adapter_config.get('r')!r}")
    if int(adapter_config.get("lora_alpha", -1)) != 32:
        raise RuntimeError(
            f"Unexpected LoRA alpha: {adapter_config.get('lora_alpha')!r}"
        )
    if not math.isclose(
        float(adapter_config.get("lora_dropout", -1)), 0.05, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"Unexpected LoRA dropout: {adapter_config.get('lora_dropout')!r}"
        )
    actual_targets = set(adapter_config.get("target_modules") or [])
    if actual_targets != EXPECTED_TARGET_MODULES:
        raise RuntimeError(f"Unexpected LoRA target modules: {sorted(actual_targets)}")

    a_keys: list[str] = []
    b_keys: list[str] = []
    nonfinite: list[str] = []
    b_squared_norm = 0.0
    with safe_open(adapter_weights_path, framework="pt", device="cpu") as handle:
        tensor_keys = list(handle.keys())
        for key in tensor_keys:
            tensor = handle.get_tensor(key)
            if not bool(torch.isfinite(tensor).all()):
                nonfinite.append(key)
            if ".lora_A." in key:
                a_keys.append(key)
            if ".lora_B." in key:
                b_keys.append(key)
                b_squared_norm += float(tensor.float().square().sum().item())
    expected_per_factor = EXPECTED_LAYERS * len(EXPECTED_TARGET_MODULES)
    if len(a_keys) != expected_per_factor or len(b_keys) != expected_per_factor:
        raise RuntimeError(
            f"Expected {expected_per_factor} LoRA A and B tensors, got A={len(a_keys)}, B={len(b_keys)}"
        )
    if nonfinite:
        raise FloatingPointError(
            f"Adapter contains non-finite tensors: {nonfinite[:5]}"
        )
    if not b_squared_norm > 0.0:
        raise RuntimeError(
            "All LoRA B tensors are still zero; no effective update was learned."
        )

    trainer_state = _read_json(trainer_state_path)
    train_results = _read_json(train_results_path)
    if float(trainer_state.get("epoch", 0.0)) < 0.999:
        raise RuntimeError(
            f"Training did not complete one epoch: epoch={trainer_state.get('epoch')!r}"
        )
    if int(trainer_state.get("global_step", 0)) < 1:
        raise RuntimeError("Training did not complete an optimizer step.")
    train_loss = float(train_results.get("train_loss", float("nan")))
    if not math.isfinite(train_loss):
        raise FloatingPointError(f"Non-finite train_loss: {train_loss}")

    result = manifest.get("result", {})
    if int(result.get("fallback_count", -1)) != 0:
        raise RuntimeError(
            f"Focal loss fallback count is not zero: {result.get('fallback_count')!r}"
        )
    peak_reserved = float(manifest.get("peak_memory_reserved_gib", float("inf")))
    if peak_reserved > max_reserved_gib:
        raise RuntimeError(
            f"Peak reserved memory {peak_reserved:.3f} GiB exceeds {max_reserved_gib:.3f} GiB"
        )

    return {
        "adapter_config": str(adapter_config_path),
        "profile": selected.name,
        "input_artifact_identity": (
            artifact_identity(manifest_integrity) if manifest_integrity else None
        ),
        "adapter_weights": str(adapter_weights_path),
        "adapter_sha256": sha256_file(adapter_weights_path),
        "adapter_bytes": adapter_weights_path.stat().st_size,
        "verified_output_artifacts": verified_outputs,
        "tensor_count": len(tensor_keys),
        "lora_a_tensors": len(a_keys),
        "lora_b_tensors": len(b_keys),
        "lora_b_l2_norm": math.sqrt(b_squared_norm),
        "trainer_state": {
            "epoch": trainer_state["epoch"],
            "global_step": trainer_state["global_step"],
            "train_loss": train_loss,
        },
        "full_manifest": str(manifest_path.resolve()),
        "implementation_fingerprint": current_fingerprint,
        "peak_memory_reserved_gib": peak_reserved,
        "fallback_count": result["fallback_count"],
    }


def verify_adapter_runtime(
    model_path: Path,
    adapter_path: Path,
    *,
    min_free_gib: float = 20.5,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the adapter runtime check.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Runtime-check GPU gate failed: {free_gib:.3f} GiB free, need {min_free_gib:.3f} GiB."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    prompt = "<|im_start|>user\n请根据用户历史推荐下一个内容。<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    with torch.inference_mode():
        with model.disable_adapter():
            base_logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
        adapter_logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=8, use_cache=True
        )
    if not bool(torch.isfinite(adapter_logits).all()):
        raise FloatingPointError("Adapter runtime logits contain NaN or Inf.")
    max_abs_logit_delta = float((adapter_logits - base_logits).abs().max().item())
    if not max_abs_logit_delta > 0.0:
        raise RuntimeError(
            "Loaded adapter did not change the checked next-token logits."
        )
    continuation = tokenizer.decode(
        generated[0, inputs["input_ids"].size(1) :], skip_special_tokens=False
    )
    return {
        "gpu_free_gib_before": free_gib,
        "gpu_total_gib": total_bytes / 1024**3,
        "max_abs_logit_delta": max_abs_logit_delta,
        "generated_token_ids": generated[0, inputs["input_ids"].size(1) :].tolist(),
        "generated_text": continuation,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
