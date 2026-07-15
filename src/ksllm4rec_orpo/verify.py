"""Structural, numerical, and independent-load verification for both epochs."""

from __future__ import annotations

import gc
import json
import math
import traceback
from pathlib import Path
from typing import Any

from ksllm4rec_sft.checkpoint import validate_checkpoint
from ksllm4rec_sft.data import sha256_file
from ksllm4rec_sft.integrity import verify_environment_lock
from ksllm4rec_sft.manifest import atomic_write_json, now_iso

from .contract import EXPECTED_TARGETS, FULL_STAGE, validate_training_contract
from .fingerprint import implementation_fingerprint
from .integrity import verify_training_lock


EXPECTED_LAYERS = 28


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_full_manifest(log_root: Path, output_dir: Path) -> tuple[Path, dict]:
    expected = output_dir.resolve()
    matches: list[tuple[str, Path, dict]] = []
    for path in log_root.glob("*/manifest.json"):
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        resolved_output = Path(
            manifest.get("resolved_config", {}).get("output_dir", "")
        ).resolve()
        if (
            manifest.get("stage") == FULL_STAGE
            and manifest.get("status") == "passed"
            and resolved_output == expected
        ):
            matches.append((str(manifest.get("updated_at", "")), path, manifest))
    if not matches:
        raise RuntimeError(f"No passed full ORPO manifest is bound to {expected}.")
    _, path, manifest = max(matches, key=lambda item: item[0])
    return path, manifest


def _verify_snapshot(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    expected = manifest.get("output_artifacts", {})
    names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
        "train_results.json",
    }
    if set(expected) != names:
        raise RuntimeError(f"Unexpected full output snapshot keys: {sorted(expected)}")
    result: dict[str, Any] = {}
    for name in sorted(names):
        path = output_dir / name
        entry = expected[name]
        if not path.is_file():
            raise FileNotFoundError(f"Full output is missing {path}")
        actual = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        if actual["size"] != int(entry["size"]) or actual["sha256"] != entry["sha256"]:
            raise RuntimeError(f"Full output changed after training: {path}")
        result[name] = actual
    return result


def _adapter_summary(adapter_dir: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    config = _read_json(config_path)
    if int(config.get("r", -1)) != 32 or int(config.get("lora_alpha", -1)) != 32:
        raise RuntimeError(f"Unexpected LoRA rank/alpha in {adapter_dir}")
    if not math.isclose(float(config.get("lora_dropout", -1)), 0.0, abs_tol=1e-12):
        raise RuntimeError(f"Unexpected LoRA dropout in {adapter_dir}")
    if set(config.get("target_modules") or ()) != EXPECTED_TARGETS:
        raise RuntimeError(f"Unexpected LoRA targets in {adapter_dir}")
    a_keys: list[str] = []
    b_keys: list[str] = []
    b_square = 0.0
    tensor_count = 0
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            tensor = handle.get_tensor(key)
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"Non-finite adapter tensor: {key}")
            if ".lora_A." in key:
                a_keys.append(key)
            if ".lora_B." in key:
                b_keys.append(key)
                b_square += float(tensor.float().square().sum().item())
    expected = EXPECTED_LAYERS * len(EXPECTED_TARGETS)
    if len(a_keys) != expected or len(b_keys) != expected:
        raise RuntimeError(
            f"Expected {expected} A/B tensors in {adapter_dir}, "
            f"got A={len(a_keys)}, B={len(b_keys)}."
        )
    if b_square <= 0.0:
        raise RuntimeError(f"All LoRA B tensors remain zero in {adapter_dir}")
    return {
        "path": str(adapter_dir.resolve()),
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "weights_size": weights_path.stat().st_size,
        "tensor_count": tensor_count,
        "lora_a_tensors": len(a_keys),
        "lora_b_tensors": len(b_keys),
        "lora_b_l2_norm": math.sqrt(b_square),
    }


def _runtime_check(
    model_path: Path, adapters: list[Path], *, min_free_gib: float
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for independent adapter loading checks.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    prompt = (
        "<|im_start|>user\n请根据用户历史推荐下一个内容。/no_think"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs_cpu = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    results: list[dict[str, Any]] = []
    for adapter_path in adapters:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
        if free_gib < min_free_gib:
            raise RuntimeError(
                f"Runtime GPU gate failed: {free_gib:.3f} GiB free, "
                f"need {min_free_gib:.3f}."
            )
        torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda:0")
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        model.eval()
        inputs = {key: value.to("cuda:0") for key, value in inputs_cpu.items()}
        with torch.inference_mode():
            with model.disable_adapter():
                base_logits = model(**inputs, use_cache=False).logits[:, -1].float()
            adapter_logits = model(**inputs, use_cache=False).logits[:, -1].float()
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=16, use_cache=True
            )
        if not bool(torch.isfinite(adapter_logits).all()):
            raise FloatingPointError(f"Non-finite runtime logits for {adapter_path}")
        delta = float((adapter_logits - base_logits).abs().max().item())
        if delta <= 0.0:
            raise RuntimeError(f"Adapter does not change logits: {adapter_path}")
        continuation = generated[0, inputs["input_ids"].size(1) :]
        results.append(
            {
                "adapter_path": str(adapter_path.resolve()),
                "gpu_free_gib_before": free_gib,
                "gpu_total_gib": total_bytes / 1024**3,
                "max_abs_logit_delta": delta,
                "generated_token_ids": continuation.tolist(),
                "generated_text": tokenizer.decode(
                    continuation, skip_special_tokens=False
                ),
                "peak_memory_reserved_gib": torch.cuda.max_memory_reserved()
                / 1024**3,
            }
        )
        del model, base, adapter_logits, base_logits, generated, inputs
        gc.collect()
        torch.cuda.empty_cache()
    return results


def verify_full_run(
    model_path: Path,
    output_dir: Path,
    log_root: Path,
    report_path: Path,
    artifact_lock: Path,
    environment_lock: Path,
    *,
    min_free_gib: float = 20.5,
    max_reserved_gib: float = 20.0,
) -> dict[str, Any]:
    import llamafactory

    report: dict[str, Any] = {"status": "running", "created_at": now_iso()}
    atomic_write_json(report_path, report)
    try:
        output_dir = output_dir.resolve()
        manifest_path, manifest = _latest_full_manifest(log_root, output_dir)
        validate_training_contract(
            manifest["resolved_config"], manifest["custom_orpo"], FULL_STAGE
        )
        current_fingerprint = implementation_fingerprint(
            Path(__file__).resolve().parents[2]
        )
        if manifest.get("implementation_fingerprint", {}).get(
            "sha256"
        ) != current_fingerprint["sha256"]:
            raise RuntimeError("Full-run implementation fingerprint no longer matches.")
        peak_reserved = float(manifest.get("peak_memory_reserved_gib", float("inf")))
        if peak_reserved > max_reserved_gib:
            raise RuntimeError(
                f"Full run reserved {peak_reserved:.3f} GiB > {max_reserved_gib:.3f}."
            )
        result = manifest.get("result", {})
        if result.get("reference_model_used") is not False:
            raise RuntimeError("Full run unexpectedly used a reference model.")
        if float(result.get("completed_epoch", 0.0)) < 1.999:
            raise RuntimeError("Full ORPO run did not finish two epochs.")

        checkpoint_paths = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
        if len(checkpoint_paths) != 2:
            raise RuntimeError(
                f"Expected exactly two epoch checkpoints, got {len(checkpoint_paths)}."
            )
        checkpoints: list[dict[str, Any]] = []
        for index, path in enumerate(checkpoint_paths, start=1):
            checkpoint = validate_checkpoint(path)
            state = _read_json(path / "trainer_state.json")
            epoch = float(state.get("epoch", 0.0))
            if abs(epoch - index) > 0.01:
                raise RuntimeError(
                    f"Checkpoint {path.name} epoch={epoch}, expected {index}."
                )
            checkpoint["epoch"] = epoch
            checkpoint["adapter"] = _adapter_summary(path)
            checkpoints.append(checkpoint)
        if checkpoints[0]["adapter"]["weights_sha256"] == checkpoints[1]["adapter"][
            "weights_sha256"
        ]:
            raise RuntimeError("Epoch 1 and epoch 2 adapter weights are identical.")

        report.update(
            status="passed",
            updated_at=now_iso(),
            manifest_path=str(manifest_path.resolve()),
            input_integrity=verify_training_lock(
                artifact_lock,
                llamafactory_module_file=Path(llamafactory.__file__),
            ),
            environment_integrity=verify_environment_lock(environment_lock),
            implementation_fingerprint=current_fingerprint,
            output_snapshot=_verify_snapshot(manifest, output_dir),
            final_adapter=_adapter_summary(output_dir),
            checkpoints=checkpoints,
            runtime=_runtime_check(
                model_path, checkpoint_paths, min_free_gib=min_free_gib
            ),
            peak_memory_reserved_gib=peak_reserved,
            reference_model_used=False,
            completed_epoch=result["completed_epoch"],
            optimizer_steps=result["optimizer_steps"],
        )
        atomic_write_json(report_path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            updated_at=now_iso(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        atomic_write_json(report_path, report)
        raise
