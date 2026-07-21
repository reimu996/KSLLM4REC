"""Hard pre-flight gates for the frozen reference-free RLOO run.

The validators in this module are intentionally pure: a report is accepted only
when its measured fields satisfy Spec V2.0 and it is bound to the exact runtime
signature and calibrated resolved contract.  CUDA work is delegated lazily to
``trainer`` so importing this module never loads a model or reserves GPU memory.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from . import contract
from .fingerprint import runtime_signature, validate_runtime_signature
from .integrity import canonical_sha256, sha256_file
from .objective import calibrate_anchor_lambda


GATE_SCHEMA_VERSION = 1
TOTAL_GROUP_VISITS = contract.EXPECTED_RECOMMEND_GROUPS * contract.EPOCHS
TIMING_SAFETY_FACTOR = 1.2


@dataclass(frozen=True)
class GateBundle:
    """All evidence that the formal trainer is allowed to consume."""

    runtime_signature: dict[str, Any]
    resolved_contract: dict[str, Any]
    lambda0: float
    rollout_chunk: int
    loss_chunk: int
    gt_chunk: int
    reports: dict[str, dict[str, Any]]


def expected_structure(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact shape/count contract checked by the structure gate."""

    return {
        "groups": int(config["data"]["groups"]),
        "positive_edges": int(config["data"]["positives"]),
        "trie_leaves": int(config["trie"]["unique_sids"]),
        "trie_a_nodes": contract.EXPECTED_DOMAIN_A_NODES,
        "trie_ab_nodes": contract.EXPECTED_DOMAIN_AB_NODES,
        "group_size": int(config["rollout"]["num_generations"]),
        "candidate_chunk_size": int(config["rollout"]["candidate_chunk_size"]),
        "loss_chunk_size": int(config["memory"]["loss_chunk_size"]),
        "gt_chunk_size": int(config["memory"]["gt_chunk_size"]),
        "epochs": int(config["train"]["epochs"]),
        "total_group_visits": int(config["data"]["groups"])
        * int(config["train"]["epochs"]),
    }


def expected_input_sha256() -> dict[str, str]:
    """Return the approved content identities that predate this implementation."""

    return {
        "base_model": contract.BASE_MODEL_SHA256,
        "sft_adapter": contract.SFT_ADAPTER_SHA256,
        "sft_adapter_config": contract.SFT_CONFIG_SHA256,
        "tokenizer": contract.TOKENIZER_SHA256,
        "source": contract.SOURCE_DATA_SHA256,
        "provenance": contract.PROVENANCE_SHA256,
        "groups": contract.GROUPS_SHA256,
        "trie_manifest": contract.TRIE_MANIFEST_SHA256,
        "fixed_probe": contract.FIXED_PROBE_SHA256,
    }


def _groups_file(path: Path) -> Path:
    path = Path(path)
    return path / "groups.jsonl" if path.is_dir() else path


def collect_input_sha256(
    config: Mapping[str, Any], groups_path: Path, trie_dir: Path
) -> dict[str, str]:
    """Hash the concrete files represented by :func:`expected_input_sha256`."""

    base = Path(config["model"]["base_model"])
    adapter = Path(config["model"]["sft_adapter"])
    tokenizer = Path(config["model"]["tokenizer"])
    return {
        "base_model": sha256_file(base / "model.safetensors"),
        "sft_adapter": sha256_file(adapter / "adapter_model.safetensors"),
        "sft_adapter_config": sha256_file(adapter / "adapter_config.json"),
        "tokenizer": sha256_file(tokenizer / "tokenizer.json"),
        "source": sha256_file(Path(config["data"]["source"])),
        "provenance": sha256_file(Path(config["data"]["provenance"])),
        "groups": sha256_file(_groups_file(Path(groups_path))),
        "trie_manifest": sha256_file(Path(trie_dir) / "manifest.json"),
        "fixed_probe": sha256_file(Path(config["evaluation"]["fixed_probe"])),
    }


def _scan_structure(groups_path: Path, trie_dir: Path) -> dict[str, int]:
    from ksllm4rec_grpo.data import iter_groups

    group_count = 0
    positive_edges = 0
    previous_group_id: str | None = None
    seen: set[str] = set()
    for group in iter_groups(_groups_file(groups_path)):
        if group.group_id in seen:
            raise RuntimeError(f"Duplicate recommendation group: {group.group_id}")
        if previous_group_id is not None and group.group_id <= previous_group_id:
            raise RuntimeError(
                "Recommendation groups are not strictly group-id sorted."
            )
        if not group.positive_sids or len(set(group.positive_sids)) != len(
            group.positive_sids
        ):
            raise RuntimeError(f"Invalid positive set for group {group.group_id}.")
        if not group.prompt.endswith("/no_think"):
            raise RuntimeError(f"Group {group.group_id} is not a direct prompt.")
        seen.add(group.group_id)
        previous_group_id = group.group_id
        group_count += 1
        positive_edges += len(group.positive_sids)

    manifest = _load_json(Path(trie_dir) / "manifest.json", "trie manifest")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise RuntimeError("Trie manifest has no counts mapping.")
    return {
        "groups": group_count,
        "positive_edges": positive_edges,
        "trie_leaves": _integer(counts.get("leaves"), "trie leaves"),
        "trie_a_nodes": _integer(counts.get("a_nodes"), "trie a nodes"),
        "trie_ab_nodes": _integer(counts.get("ab_nodes"), "trie ab nodes"),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {label} gate report: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} gate report must be a JSON object: {path}")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}, got {value!r}.")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise RuntimeError(f"{label} must be finite and >= {minimum}, got {value!r}.")
    return result


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        raise RuntimeError(f"{label} must be true.")


def _lambda_values(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "lambda0":
                result.append(_finite(child, "resolved contract lambda0", minimum=0.0))
            result.extend(_lambda_values(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_lambda_values(child))
    return result


def _resolved_lambda0(
    resolved_contract: Any, config: Mapping[str, Any]
) -> tuple[dict[str, Any], float]:
    if not isinstance(resolved_contract, dict):
        raise RuntimeError("Resolved contract must be a JSON object.")
    anchor = resolved_contract.get("anchor")
    if not isinstance(anchor, dict) or "lambda0" not in anchor:
        raise RuntimeError("Resolved contract must contain anchor.lambda0.")
    values = _lambda_values(resolved_contract)
    if not values:
        raise RuntimeError("Resolved contract contains no lambda0.")
    lambda0 = values[0]
    if any(
        not math.isclose(value, lambda0, rel_tol=0.0, abs_tol=1.0e-15)
        for value in values[1:]
    ):
        raise RuntimeError("Resolved contract contains conflicting lambda0 values.")
    maximum = float(config["anchor"]["max_weight"])
    if lambda0 > maximum:
        raise RuntimeError(
            f"Resolved contract lambda0={lambda0} exceeds max_weight={maximum}."
        )
    return resolved_contract, lambda0


def _validate_bound_report(
    name: str,
    report: Mapping[str, Any],
    signature: dict[str, Any],
    *,
    config: Mapping[str, Any],
    resolved_contract: dict[str, Any] | None = None,
) -> float | None:
    if report.get("schema_version") != GATE_SCHEMA_VERSION:
        raise RuntimeError(f"{name} gate has an unsupported schema version.")
    if report.get("gate") != name:
        raise RuntimeError(f"{name} gate has the wrong gate discriminator.")
    _require_true(report.get("passed"), f"{name} gate passed")
    if report.get("runtime_signature") != signature:
        raise RuntimeError(f"{name} gate belongs to a different runtime signature.")
    if resolved_contract is None:
        return None

    actual, lambda0 = _resolved_lambda0(report.get("resolved_contract"), config)
    if canonical_sha256(actual) != canonical_sha256(resolved_contract):
        raise RuntimeError(f"{name} gate belongs to a different resolved contract.")
    recorded_hash = report.get("resolved_contract_sha256")
    expected_hash = canonical_sha256(resolved_contract)
    if recorded_hash != expected_hash:
        raise RuntimeError(f"{name} gate resolved-contract SHA256 is invalid.")
    reported_lambda = _finite(report.get("lambda0"), f"{name} lambda0", minimum=0.0)
    if not math.isclose(reported_lambda, lambda0, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError(f"{name} gate lambda0 differs from its resolved contract.")
    return lambda0


def _validate_legal_finite(
    report: Mapping[str, Any], groups_run: int, config: Mapping[str, Any], label: str
) -> None:
    expected_candidates = groups_run * int(config["rollout"]["num_generations"])
    if (
        _integer(report.get("candidates_run"), f"{label} candidates_run")
        != expected_candidates
    ):
        raise RuntimeError(
            f"{label} gate must inspect exactly {expected_candidates} candidates."
        )
    _require_true(report.get("all_legal"), f"{label} all_legal")
    _require_true(report.get("all_finite"), f"{label} all_finite")


def _validate_branches(
    report: Mapping[str, Any], groups_run: int, config: Mapping[str, Any], label: str
) -> float:
    rloo = _integer(report.get("rloo_groups"), f"{label} rloo_groups")
    anchor = _integer(report.get("anchor_groups"), f"{label} anchor_groups")
    skip = _integer(report.get("skip_groups"), f"{label} skip_groups")
    if rloo + anchor + skip != groups_run:
        raise RuntimeError(f"{label} branch counts do not sum to groups_run.")
    rate = rloo / groups_run
    reported_rate = _finite(
        report.get("rloo_group_rate"), f"{label} rloo_group_rate", minimum=0.0
    )
    if not math.isclose(rate, reported_rate, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"{label} rloo_group_rate disagrees with branch counts.")
    minimum = float(config["gates"]["min_rloo_group_rate"])
    if rate < minimum:
        raise RuntimeError(
            f"{label} RLOO group rate {rate:.6f} is below {minimum:.6f}."
        )
    return rate


def validate_structure_report(
    config: Mapping[str, Any], report: Mapping[str, Any], signature: dict[str, Any]
) -> None:
    _validate_bound_report("structure", report, signature, config=config)
    structure = report.get("structure")
    if structure != expected_structure(config):
        raise RuntimeError("Structure gate counts/chunks differ from Spec V2.0.")
    if report.get("structure_sha256") != canonical_sha256(structure):
        raise RuntimeError("Structure gate SHA256 is invalid.")
    if report.get("input_sha256") != expected_input_sha256():
        raise RuntimeError("Structure gate input SHA256 values differ from the lock.")


def validate_calibration_report(
    config: Mapping[str, Any], report: Mapping[str, Any], signature: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    _validate_bound_report("calibration", report, signature, config=config)
    groups_run = _integer(report.get("groups_run"), "calibration groups_run")
    if groups_run != int(config["anchor"]["calibration_groups"]):
        raise RuntimeError("Calibration gate must consume the frozen 512 groups.")
    _validate_legal_finite(report, groups_run, config, "calibration")
    _validate_branches(report, groups_run, config, "calibration")
    if _integer(report.get("rloo_groups"), "calibration rloo_groups") < 1:
        raise RuntimeError("Calibration RLOO set is empty.")
    if _integer(report.get("anchor_groups"), "calibration anchor_groups") < 1:
        raise RuntimeError("Calibration anchor set is empty.")
    rloo_norm = _finite(
        report.get("rloo_grad_norm"), "calibration rloo_grad_norm", minimum=0.0
    )
    anchor_norm = _finite(
        report.get("anchor_grad_norm"), "calibration anchor_grad_norm", minimum=0.0
    )
    resolved, lambda0 = _resolved_lambda0(report.get("resolved_contract"), config)
    expected_lambda = calibrate_anchor_lambda(
        rloo_norm,
        anchor_norm,
        target_ratio=float(config["anchor"]["target_gradient_ratio"]),
        max_weight=float(config["anchor"]["max_weight"]),
        min_grad_norm=float(config["anchor"]["min_grad_norm"]),
    )
    if not math.isclose(lambda0, expected_lambda, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("Calibration lambda0 does not match the frozen formula.")
    if report.get("resolved_contract_sha256") != canonical_sha256(resolved):
        raise RuntimeError("Calibration resolved-contract SHA256 is invalid.")
    reported_lambda = _finite(
        report.get("lambda0"), "calibration lambda0", minimum=0.0
    )
    if not math.isclose(reported_lambda, lambda0, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("Calibration report lambda0 differs from resolved contract.")
    if _integer(report.get("optimizer_updates"), "calibration optimizer_updates") != 0:
        raise RuntimeError("Calibration must not update the policy.")
    parameter_change = _finite(
        report.get("parameter_max_change"),
        "calibration parameter_max_change",
        minimum=0.0,
    )
    if parameter_change != 0.0:
        raise RuntimeError("Calibration changed a trainable parameter.")
    return resolved, lambda0


def validate_probability_report(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    signature: dict[str, Any],
    resolved_contract: dict[str, Any],
) -> None:
    _validate_bound_report(
        "probability",
        report,
        signature,
        config=config,
        resolved_contract=resolved_contract,
    )
    groups_run = _integer(report.get("groups_run"), "probability groups_run", minimum=1)
    _validate_legal_finite(report, groups_run, config, "probability")
    _require_true(report.get("pre_update"), "probability pre_update")
    _integer(report.get("comparisons"), "probability comparisons", minimum=1)
    difference = _finite(
        report.get("max_abs_logp_difference"),
        "probability max_abs_logp_difference",
        minimum=0.0,
    )
    maximum = float(config["gates"]["max_logp_difference"])
    if difference > maximum:
        raise RuntimeError(
            f"Probability replay difference {difference} exceeds {maximum}."
        )


def _validate_chunks(
    report: Mapping[str, Any], config: Mapping[str, Any], label: str
) -> None:
    expected = int(config["rollout"]["candidate_chunk_size"])
    observed = (
        _integer(report.get("rollout_chunk"), f"{label} rollout_chunk", minimum=1),
        _integer(report.get("loss_chunk"), f"{label} loss_chunk", minimum=1),
        _integer(report.get("gt_chunk"), f"{label} gt_chunk", minimum=1),
    )
    expected_values = (
        expected,
        int(config["memory"]["loss_chunk_size"]),
        int(config["memory"]["gt_chunk_size"]),
    )
    if observed != expected_values:
        raise RuntimeError(f"{label} gate chunks differ from Spec V2.0.")


def validate_memory_report(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    signature: dict[str, Any],
    resolved_contract: dict[str, Any],
) -> None:
    _validate_bound_report(
        "memory",
        report,
        signature,
        config=config,
        resolved_contract=resolved_contract,
    )
    groups_run = _integer(report.get("groups_run"), "memory groups_run")
    if groups_run != 32:
        raise RuntimeError("Memory gate must run the 32 longest prompts.")
    _validate_legal_finite(report, groups_run, config, "memory")
    _validate_chunks(report, config, "memory")
    if report.get("selection") != "longest_prompt_tokens_desc_group_id_tiebreak":
        raise RuntimeError(
            "Memory gate did not use the deterministic longest-32 selection."
        )
    group_ids = report.get("group_ids")
    lengths = report.get("prompt_token_lengths")
    if (
        not isinstance(group_ids, list)
        or len(group_ids) != 32
        or len(set(group_ids)) != 32
        or not all(isinstance(value, str) and value for value in group_ids)
    ):
        raise RuntimeError("Memory gate must record 32 unique non-empty group IDs.")
    if (
        not isinstance(lengths, list)
        or len(lengths) != 32
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in lengths
        )
        or lengths != sorted(lengths, reverse=True)
    ):
        raise RuntimeError(
            "Memory prompt lengths must be 32 positive descending integers."
        )
    for index in range(1, len(lengths)):
        if (
            lengths[index] == lengths[index - 1]
            and group_ids[index] < group_ids[index - 1]
        ):
            raise RuntimeError("Memory prompt ties are not ordered by group ID.")
    if max(lengths) > int(config["data"]["cutoff_len"]):
        raise RuntimeError("Memory prompt length exceeds the frozen cutoff.")
    gpu_name = report.get("gpu_name")
    if not isinstance(gpu_name, str) or "4090" not in gpu_name:
        raise RuntimeError("Memory gate was not run on an RTX 4090.")
    allocated = _finite(
        report.get("peak_allocated_gib"), "memory peak_allocated_gib", minimum=0.0
    )
    reserved = _finite(
        report.get("peak_reserved_gib"), "memory peak_reserved_gib", minimum=0.0
    )
    if allocated > reserved:
        raise RuntimeError("Peak allocated memory exceeds peak reserved memory.")
    maximum = float(config["memory"]["max_reserved_gib"])
    if reserved > maximum:
        raise RuntimeError(f"Memory gate reserved {reserved} GiB, above {maximum} GiB.")
    if _integer(report.get("backward_passes"), "memory backward_passes") < 1:
        raise RuntimeError("Memory gate performed no backward pass.")
    if _integer(report.get("optimizer_updates"), "memory optimizer_updates") < 1:
        raise RuntimeError(
            "Memory gate did not allocate steady-state optimizer memory."
        )


def validate_signal_report(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    signature: dict[str, Any],
    resolved_contract: dict[str, Any],
) -> None:
    _validate_bound_report(
        "signal",
        report,
        signature,
        config=config,
        resolved_contract=resolved_contract,
    )
    groups_run = _integer(report.get("groups_run"), "signal groups_run")
    if groups_run != int(config["gates"]["signal_groups"]):
        raise RuntimeError("Signal gate must run exactly 512 groups.")
    _validate_legal_finite(report, groups_run, config, "signal")
    _validate_branches(report, groups_run, config, "signal")
    _validate_chunks(report, config, "signal")
    if _integer(report.get("optimizer_updates"), "signal optimizer_updates") < 1:
        raise RuntimeError("Signal gate performed no optimizer update.")
    if _finite(report.get("gradient_norm_max"), "signal gradient_norm_max") <= 0.0:
        raise RuntimeError("Signal gate has no non-zero gradient.")
    if _finite(
        report.get("parameter_max_change"), "signal parameter_max_change"
    ) <= 0.0:
        raise RuntimeError("Signal gate did not change a trainable parameter.")
    _finite(report.get("loss_mean"), "signal loss_mean")


def projected_training_hours(seconds: float, groups_run: int) -> float:
    """Project two epochs with the approved 20% safety margin."""

    duration = _finite(seconds, "timing seconds", minimum=0.0)
    count = _integer(groups_run, "timing groups_run", minimum=1)
    if duration <= 0.0:
        raise RuntimeError("Timing seconds must be positive.")
    return TIMING_SAFETY_FACTOR * (duration / count * TOTAL_GROUP_VISITS) / 3600.0


def validate_timing_report(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    signature: dict[str, Any],
    resolved_contract: dict[str, Any],
) -> None:
    _validate_bound_report(
        "timing",
        report,
        signature,
        config=config,
        resolved_contract=resolved_contract,
    )
    groups_run = _integer(report.get("groups_run"), "timing groups_run")
    if groups_run != int(config["gates"]["timing_groups"]):
        raise RuntimeError("Timing gate must run exactly 256 groups.")
    _validate_legal_finite(report, groups_run, config, "timing")
    _validate_branches(report, groups_run, config, "timing")
    _validate_chunks(report, config, "timing")
    _require_true(report.get("end_to_end"), "timing end_to_end")
    expected = projected_training_hours(report.get("seconds"), groups_run)
    recorded = _finite(
        report.get("projected_two_epoch_hours"),
        "timing projected_two_epoch_hours",
        minimum=0.0,
    )
    if not math.isclose(recorded, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError("Timing projection does not use the frozen formula.")
    maximum = float(config["gates"]["max_projected_hours"])
    if expected > maximum:
        raise RuntimeError(
            f"Projected training time {expected:.3f} h exceeds {maximum} h."
        )


def _bind_result(
    name: str,
    result: Mapping[str, Any],
    signature: dict[str, Any],
    resolved_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = dict(result)
    fixed: dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": name,
        "passed": True,
        "runtime_signature": signature,
    }
    if resolved_contract is not None:
        _, lambda0 = _resolved_lambda0(
            resolved_contract,
            {"anchor": {"max_weight": contract.ANCHOR_MAX_WEIGHT}},
        )
        fixed.update(
            {
                "resolved_contract": resolved_contract,
                "resolved_contract_sha256": canonical_sha256(resolved_contract),
                "lambda0": lambda0,
            }
        )
    for key, value in fixed.items():
        if key in report and report[key] != value:
            raise RuntimeError(f"{name} runner returned conflicting {key}.")
        report[key] = value
    return report


def run_structure_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_path: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    signature = runtime_signature(
        config,
        groups_path,
        trie_dir,
        calibration_ids_path,
        config_path=config_path,
    )
    scanned = _scan_structure(groups_path, trie_dir)
    structure = expected_structure(config)
    for key, value in scanned.items():
        if structure[key] != value:
            raise RuntimeError(
                "Structure mismatch for "
                f"{key}: expected={structure[key]}, actual={value}."
            )
    report = _bind_result(
        "structure",
        {
            "structure": structure,
            "structure_sha256": canonical_sha256(structure),
            "input_sha256": collect_input_sha256(config, groups_path, trie_dir),
        },
        signature,
    )
    validate_structure_report(config, report, signature)
    _atomic_write_json(output_path, report)
    return report


def run_calibration_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_path: Path,
    device: str = "cuda:0",
    config_path: Path | None = None,
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    signature = runtime_signature(
        config,
        groups_path,
        trie_dir,
        calibration_ids_path,
        config_path=config_path,
    )
    if runner is None:
        from .trainer import run_calibration as runner

    result = runner(
        config=config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        calibration_ids_path=calibration_ids_path,
        runtime_signature=signature,
        output_path=output_path,
        device=device,
    )
    report = _bind_result("calibration", result, signature)
    validate_calibration_report(config, report, signature)
    _atomic_write_json(output_path, report)
    return report


def _run_training_gate(
    name: str,
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_dir: Path,
    output_path: Path,
    resolved_contract: dict[str, Any],
    max_groups: int,
    device: str,
    config_path: Path | None,
    runner: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    signature = runtime_signature(
        config,
        groups_path,
        trie_dir,
        calibration_ids_path,
        config_path=config_path,
    )
    if runner is None:
        from .trainer import run_training as runner

    if output_dir.exists():
        shutil.rmtree(output_dir)
    result = runner(
        config=config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        output_dir=output_dir,
        resolved_contract=resolved_contract,
        runtime_signature=signature,
        max_groups=max_groups,
        resume=False,
        save_recovery=False,
        save_epochs=False,
        device=device,
        gate_mode=name,
    )
    report = _bind_result(name, result, signature, resolved_contract)
    validator = validate_signal_report if name == "signal" else validate_timing_report
    validator(config, report, signature, resolved_contract)
    _atomic_write_json(output_path, report)
    return report


def run_signal_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_dir: Path,
    output_path: Path,
    resolved_contract: dict[str, Any],
    device: str = "cuda:0",
    config_path: Path | None = None,
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return _run_training_gate(
        "signal",
        config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        calibration_ids_path=calibration_ids_path,
        output_dir=output_dir,
        output_path=output_path,
        resolved_contract=resolved_contract,
        max_groups=int(config["gates"]["signal_groups"]),
        device=device,
        config_path=config_path,
        runner=runner,
    )


def run_timing_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_dir: Path,
    output_path: Path,
    resolved_contract: dict[str, Any],
    device: str = "cuda:0",
    config_path: Path | None = None,
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return _run_training_gate(
        "timing",
        config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        calibration_ids_path=calibration_ids_path,
        output_dir=output_dir,
        output_path=output_path,
        resolved_contract=resolved_contract,
        max_groups=int(config["gates"]["timing_groups"]),
        device=device,
        config_path=config_path,
        runner=runner,
    )


def _run_special_gate(
    name: str,
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_path: Path,
    resolved_contract: dict[str, Any],
    max_groups: int,
    device: str,
    config_path: Path | None,
    runner: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    signature = runtime_signature(
        config,
        groups_path,
        trie_dir,
        calibration_ids_path,
        config_path=config_path,
    )
    if runner is None:
        from . import trainer

        runner = getattr(trainer, f"run_{name}_check")
    result = runner(
        config=config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        resolved_contract=resolved_contract,
        runtime_signature=signature,
        max_groups=max_groups,
        output_path=output_path,
        device=device,
    )
    report = _bind_result(name, result, signature, resolved_contract)
    validator = (
        validate_probability_report if name == "probability" else validate_memory_report
    )
    validator(config, report, signature, resolved_contract)
    _atomic_write_json(output_path, report)
    return report


def run_probability_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_path: Path,
    resolved_contract: dict[str, Any],
    max_groups: int = 1,
    device: str = "cuda:0",
    config_path: Path | None = None,
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return _run_special_gate(
        "probability",
        config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        calibration_ids_path=calibration_ids_path,
        output_path=output_path,
        resolved_contract=resolved_contract,
        max_groups=max_groups,
        device=device,
        config_path=config_path,
        runner=runner,
    )


def run_memory_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    output_path: Path,
    resolved_contract: dict[str, Any],
    device: str = "cuda:0",
    config_path: Path | None = None,
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return _run_special_gate(
        "memory",
        config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        calibration_ids_path=calibration_ids_path,
        output_path=output_path,
        resolved_contract=resolved_contract,
        max_groups=32,
        device=device,
        config_path=config_path,
        runner=runner,
    )


def load_training_gates(
    *,
    config: dict[str, Any],
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    structure_path: Path,
    calibration_path: Path,
    probability_path: Path,
    memory_path: Path,
    signal_path: Path,
    timing_path: Path,
    config_path: Path | None = None,
) -> GateBundle:
    """Load six reports and reject any cross-run or cross-lambda mixture."""

    signature = runtime_signature(
        config,
        groups_path,
        trie_dir,
        calibration_ids_path,
        config_path=config_path,
    )
    validate_runtime_signature(signature)
    reports = {
        "structure": _load_json(structure_path, "structure"),
        "calibration": _load_json(calibration_path, "calibration"),
        "probability": _load_json(probability_path, "probability"),
        "memory": _load_json(memory_path, "memory"),
        "signal": _load_json(signal_path, "signal"),
        "timing": _load_json(timing_path, "timing"),
    }
    validate_structure_report(config, reports["structure"], signature)
    resolved_contract, lambda0 = validate_calibration_report(
        config, reports["calibration"], signature
    )
    validate_probability_report(
        config, reports["probability"], signature, resolved_contract
    )
    validate_memory_report(config, reports["memory"], signature, resolved_contract)
    validate_signal_report(config, reports["signal"], signature, resolved_contract)
    validate_timing_report(config, reports["timing"], signature, resolved_contract)
    return GateBundle(
        runtime_signature=signature,
        resolved_contract=resolved_contract,
        lambda0=lambda0,
        rollout_chunk=int(config["rollout"]["candidate_chunk_size"]),
        loss_chunk=int(config["memory"]["loss_chunk_size"]),
        gt_chunk=int(config["memory"]["gt_chunk_size"]),
        reports=reports,
    )
