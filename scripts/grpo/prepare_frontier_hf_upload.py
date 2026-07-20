#!/usr/bin/env python3
"""Materialize the upload contract for the Frontier-started GRPO run.

This command is intentionally local-only.  It reads the two completed LoRA
epoch directories, computes their exact file metadata, and writes a strict
configuration consumed by ``scripts/orpo/upload_hf_adapters.py``.  It never
contacts Hugging Face and never uploads files.

The generated configuration is deliberately separate from the historical
``configs/grpo/hf_upload.json`` contract.  The latter remains the contract
for the earlier baseline-started GRPO run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    "artifacts/grpo/runs/"
    "frontier_sft_epoch2_grpo_g8_forcedgt_p050_zscore_2epoch"
)
DEFAULT_OUTPUT = "configs/grpo/hf_upload_frontier_sft_epoch2.json"
DEFAULT_TEMPLATE = "configs/grpo/frontier_sft_epoch2_grpo_README.md.tmpl"
TARGET_REPOS = (
    "reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch1",
    "reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch2",
)
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class PrepareError(RuntimeError):
    """A local upload-contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PrepareError(f"required file does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise PrepareError(f"required file is empty: {path}")
    return {"size": size, "sha256": sha256_file(path)}


def validate_adapter_config(path: Path) -> None:
    """Check the immutable LoRA shape before creating an upload contract."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrepareError(f"invalid adapter_config.json: {path}") from exc
    if not isinstance(value, dict):
        raise PrepareError(f"adapter_config.json must contain an object: {path}")
    if value.get("peft_type") != "LORA":
        raise PrepareError(f"adapter_config.json is not a LoRA config: {path}")
    if value.get("r") != 32 or value.get("lora_alpha") != 32:
        raise PrepareError(
            f"unexpected LoRA rank/alpha in {path}: "
            f"r={value.get('r')!r}, alpha={value.get('lora_alpha')!r}"
        )


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PrepareError(f"{label} escapes project root: {path}") from exc
    return resolved


def resolve_path(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return _inside(root, candidate, label)


def relative_path(root: Path, path: Path, label: str) -> str:
    resolved = _inside(root, path, label)
    value = resolved.relative_to(root).as_posix()
    if not value or value == ".":
        raise PrepareError(f"{label} must not be the project root")
    return value


def check_epoch_dir(root: Path, run_dir: Path, epoch: int) -> dict[str, Any]:
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    if not epoch_dir.is_dir():
        raise PrepareError(f"missing completed epoch directory: {epoch_dir}")

    records: dict[str, Any] = {}
    for name in REQUIRED_FILES:
        records[name] = file_record(epoch_dir / name)
    validate_adapter_config(epoch_dir / "adapter_config.json")

    # The upload tool stages only these two checkpoint files plus a generated
    # README.  Extra audit/checkpoint files in an epoch directory are allowed
    # here but can never enter the staging allowlist.
    return {
        "epoch": epoch,
        "source_dir": relative_path(root, epoch_dir, f"epoch {epoch} source_dir"),
        "files": records,
    }


def build_config(
    *, root: Path, run_dir: Path, template_path: Path
) -> dict[str, Any]:
    run_dir = _inside(root, run_dir, "run_dir")
    if not run_dir.is_dir():
        raise PrepareError(f"GRPO run directory does not exist: {run_dir}")
    template_path = _inside(root, template_path, "README template")
    template_record = file_record(template_path)

    epochs = [check_epoch_dir(root, run_dir, epoch) for epoch in (1, 2)]
    config_records = [item["files"]["adapter_config.json"] for item in epochs]
    if config_records[0] != config_records[1]:
        raise PrepareError(
            "epoch adapter_config.json files differ; refusing to publish "
            "mixed LoRA contracts"
        )

    run_relative = relative_path(root, run_dir, "run_dir")
    template_relative = relative_path(root, template_path, "README template")
    repositories = []
    for item, repo_id in zip(epochs, TARGET_REPOS):
        repositories.append(
            {
                "epoch": item["epoch"],
                "repo_id": repo_id,
                "source_dir": item["source_dir"],
                "adapter_model": item["files"]["adapter_model.safetensors"],
                "adapter_config": item["files"]["adapter_config.json"],
            }
        )

    return {
        "schema_version": 1,
        "profile": "frontier_sft_epoch2_grpo_v2",
        "namespace": "reimu996",
        "visibility": "public",
        "artifact_root": "artifacts/grpo/hf_upload_frontier_sft_epoch2",
        "log_root": "operation_logs/grpo/frontier_sft_epoch2_hf_upload",
        "required_uploaded_files": [
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
        ],
        "allowed_hub_managed_files": [".gitattributes"],
        "required_remote_files": [
            ".gitattributes",
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
        ],
        "commit_message": (
            "Upload OneReason-0.8B Frontier SFT Epoch2 GRPO LoRA adapter"
        ),
        "readme_template": {
            "path": template_relative,
            "size": template_record["size"],
            "sha256": template_record["sha256"],
        },
        "training": {
            "title": "OneReason-0.8B Frontier SFT Epoch 2 -> Online GRPO LoRA",
            "base_description": (
                "OneReason-0.8B competition base with the official Frontier SFT "
                "Epoch 2 adapter"
            ),
            "method": "online GRPO",
            "method_details": (
                "G=8 live legal-SID sampling, /think and /no_think prompt "
                "normalization, final-SID-only reward, forced-GT p=0.5, "
                "group sample Z-score, two epochs"
            ),
            "data_label": "Frontier GRPO groups",
            "data_value": "17,016 normalized groups x 2 epochs",
            "lora_rank": 32,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "evaluation_status": (
                "Only a local Frontier-derived constrained beam-16 probe was run; "
                "no official competition score is claimed."
            ),
        },
        "provenance": {
            "grpo_run_dir": run_relative,
            "source_dataset": (
                "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1"
            ),
            "source_dataset_policy": (
                "raw direct/thinking records retained; /think and /no_think "
                "normalized; "
                "duplicate GTs collapsed only inside each positive SID set"
            ),
            "recommendation_groups": 17016,
            "raw_recommendation_rows": 30902,
            "deduplicated_positive_edges": 30465,
            "unique_positive_sids": 29414,
            "legal_sid_leaves": 905469,
            "readme_template": template_relative,
        },
        "repositories": repositories,
    }


def write_json(path: Path, value: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise PrepareError(
            "output already exists; refusing overwrite (use --force only after "
            f"review): {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        root = args.project_root.resolve()
        if not root.is_dir():
            raise PrepareError(f"project root does not exist: {root}")
        run_dir = resolve_path(root, args.run_dir, "run_dir")
        template_path = resolve_path(root, args.template, "README template")
        output_path = resolve_path(root, args.output, "output config")
        config = build_config(
            root=root, run_dir=run_dir, template_path=template_path
        )
        write_json(output_path, config, force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"config={output_path}")
    print("remote_upload=not_performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
