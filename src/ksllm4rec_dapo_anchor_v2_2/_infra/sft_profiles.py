"""Immutable identities for reproducible SFT experiment profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PROJECT_ROOT = Path("/home/lyc/REC_PROJECTS/KSLLM4REC")
MODEL_PATH = Path("/home/lyc/models/OneReason-0.8B-pretrain-competition")
DEFAULT_PROFILE_NAME = "baseline"
FRONTIER_PROFILE_NAME = "frontier_feedbackcore_listwise_invariant_v1"


@dataclass(frozen=True)
class SFTProfile:
    """Values that distinguish one approved SFT data run from another."""

    name: str
    dataset_name: str
    dataset_dir: Path
    source_path: Path
    source_sha256: str
    source_size: int
    source_records: int
    gate_cutoffs: Mapping[str, int]
    full_stage: str
    config_check_stage: str
    config_path: Path
    artifact_lock_path: Path
    gate_output_root: Path
    full_output_dir: Path
    fingerprint_paths: tuple[str, ...]
    fingerprint_script_dirs: tuple[str, ...] = ()

    @property
    def derived_path(self) -> Path:
        return self.dataset_dir / "train_alpaca.jsonl"

    def stage_cutoff(self, run_stage: str) -> int:
        if run_stage in (self.full_stage, self.config_check_stage):
            return 16384
        try:
            return self.gate_cutoffs[run_stage]
        except KeyError as exc:
            raise RuntimeError(
                f"Unapproved SFT run stage for profile {self.name!r}: {run_stage!r}"
            ) from exc


BASELINE_PROFILE = SFTProfile(
    name=DEFAULT_PROFILE_NAME,
    dataset_name="hf_kuaishou_llmrec_sft_baseline_0_91",
    dataset_dir=PROJECT_ROOT / "artifacts/sft/data/hf_baseline_091",
    source_path=Path("/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl"),
    source_sha256="4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8",
    source_size=242_712_094,
    source_records=32_705,
    gate_cutoffs=MappingProxyType(
        {
            "gate_00512": 512,
            "gate_02048": 2048,
            "gate_08192": 8192,
            "gate_16384": 16384,
        }
    ),
    full_stage="full_epoch_001",
    config_check_stage="config_check",
    config_path=PROJECT_ROOT / "configs/sft/onereason_lora_focal_item.yaml",
    artifact_lock_path=PROJECT_ROOT / "configs/sft/artifacts.lock.json",
    gate_output_root=PROJECT_ROOT / "artifacts/sft/runs/gates",
    full_output_dir=PROJECT_ROOT / "artifacts/sft/runs/full_epoch_001",
    fingerprint_paths=(
        "configs/sft/artifacts.lock.json",
        "configs/sft/constraints.txt",
        "configs/sft/environment.lock.txt",
        "configs/sft/onereason_lora_focal_item.yaml",
    ),
)

FRONTIER_PROFILE = SFTProfile(
    name=FRONTIER_PROFILE_NAME,
    dataset_name=FRONTIER_PROFILE_NAME,
    dataset_dir=(
        PROJECT_ROOT / "artifacts/sft/data/frontier_feedbackcore_listwise_invariant_v1"
    ),
    source_path=Path(
        "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"
    ),
    source_sha256="9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a",
    source_size=269_105_772,
    source_records=63_700,
    gate_cutoffs=MappingProxyType(
        {
            "frontier_gate_00512": 512,
            "frontier_gate_02048": 2048,
            "frontier_gate_08192": 8192,
            "frontier_gate_16384": 16384,
        }
    ),
    full_stage="frontier_full_epoch_001",
    config_check_stage="frontier_config_check",
    config_path=(
        PROJECT_ROOT / "configs/sft/frontier_feedbackcore_listwise_invariant_v1.yaml"
    ),
    artifact_lock_path=(
        PROJECT_ROOT
        / "configs/sft/frontier_feedbackcore_listwise_invariant_v1.artifacts.lock.json"
    ),
    gate_output_root=(
        PROJECT_ROOT
        / "artifacts/sft/runs/frontier_feedbackcore_listwise_invariant_v1_gates"
    ),
    full_output_dir=(
        PROJECT_ROOT
        / "artifacts/sft/runs/frontier_feedbackcore_listwise_invariant_v1_epoch_001"
    ),
    fingerprint_paths=(
        "configs/sft/constraints.txt",
        "configs/sft/environment.lock.txt",
        "configs/sft/frontier_feedbackcore_listwise_invariant_v1.yaml",
        "configs/sft/frontier_feedbackcore_listwise_invariant_v1.artifacts.lock.json",
    ),
    fingerprint_script_dirs=("scripts/sft/frontier",),
)

PROFILES = MappingProxyType(
    {
        BASELINE_PROFILE.name: BASELINE_PROFILE,
        FRONTIER_PROFILE.name: FRONTIER_PROFILE,
    }
)
PROFILE_NAMES = tuple(PROFILES)


def get_profile(profile: str | SFTProfile = DEFAULT_PROFILE_NAME) -> SFTProfile:
    if isinstance(profile, SFTProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SFT profile {profile!r}; expected one of {PROFILE_NAMES!r}."
        ) from exc
