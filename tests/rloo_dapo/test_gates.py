from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_rloo.integrity import canonical_sha256, sha256_file, snapshot_directory
from ksllm4rec_rloo_dapo import contract
from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.device import require_gpu_identity
from ksllm4rec_rloo_dapo.gates import (
    expected_input_sha256,
    load_and_validate_gate_reports,
)


def _signature() -> dict:
    inputs = {"test": "gate"}
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


GPU_IDENTITY = {
    "device": "cuda:0",
    "name": "NVIDIA GeForce RTX 4090",
    "uuid": "e1f24bb9-9bcb-c31f-52ba-fb5f63a146d5",
    "total_memory": 25_756_696_576,
    "compute_capability": [8, 9],
}


class GateValidationTest(unittest.TestCase):
    def _write_reports(
        self, root: Path, config: dict, exact_speedup: float
    ) -> dict[str, Path]:
        signature = _signature()
        expected = expected_input_sha256(config)
        structure = {
            "groups": config["data"]["groups"],
            "positive_edges": config["data"]["positives"],
            "unique_group_ids": config["data"]["groups"],
            "trie_leaves": config["trie"]["unique_sids"],
            "trie_a_nodes": config["trie"]["domain_a_nodes"],
            "trie_ab_nodes": config["trie"]["domain_ab_nodes"],
            "effective_groups_per_window": config["sampling"][
                "effective_groups_per_window"
            ],
            "group_size": config["rollout"]["num_generations"],
            "minibatch_groups": config["loss"]["minibatch_groups"],
            "optimizer_updates_per_window": config["train"][
                "optimizer_updates_per_window"
            ],
            "total_windows": config["train"]["total_windows"],
            "total_optimizer_updates": config["train"]["total_optimizer_updates"],
        }
        baseline_seconds = exact_speedup * 2.0
        pilot_windows = config["gates"]["pilot_windows"]
        updates = pilot_windows * config["train"]["optimizer_updates_per_window"]
        effective_groups = (
            pilot_windows * config["sampling"]["effective_groups_per_window"]
        )
        pilot_root = root / "pilot-artifact"
        baseline_root = root / "dense-artifact"
        checkpoint_name = f"checkpoint-window-{pilot_windows:06d}-update-{updates:06d}"
        checkpoint_relative = Path("recovery") / checkpoint_name
        checkpoint = pilot_root / checkpoint_relative
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"pilot-final-adapter")
        (checkpoint / "adapter_config.json").write_text(
            '{"target_modules":["q_proj","k_proj"]}\n', encoding="utf-8"
        )
        (checkpoint / "training_state.pt").write_bytes(b"exact-training-state")
        baseline_root.mkdir()
        signature = _signature()
        pilot_summary = {
            "schema_version": 1,
            "completed_windows": pilot_windows,
            "optimizer_update_step": updates,
            "windows_run_this_invocation": pilot_windows,
            "source_cursor": {"cycle_index": 0, "offset": 80},
            "last_checkpoint": checkpoint_relative.as_posix(),
            "complete": False,
            "anchor_groups": 0,
            "gt_injection_count": 0,
            "k": 1,
        }
        baseline_updates = (
            config["gates"]["end_to_end_windows"]
            * config["train"]["optimizer_updates_per_window"]
        )
        baseline_summary = {
            "schema_version": 1,
            "completed_windows": config["gates"]["end_to_end_windows"],
            "optimizer_update_step": baseline_updates,
            "windows_run_this_invocation": config["gates"]["end_to_end_windows"],
            "source_cursor": {"cycle_index": 0, "offset": 40},
            "last_checkpoint": ("recovery/checkpoint-window-000001-update-000004"),
            "complete": False,
            "anchor_groups": 0,
            "gt_injection_count": 0,
            "k": 1,
        }

        def write_json(path: Path, value: object) -> None:
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")

        def window_row(index: int) -> dict:
            return {
                "window_index": index,
                "effective_groups": config["sampling"]["effective_groups_per_window"],
                "optimizer_updates": config["train"]["optimizer_updates_per_window"],
                "group_ids": [f"window-{index}-group-{value}" for value in range(32)],
                "anchor_groups": 0,
                "gt_injection_count": 0,
                "k": 1,
            }

        def group_rows(count: int) -> list[dict]:
            return [
                {
                    "group_id": f"group-{index}",
                    "effective": True,
                    "anchor_groups": 0,
                    "gt_injection_count": 0,
                    "k": 1,
                }
                for index in range(count)
            ]

        for artifact_root, summary, artifact_windows in (
            (
                pilot_root,
                pilot_summary,
                [window_row(index) for index in range(pilot_windows)],
            ),
            (baseline_root, baseline_summary, [window_row(0)]),
        ):
            write_json(artifact_root / "runtime_signature.json", signature)
            write_json(artifact_root / "resolved_config.json", config)
            write_json(artifact_root / "run_summary.json", summary)
            (artifact_root / "windows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in artifact_windows),
                encoding="utf-8",
            )
            (artifact_root / "groups.jsonl").write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in group_rows(len(artifact_windows) * 32)
                ),
                encoding="utf-8",
            )
        recovery_root = root / "recovery-artifact"
        shutil.copytree(pilot_root, recovery_root)
        recovery_summary = dict(pilot_summary)
        recovery_summary["windows_run_this_invocation"] = 1
        write_json(recovery_root / "run_summary.json", recovery_summary)
        continuous_groups_hash = sha256_file(pilot_root / "groups.jsonl")
        recovery_groups_hash = sha256_file(recovery_root / "groups.jsonl")
        continuous_state_hash = sha256_file(checkpoint / "training_state.pt")
        recovery_checkpoint = recovery_root / checkpoint_relative
        recovery_state_hash = sha256_file(recovery_checkpoint / "training_state.pt")
        recovery_adapter_hash = sha256_file(
            recovery_checkpoint / "adapter_model.safetensors"
        )
        adapter_config_semantic_hash = canonical_sha256(
            {"target_modules": ["k_proj", "q_proj"]}
        )
        reports = {
            "structure": {
                "input_sha256": expected,
                "expected_input_sha256": expected,
                "structure": structure,
                "structure_sha256": canonical_sha256(structure),
                "anchor_fields_present": False,
            },
            "probability": {
                "groups_run": config["gates"]["parity_groups"],
                "candidates_run": (
                    config["gates"]["parity_groups"]
                    * config["rollout"]["num_generations"]
                ),
                "all_legal": True,
                "all_finite": True,
                "max_sample_canonical_logp_difference": 0.0,
                "max_canonical_replay_logp_difference": 0.0,
                "max_proposal_canonical_logp_difference": 0.0,
                "corrected_candidates": 0,
                "anchor_groups": 0,
                "gt_injection_count": 0,
                "k": 1,
                "gpu_identity": GPU_IDENTITY,
            },
            "memory": {
                "prompt_group_id": "test-group",
                "prompt_length": config["memory"]["max_observed_prompt_length"],
                "prompt_batch_size": config["rollout"]["prompt_batch_size"],
                "peak_reserved_gib": 19.0,
                "max_reserved_gib": config["memory"]["max_reserved_gib"],
                "gradients_finite": True,
                "cache_stats": {
                    "prompt_count": config["rollout"]["prompt_batch_size"],
                    "rollout_count": (
                        config["rollout"]["prompt_batch_size"]
                        * config["rollout"]["num_generations"]
                    ),
                    "prefill_calls": config["rollout"]["prompt_batch_size"],
                    "decode_calls": 63,
                    "cache_forks": 0,
                    "requested_active_sequences": config["rollout"][
                        "max_active_sequences"
                    ],
                    "resolved_active_sequences": config["rollout"][
                        "max_active_sequences"
                    ],
                    "estimated_peak_cache_bytes": 335_921_152,
                    "fallback_prompt_count": 0,
                },
                "gpu_identity": GPU_IDENTITY,
            },
            "throughput": {
                "groups_run": config["gates"]["throughput_groups"],
                "rollouts": (
                    config["gates"]["throughput_groups"]
                    * config["rollout"]["num_generations"]
                ),
                "baseline_seconds": baseline_seconds,
                "cache_proposal_seconds": 1.5,
                "correction_seconds": 0.5,
                "exact_cached_rollout_seconds": 2.0,
                "cache_proposal_speedup": baseline_seconds / 1.5,
                "exact_cached_rollout_speedup": exact_speedup,
                "required_rollout_speedup": config["gates"]["min_rollout_speedup"],
                "gpu_identity": GPU_IDENTITY,
            },
            "pilot": {
                "windows": pilot_windows,
                "effective_groups": effective_groups,
                "optimizer_updates": updates,
                "unique_group_occurrences": effective_groups,
                "initial_adapter_sha256": expected["sft_adapter"],
                "final_adapter_sha256": sha256_file(
                    checkpoint / "adapter_model.safetensors"
                ),
                "parameter_changed": True,
                "peak_reserved_gib": 19.0,
                "elapsed_seconds": 20.0,
                "end_to_end_windows": config["gates"]["end_to_end_windows"],
                "pilot_dir": str(pilot_root.resolve()),
                "dense_baseline_dir": str(baseline_root.resolve()),
                "recovery_check_dir": str(recovery_root.resolve()),
                "pilot_files": snapshot_directory(pilot_root),
                "dense_baseline_files": snapshot_directory(baseline_root),
                "recovery_check_files": snapshot_directory(recovery_root),
                "baseline_window_seconds": 12.0,
                "optimized_window_seconds": 10.0,
                "max_optimized_window_seconds": 600.0,
                "end_to_end_window_speedup": 1.2,
                "required_window_speedup": config["gates"]["min_window_speedup"],
                "run_summary": pilot_summary,
                "recovery_run_summary": recovery_summary,
                "recovery_elapsed_seconds": 20.0,
                "recovery_exact": True,
                "recovery_processes": 2,
                "continuous_groups_sha256": continuous_groups_hash,
                "recovery_groups_sha256": recovery_groups_hash,
                "continuous_training_state_sha256": continuous_state_hash,
                "recovery_training_state_sha256": recovery_state_hash,
                "continuous_adapter_config_semantic_sha256": (
                    adapter_config_semantic_hash
                ),
                "recovery_adapter_config_semantic_sha256": (
                    adapter_config_semantic_hash
                ),
                "recovery_final_adapter_sha256": recovery_adapter_hash,
                "anchor_groups": 0,
                "gt_injection_count": 0,
                "k": 1,
                "gpu_identity": GPU_IDENTITY,
            },
        }
        paths = {}
        for name, fields in reports.items():
            value = {
                "schema_version": 1,
                "gate": name,
                "passed": True,
                "runtime_signature_sha256": signature["sha256"],
                **fields,
            }
            path = root / f"{name}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            paths[name] = path
        return paths

    def test_exact_corrected_rollout_must_meet_the_speed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.49)
            with self.assertRaisesRegex(RuntimeError, "rollout threshold"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_complete_bound_reports_are_accepted(self) -> None:
        for profile in (contract.PROFILE, contract.SFT372_PROFILE):
            with (
                self.subTest(profile=profile),
                tempfile.TemporaryDirectory() as directory,
            ):
                config = approved_config(profile)
                paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
                reports = load_and_validate_gate_reports(config, _signature(), paths)
                self.assertEqual(set(reports), set(paths))

    def test_end_to_end_window_speedup_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["end_to_end_window_speedup"] = 1.19
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "end_to_end_window_speedup|end-to-end window"
            ):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_reports_from_old_profile_are_rejected_by_sft372_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old = approved_config()
            paths = self._write_reports(Path(directory), old, exact_speedup=1.5)
            with self.assertRaisesRegex(RuntimeError, "input hashes"):
                load_and_validate_gate_reports(
                    approved_config(contract.SFT372_PROFILE), _signature(), paths
                )

    def test_pilot_must_start_from_profile_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config(contract.SFT372_PROFILE)
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["initial_adapter_sha256"] = "0" * 64
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "wrong SFT adapter"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_missing_required_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            probability = json.loads(paths["probability"].read_text(encoding="utf-8"))
            del probability["all_legal"]
            paths["probability"].write_text(json.dumps(probability), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing fields"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_nan_is_rejected_even_when_passed_is_true(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            memory = json.loads(paths["memory"].read_text(encoding="utf-8"))
            memory["peak_reserved_gib"] = float("nan")
            paths["memory"].write_text(json.dumps(memory), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_forged_derived_speedup_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            throughput = json.loads(paths["throughput"].read_text(encoding="utf-8"))
            throughput["exact_cached_rollout_speedup"] = 99.0
            paths["throughput"].write_text(json.dumps(throughput), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "was not reproduced"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_forged_pilot_speedup_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["end_to_end_window_speedup"] = 9.0
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "was not reproduced"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_optimized_pilot_window_must_finish_within_ten_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["optimized_window_seconds"] = 601.0
            pilot["baseline_window_seconds"] = 721.2
            pilot["end_to_end_window_speedup"] = 1.2
            pilot["elapsed_seconds"] = 700.0
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "10-minute"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_deleted_pilot_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            (Path(pilot["pilot_dir"]) / "groups.jsonl").unlink()
            with self.assertRaisesRegex(RuntimeError, "snapshot"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_recovery_comparison_must_be_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["recovery_exact"] = False
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "recovery_exact"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_exact_counts_and_anchor_fields_are_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            probability = json.loads(paths["probability"].read_text(encoding="utf-8"))
            probability["anchor_groups"] = 1
            paths["probability"].write_text(json.dumps(probability), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "anchor_groups"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_pilot_run_summary_must_match_top_level_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["run_summary"]["optimizer_update_step"] -= 1
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "run_summary disagrees"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_all_gpu_reports_must_share_one_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["gpu_identity"]["uuid"] = "00000000-0000-0000-0000-000000000001"
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different devices"):
                load_and_validate_gate_reports(config, _signature(), paths)

    def test_current_gpu_identity_can_be_bound_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = approved_config()
            paths = self._write_reports(Path(directory), config, exact_speedup=1.5)
            expected = {**GPU_IDENTITY, "total_memory": 1}
            with self.assertRaisesRegex(RuntimeError, "current GPU"):
                load_and_validate_gate_reports(
                    config,
                    _signature(),
                    paths,
                    expected_gpu_identity=expected,
                )

    def test_formal_training_gpu_helper_rechecks_the_live_device(self) -> None:
        with patch(
            "ksllm4rec_rloo_dapo.device.gpu_identity",
            return_value=dict(GPU_IDENTITY),
        ):
            self.assertEqual(require_gpu_identity(GPU_IDENTITY), GPU_IDENTITY)
        other = {**GPU_IDENTITY, "total_memory": 1}
        with (
            patch(
                "ksllm4rec_rloo_dapo.device.gpu_identity",
                return_value=other,
            ),
            self.assertRaisesRegex(RuntimeError, "Current GPU identity"),
        ):
            require_gpu_identity(GPU_IDENTITY)


if __name__ == "__main__":
    unittest.main()
