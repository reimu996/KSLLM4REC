from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from ksllm4rec_rloo_dapo import contract
from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.trainer import learning_rate_for_window
from ksllm4rec_rloo_dapo.verify import (
    _normalized_adapter_config,
    validate_final_adapter,
    validate_group_rows,
    validate_window_rows,
)


def _window(config, index: int):
    ids = [f"w{index}-g{group}" for group in range(32)]
    lr = learning_rate_for_window(config, index)
    steps = []
    for start in range(0, 32, 8):
        steps.append(
            {
                "group_ids": ids[start : start + 8],
                "learning_rate": lr,
                "max_replay_logp_difference": 0.0,
                "ratio_min": 0.9,
                "ratio_max": 1.1,
                "ratio_mean": 1.0,
            }
        )
    return {
        "window_index": index,
        "learning_rate": lr,
        "group_ids": ids,
        "steps": steps,
        "peak_reserved_gib": 10.0,
        "anchor_groups": 0,
        "gt_injection_count": 0,
        "k": 1,
    }


class VerifyTest(unittest.TestCase):
    def test_two_complete_windows_have_eight_updates(self) -> None:
        config = approved_config()
        result = validate_window_rows(
            config, [_window(config, 0), _window(config, 1)], expected_windows=2
        )
        self.assertEqual(result["optimizer_updates"], 8)
        self.assertEqual(result["group_occurrences"], 64)

    def test_duplicate_group_in_one_window_is_rejected(self) -> None:
        config = approved_config()
        row = _window(config, 0)
        row["group_ids"][-1] = row["group_ids"][0]
        with self.assertRaisesRegex(RuntimeError, "K=1"):
            validate_window_rows(config, [row], expected_windows=1)

    def test_group_rows_are_bound_to_their_window_ids(self) -> None:
        config = approved_config()
        window = _window(config, 0)
        groups = [
            {
                "window_index": 0,
                "group_id": group_id,
                "candidate_indices": list(range(16)),
                "candidate_sids": ["sid"] * 16,
                "candidate_token_ids": [[1]] * 16,
                "rewards": [0.0] * 16,
                "reward_tiers": ["same_domain"] * 16,
                "advantages": [0.0] * 16,
                "effective": True,
                "anchor_groups": 0,
                "gt_injection_count": 0,
                "k": 1,
            }
            for group_id in window["group_ids"]
        ]
        self.assertEqual(validate_group_rows([window], groups)["group_rows"], 32)
        groups[-1]["group_id"] = groups[0]["group_id"]
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            validate_group_rows([window], groups)

    def test_final_adapter_is_parsed_and_all_tensors_must_be_finite(self) -> None:
        class FakeSafeOpen:
            def __init__(self, tensors):
                self.tensors = tensors

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def keys(self):
                return self.tensors.keys()

            def get_tensor(self, key):
                return self.tensors[key]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_root = root / "reference"
            reference_root.mkdir()
            (root / "adapter_model.safetensors").write_bytes(b"not-empty")
            (reference_root / "adapter_model.safetensors").write_bytes(b"reference")
            adapter = SimpleNamespace(
                rank=64,
                alpha=64,
                source_dropout=0.0,
            )
            tensors = {"a": torch.ones(2), "b": torch.ones(3)}
            reference_tensors = {"a": torch.zeros(2), "b": torch.ones(3)}

            def open_adapter(path, **_kwargs):
                selected = (
                    reference_tensors
                    if Path(path).parent == reference_root
                    else tensors
                )
                return FakeSafeOpen(selected)

            with (
                patch(
                    "ksllm4rec_rloo_dapo.verify.validate_adapter_contract",
                    return_value=adapter,
                ),
                patch(
                    "ksllm4rec_rloo_dapo.verify.safe_open",
                    side_effect=open_adapter,
                ),
                patch.object(contract, "EXPECTED_LORA_TENSOR_COUNT", 2),
                patch.object(contract, "EXPECTED_LORA_PARAMETER_COUNT", 5),
            ):
                report = validate_final_adapter(root, reference_root)
                self.assertEqual(report["tensor_count"], 2)
                self.assertEqual(report["parameter_count"], 5)
                self.assertEqual(report["changed_tensor_count"], 1)

                tensors["b"] = torch.tensor([float("nan"), 0.0, 1.0])
                with self.assertRaisesRegex(RuntimeError, "non-floating or non-finite"):
                    validate_final_adapter(root, reference_root)

    def test_adapter_config_target_order_is_semantically_irrelevant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(
                '{"r":64,"target_modules":["q_proj","k_proj"]}\n',
                encoding="utf-8",
            )
            right.write_text(
                '{"target_modules":["k_proj","q_proj"],"r":64}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                _normalized_adapter_config(left), _normalized_adapter_config(right)
            )


if __name__ == "__main__":
    unittest.main()
