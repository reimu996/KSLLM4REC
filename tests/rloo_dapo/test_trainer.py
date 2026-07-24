from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ksllm4rec_rloo.integrity import canonical_sha256
from ksllm4rec_rloo_dapo.config import approved_config
import ksllm4rec_rloo_dapo.trainer as trainer
from ksllm4rec_rloo_dapo.trainer import (
    WindowLRScheduler,
    configure_deterministic_runtime,
    learning_rate_for_window,
    partition_effective_groups,
)


@dataclass(frozen=True)
class _Source:
    group_id: str


@dataclass(frozen=True)
class _Prepared:
    group: _Source


class WindowScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = approved_config()

    def test_linear_warmup_is_counted_in_complete_windows(self) -> None:
        expected = {
            0: 0.0,
            1: 1.0e-7,
            9: 9.0e-7,
            10: 1.0e-6,
            1063: 1.0e-6,
        }
        for window, value in expected.items():
            with self.subTest(window=window):
                self.assertAlmostEqual(
                    learning_rate_for_window(self.config, window), value, places=15
                )

    def test_deterministic_runtime_enables_flash_and_torch_controls(self) -> None:
        import os

        previous = torch.are_deterministic_algorithms_enabled()
        try:
            configure_deterministic_runtime(42)
            self.assertEqual(os.environ["FLASH_ATTENTION_DETERMINISTIC"], "1")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_scheduler_steps_once_after_four_optimizer_updates(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=9.0)
        scheduler = WindowLRScheduler(optimizer, self.config)
        self.assertEqual(scheduler.completed_windows, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        # The trainer may call optimizer.step four times; LR remains unchanged.
        for _ in range(4):
            optimizer.step()
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        scheduler.step()
        self.assertEqual(scheduler.completed_windows, 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0e-7)


class MinibatchPartitionTest(unittest.TestCase):
    def test_every_group_occurs_once_across_four_minibatches(self) -> None:
        groups = tuple(_Prepared(_Source(f"g{index:02d}")) for index in range(32))
        batches = partition_effective_groups(groups, window_index=7, seed=42)
        self.assertEqual([len(batch) for batch in batches], [8, 8, 8, 8])
        ids = [item.group.group_id for batch in batches for item in batch]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {item.group.group_id for item in groups})
        self.assertEqual(
            batches,
            partition_effective_groups(groups, window_index=7, seed=42),
        )


def _signature() -> dict:
    inputs = {"test": "run-directory"}
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class RunDirectoryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = approved_config()
        self.signature = _signature()

    def _write_contract(
        self,
        root: Path,
        *,
        signature: object | None = None,
        config: object | None = None,
    ) -> None:
        if signature is not None:
            (root / "runtime_signature.json").write_text(
                json.dumps(signature), encoding="utf-8"
            )
        if config is not None:
            (root / "resolved_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

    def test_unclaimed_non_empty_directory_is_rejected_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "existing.bin").write_bytes(b"do-not-touch")
            before = _file_snapshot(output)

            with self.assertRaisesRegex(RuntimeError, "runtime_signature.json"):
                trainer._prepare_run_directory(
                    output, self.config, self.signature, resume=True
                )

            self.assertEqual(_file_snapshot(output), before)

    def test_missing_epoch_adapter_is_backfilled_only_at_exact_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle = object()
            with patch.object(trainer, "save_policy_atomic") as save:
                trainer._ensure_epoch_adapters(
                    bundle,
                    output,
                    completed_windows=trainer.contract.WINDOWS_PER_EFFECTIVE_EPOCH,
                )
            save.assert_called_once_with(bundle, output / "epoch-01-adapter")
            with self.assertRaisesRegex(RuntimeError, "cannot be reconstructed"):
                trainer._ensure_epoch_adapters(
                    bundle,
                    output,
                    completed_windows=(
                        trainer.contract.WINDOWS_PER_EFFECTIVE_EPOCH + 1
                    ),
                )

    def test_both_contract_files_are_required_without_writes(self) -> None:
        for present in ("signature", "config"):
            with (
                self.subTest(present=present),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory)
                self._write_contract(
                    output,
                    signature=self.signature if present == "signature" else None,
                    config=self.config if present == "config" else None,
                )
                before = _file_snapshot(output)

                with self.assertRaisesRegex(RuntimeError, "missing"):
                    trainer._prepare_run_directory(
                        output, self.config, self.signature, resume=True
                    )

                self.assertEqual(_file_snapshot(output), before)

    def test_malformed_or_mismatched_contract_is_rejected_without_writes(self) -> None:
        cases = {
            "malformed": (b"{", self.config),
            "signature_mismatch": ({"sha256": "wrong"}, self.config),
            "config_mismatch": (self.signature, {"profile": "wrong"}),
        }
        for name, (signature, config) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                if isinstance(signature, bytes):
                    (output / "runtime_signature.json").write_bytes(signature)
                else:
                    self._write_contract(output, signature=signature)
                self._write_contract(output, config=config)
                (output / "windows.jsonl").write_bytes(b'{"partial":true}\n')
                before = _file_snapshot(output)

                with self.assertRaises(RuntimeError):
                    trainer._prepare_run_directory(
                        output, self.config, self.signature, resume=True
                    )

                self.assertEqual(_file_snapshot(output), before)

    def test_matching_contract_allows_resume_without_rewriting_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "runtime_signature.json").write_text(
                json.dumps(self.signature, separators=(",", ":")), encoding="utf-8"
            )
            (output / "resolved_config.json").write_text(
                json.dumps(self.config, separators=(",", ":")), encoding="utf-8"
            )
            (output / "existing.bin").write_bytes(b"preserve-format-and-bytes")
            before = _file_snapshot(output)

            trainer._prepare_run_directory(
                output, self.config, self.signature, resume=True
            )

            self.assertEqual(_file_snapshot(output), before)

    def test_resume_false_rejects_even_matching_non_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._write_contract(output, signature=self.signature, config=self.config)
            before = _file_snapshot(output)

            with self.assertRaises(FileExistsError):
                trainer._prepare_run_directory(
                    output, self.config, self.signature, resume=False
                )

            self.assertEqual(_file_snapshot(output), before)

    def test_new_directory_is_claimed_with_both_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"

            trainer._prepare_run_directory(
                output, self.config, self.signature, resume=True
            )

            self.assertEqual(
                json.loads((output / "runtime_signature.json").read_text()),
                self.signature,
            )
            self.assertEqual(
                json.loads((output / "resolved_config.json").read_text()),
                self.config,
            )

    def test_training_rejects_unclaimed_directory_before_log_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "windows.jsonl").write_bytes(b'{"partial":true}\n')
            before = _file_snapshot(output)

            with (
                patch.object(trainer, "configure_deterministic_runtime") as configure,
                patch.object(trainer, "_truncate_jsonl") as truncate,
                self.assertRaises(RuntimeError),
            ):
                trainer.run_training(
                    self.config,
                    self.signature,
                    output_dir=output,
                    device="cpu",
                )

            configure.assert_not_called()
            truncate.assert_not_called()
            self.assertEqual(_file_snapshot(output), before)


class FormalTrainingModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = approved_config()
        self.signature = _signature()
        self.formal_output = Path(self.config["output"]["run_dir"])

    def test_formal_mode_accepts_only_full_training_at_configured_path(self) -> None:
        expected = {"complete": False}
        with patch.object(trainer, "_run_training_impl", return_value=expected) as run:
            actual = trainer.run_training(
                self.config,
                self.signature,
                output_dir=self.formal_output,
                device="cuda:0",
                formal=True,
            )

        self.assertEqual(actual, expected)
        run.assert_called_once()

    def test_formal_mode_rejects_another_output_directory(self) -> None:
        with (
            patch.object(trainer, "_run_training_impl") as run,
            self.assertRaisesRegex(ValueError, "output_dir"),
        ):
            trainer.run_training(
                self.config,
                self.signature,
                output_dir=self.formal_output.parent / "wrong-run",
                formal=True,
            )
        run.assert_not_called()

    def test_formal_mode_rejects_stop_after_windows(self) -> None:
        with (
            patch.object(trainer, "_run_training_impl") as run,
            self.assertRaisesRegex(ValueError, "stop_after_windows"),
        ):
            trainer.run_training(
                self.config,
                self.signature,
                output_dir=self.formal_output,
                stop_after_windows=1,
                formal=True,
            )
        run.assert_not_called()

    def test_formal_mode_rejects_dense_scoring(self) -> None:
        with (
            patch.object(trainer, "_run_training_impl") as run,
            patch.object(trainer, "dense_scoring_mode") as dense_mode,
            self.assertRaisesRegex(ValueError, "dense_scoring"),
        ):
            trainer.run_training(
                self.config,
                self.signature,
                output_dir=self.formal_output,
                dense_scoring=True,
                formal=True,
            )
        run.assert_not_called()
        dense_mode.assert_not_called()

    def test_formal_mode_rejects_unaligned_cache_or_another_device(self) -> None:
        for name, fields in (
            ("unaligned", {"aligned_cache": False}),
            ("device", {"device": "cuda:1"}),
        ):
            with (
                self.subTest(name=name),
                patch.object(trainer, "_run_training_impl") as run,
                self.assertRaisesRegex(ValueError, "aligned_cache|device"),
            ):
                trainer.run_training(
                    self.config,
                    self.signature,
                    output_dir=self.formal_output,
                    formal=True,
                    **fields,
                )
            run.assert_not_called()

    def test_pilot_mode_keeps_arbitrary_stop_and_dense_options(self) -> None:
        expected = {"complete": False}
        with (
            patch.object(trainer, "_run_training_impl", return_value=expected) as run,
            patch.object(trainer, "dense_scoring_mode", return_value=nullcontext()),
        ):
            actual = trainer.run_training(
                self.config,
                self.signature,
                output_dir=Path("arbitrary-pilot"),
                stop_after_windows=1,
                dense_scoring=True,
            )

        self.assertEqual(actual, expected)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
