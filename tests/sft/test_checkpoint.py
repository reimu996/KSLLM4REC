from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ksllm4rec_sft.checkpoint import (
    REQUIRED_CHECKPOINT_FILES,
    resolve_resume_checkpoint,
)


def write_checkpoint(root: Path, step: int, *, omit: str | None = None) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    for name in REQUIRED_CHECKPOINT_FILES:
        if name == omit:
            continue
        if name == "trainer_state.json":
            payload = json.dumps({"global_step": step})
        elif name == "adapter_config.json":
            payload = "{}"
        else:
            payload = name
        (checkpoint / name).write_text(payload, encoding="utf-8")
    return checkpoint


class CheckpointResolutionTest(unittest.TestCase):
    def test_empty_output_starts_from_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(resolve_resume_checkpoint(Path(temp_dir)))

    def test_selects_the_highest_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_checkpoint(root, 64)
            latest = write_checkpoint(root, 256)
            result = resolve_resume_checkpoint(root)
            self.assertEqual(result["path"], str(latest.resolve()))
            self.assertEqual(result["step"], 256)

    def test_rejects_a_corrupt_latest_checkpoint_without_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_checkpoint(root, 64)
            write_checkpoint(root, 256, omit="optimizer.pt")
            with self.assertRaisesRegex(RuntimeError, "optimizer.pt"):
                resolve_resume_checkpoint(root)

    def test_rejects_non_checkpoint_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                resolve_resume_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
