from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

from ksllm4rec_rloo_dapo import contract


SCRIPT_SOURCE = (
    contract.PROJECT_ROOT
    / "scripts/rloo_dapo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_t12"
)


class StatusScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        relative_script = SCRIPT_SOURCE.relative_to(contract.PROJECT_ROOT)
        self.script_dir = self.root / relative_script
        self.script_dir.mkdir(parents=True)
        for name in ("common.sh", "status.sh"):
            shutil.copy2(SCRIPT_SOURCE / name, self.script_dir / name)

        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        self.log_root = self.root / profile.log_dir.relative_to(contract.PROJECT_ROOT)
        self.run_dir = self.root / profile.run_dir.relative_to(contract.PROJECT_ROOT)
        self.train_log = self.log_root / "full_train.stdout.log"
        self.lock_path = self.log_root / "full_train.lock"
        self.log_root.mkdir(parents=True)
        self.run_dir.mkdir(parents=True)
        self.lock_process: subprocess.Popen[str] | None = None

    def tearDown(self) -> None:
        if self.lock_process is not None:
            self.lock_process.terminate()
            self.lock_process.wait(timeout=5)
        self.temporary.cleanup()

    def _write_window(
        self, completed: int, *, optimizer_update_step: int | None = None
    ) -> None:
        value = {
            "window_index": completed - 1,
            "optimizer_update_step": (
                completed * 4
                if optimizer_update_step is None
                else optimizer_update_step
            ),
            "window_seconds": 30.0,
            "raw_prompt_count": 40,
            "peak_reserved_gib": 18.0,
        }
        (self.run_dir / "windows.jsonl").write_text(
            json.dumps(value) + "\n", encoding="utf-8"
        )

    def _write_invocation(self, *, exit_code: int | None = None) -> None:
        rows = [{"event": "train_invocation_start", "unix_time": 1}]
        if exit_code is not None:
            rows.append(
                {
                    "event": "train_invocation_end",
                    "unix_time": 2,
                    "exit_code": exit_code,
                }
            )
        self.train_log.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_complete_summary(self) -> None:
        (self.run_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "completed_windows": 1064,
                    "optimizer_update_step": 4256,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def _hold_lock(self) -> None:
        ready = self.root / "lock-ready"
        command = (
            f"exec 9>'{self.lock_path}'; flock -x 9; "
            f"touch '{ready}'; sleep 30"
        )
        self.lock_process = subprocess.Popen(["bash", "-c", command], text=True)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "test process did not acquire the training lock")

    def _status(self) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        result = subprocess.run(
            [str(self.script_dir / "status.sh")],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertTrue(result.stdout.strip(), result.stderr)
        return result, json.loads(result.stdout)

    def test_final_window_without_summary_is_not_success(self) -> None:
        self._write_window(1064)
        self._write_invocation()

        result, report = self._status()

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["success"])
        self.assertEqual(report["state"], "incomplete")

    def test_inactive_incomplete_run_is_failure(self) -> None:
        self._write_window(12)
        self._write_invocation(exit_code=1)

        result, report = self._status()

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["success"])
        self.assertEqual(report["state"], "failed")

    def test_active_stalled_run_is_failure(self) -> None:
        self._write_window(12)
        self._write_invocation()
        old = time.time() - 901
        os.utime(self.train_log, (old, old))
        self._hold_lock()

        result, report = self._status()

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(report["stalled"])
        self.assertEqual(report["state"], "stalled")

    def test_active_after_last_window_is_finalizing(self) -> None:
        self._write_window(1064)
        self._write_invocation()
        self._hold_lock()

        result, report = self._status()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(report["success"])
        self.assertEqual(report["state"], "finalizing")

    def test_complete_run_requires_summary_and_zero_exit(self) -> None:
        self._write_window(1064)
        self._write_complete_summary()
        self._write_invocation(exit_code=0)

        result, report = self._status()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["success"])
        self.assertEqual(report["state"], "complete")

    def test_complete_summary_without_wrapper_exit_is_not_success(self) -> None:
        self._write_window(1064)
        self._write_complete_summary()
        self._write_invocation()

        result, report = self._status()

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["success"])
        self.assertEqual(report["state"], "incomplete")

    def test_nonzero_latest_exit_is_failure(self) -> None:
        self._write_window(1064)
        self._write_complete_summary()
        self._write_invocation(exit_code=1)

        result, report = self._status()

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["success"])
        self.assertEqual(report["state"], "failed")

    def test_last_window_update_must_match_summary(self) -> None:
        self._write_window(1064, optimizer_update_step=4255)
        self._write_complete_summary()
        self._write_invocation(exit_code=0)

        result, report = self._status()

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["success"])
        self.assertEqual(report["state"], "incomplete")

    def test_old_invocation_errors_are_ignored(self) -> None:
        self._write_window(1064)
        self._write_complete_summary()
        rows = (
            '{"event":"train_invocation_start","unix_time":1}\n'
            "Traceback (most recent call last): old invocation\n"
            '{"event":"train_invocation_end","unix_time":2,"exit_code":1}\n'
            '{"event":"train_invocation_start","unix_time":3}\n'
            '{"event":"train_invocation_end","unix_time":4,"exit_code":0}\n'
        )
        self.train_log.write_text(rows, encoding="utf-8")

        result, report = self._status()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["success"])
        self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
