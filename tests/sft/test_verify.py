from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_sft.verify import _latest_full_manifest, _verify_manifest_fingerprint
from ksllm4rec_sft.profiles import FRONTIER_PROFILE


class ManifestFingerprintVerificationTest(unittest.TestCase):
    def test_manifest_selection_is_bound_to_the_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_a = root / "output-a"
            output_b = root / "output-b"
            for index, output in enumerate((output_a, output_b)):
                run = root / f"run-{index}"
                run.mkdir()
                (run / "manifest.json").write_text(
                    json.dumps(
                        {
                            "stage": "full_epoch_001",
                            "status": "passed",
                            "updated_at": f"2026-07-14T00:00:0{index}+08:00",
                            "resolved_config": {"output_dir": str(output)},
                        }
                    ),
                    encoding="utf-8",
                )
            path, manifest = _latest_full_manifest(root, output_a)
            self.assertEqual(path.parent.name, "run-0")
            self.assertEqual(Path(manifest["resolved_config"]["output_dir"]), output_a)

    def test_accepts_current_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            current = {"sha256": "current", "files": {}}
            with patch(
                "ksllm4rec_sft.verify.implementation_fingerprint",
                return_value=current,
            ):
                result = _verify_manifest_fingerprint(
                    {"implementation_fingerprint": {"sha256": "current"}},
                    Path(temp_dir),
                )
        self.assertEqual(result, current)

    def test_rejects_stale_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "ksllm4rec_sft.verify.implementation_fingerprint",
                    return_value={"sha256": "current", "files": {}},
                ),
                self.assertRaisesRegex(RuntimeError, "manifest='stale'"),
            ):
                _verify_manifest_fingerprint(
                    {"implementation_fingerprint": {"sha256": "stale"}},
                    Path(temp_dir),
                )

    def test_frontier_manifest_selection_requires_profile_stage_and_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifests = [
                {
                    "profile": "baseline",
                    "stage": "full_epoch_001",
                    "status": "passed",
                    "updated_at": "2026-07-20T00:00:03+08:00",
                    "resolved_config": {"output_dir": str(output)},
                },
                {
                    "profile": FRONTIER_PROFILE.name,
                    "stage": "full_epoch_001",
                    "status": "passed",
                    "updated_at": "2026-07-20T00:00:02+08:00",
                    "resolved_config": {"output_dir": str(output)},
                },
                {
                    "profile": FRONTIER_PROFILE.name,
                    "stage": FRONTIER_PROFILE.full_stage,
                    "status": "passed",
                    "updated_at": "2026-07-20T00:00:01+08:00",
                    "resolved_config": {"output_dir": str(output)},
                },
            ]
            for index, manifest in enumerate(manifests):
                run = root / f"run-{index}"
                run.mkdir()
                (run / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            path, selected = _latest_full_manifest(root, output, FRONTIER_PROFILE)
            self.assertEqual(path.parent.name, "run-2")
            self.assertEqual(selected["stage"], FRONTIER_PROFILE.full_stage)


if __name__ == "__main__":
    unittest.main()
