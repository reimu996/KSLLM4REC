from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_sft.fingerprint import implementation_fingerprint
from ksllm4rec_sft.profiles import FRONTIER_PROFILE


class ImplementationFingerprintTest(unittest.TestCase):
    def test_content_change_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "src/ksllm4rec_sft"
            source_dir.mkdir(parents=True)
            source = source_dir / "module.py"
            source.write_text("before", encoding="utf-8")
            explicit = root / "required"
            explicit.write_text("fixed", encoding="utf-8")
            with patch("ksllm4rec_sft.fingerprint.EXPLICIT_PATHS", ("required",)):
                before = implementation_fingerprint(root)
                source.write_text("after", encoding="utf-8")
                after = implementation_fingerprint(root)
            self.assertNotEqual(before["sha256"], after["sha256"])

    def test_frontier_fingerprint_recursively_binds_frontier_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "src/ksllm4rec_sft"
            source_dir.mkdir(parents=True)
            (source_dir / "module.py").write_text("fixed", encoding="utf-8")
            required = root / "required"
            required.write_text("fixed", encoding="utf-8")
            script_dir = root / "scripts/sft/frontier"
            script_dir.mkdir(parents=True)
            script = script_dir / "run.sh"
            script.write_text("before", encoding="utf-8")
            profile = replace(
                FRONTIER_PROFILE,
                fingerprint_paths=("required",),
                fingerprint_script_dirs=("scripts/sft/frontier",),
            )
            before = implementation_fingerprint(root, profile)
            script.write_text("after", encoding="utf-8")
            after = implementation_fingerprint(root, profile)
            self.assertNotEqual(before["sha256"], after["sha256"])
