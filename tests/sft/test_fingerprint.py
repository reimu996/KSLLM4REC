from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_sft.fingerprint import implementation_fingerprint


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
