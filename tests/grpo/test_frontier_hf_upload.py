from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = PROJECT_ROOT / "scripts/grpo/prepare_frontier_hf_upload.py"
UPLOADER_SCRIPT = PROJECT_ROOT / "scripts/orpo/upload_hf_adapters.py"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_frontier_hf_upload", PREPARE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import upload preparation script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_uploader_module():
    spec = importlib.util.spec_from_file_location(
        "upload_hf_adapters", UPLOADER_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import uploader script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrontierHFUploadContractTest(unittest.TestCase):
    def make_run(self, root: Path, *, same_config: bool = True) -> Path:
        run = root / "artifacts/grpo/runs/frontier_run"
        for epoch in (1, 2):
            directory = run / f"epoch_{epoch:03d}"
            directory.mkdir(parents=True)
            (directory / "adapter_model.safetensors").write_bytes(
                f"epoch-{epoch}-weights".encode("ascii")
            )
            config = (
                b'{"peft_type":"LORA","r":32,"lora_alpha":32,'
                b'"lora_dropout":0.0}\n'
            )
            if epoch == 2 and not same_config:
                config = (
                    b'{"peft_type":"LORA","r":32,"lora_alpha":32,'
                    b'"revision":"different",'
                    b'"lora_dropout":0.0}\n'
                )
            (directory / "adapter_config.json").write_bytes(config)
        return run

    def write_template(self, root: Path) -> Path:
        path = root / "configs/grpo/readme.tmpl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# epoch {{EPOCH}}\n", encoding="utf-8")
        return path

    def run_prepare(self, root: Path, run: Path, template: Path, output: Path):
        return subprocess.run(
            [
                sys.executable,
                str(PREPARE_SCRIPT),
                "--project-root",
                str(root),
                "--run-dir",
                str(run),
                "--template",
                str(template),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_generates_strict_config_from_actual_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_run(root)
            template = self.write_template(root)
            output = root / "configs/grpo/hf_upload.json"
            result = self.run_prepare(root, run, template, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("remote_upload=not_performed", result.stdout)
            config = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["repo_id"] for item in config["repositories"]],
                [
                    "reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch1",
                    "reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch2",
                ],
            )
            self.assertEqual(config["provenance"]["recommendation_groups"], 17016)
            self.assertEqual(
                config["required_uploaded_files"],
                ["README.md", "adapter_config.json", "adapter_model.safetensors"],
            )
            for epoch, item in enumerate(config["repositories"], start=1):
                model = run / f"epoch_{epoch:03d}" / "adapter_model.safetensors"
                digest = hashlib.sha256(model.read_bytes()).hexdigest()
                self.assertEqual(item["adapter_model"]["sha256"], digest)
                self.assertEqual(item["adapter_model"]["size"], model.stat().st_size)

            # The existing uploader is configuration-driven.  Its strict
            # loader must accept the generated contract without contacting HF.
            uploader = load_uploader_module()
            self.assertEqual(
                uploader.load_config(output)["profile"],
                "frontier_sft_epoch2_grpo_v2",
            )

    def test_refuses_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_run(root)
            template = self.write_template(root)
            output = root / "configs/grpo/hf_upload.json"
            first = self.run_prepare(root, run, template, output)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            second = self.run_prepare(root, run, template, output)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing overwrite", second.stderr)
            self.assertEqual(output.read_bytes(), original)

    def test_rejects_mixed_adapter_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_run(root, same_config=False)
            template = self.write_template(root)
            output = root / "configs/grpo/hf_upload.json"
            result = self.run_prepare(root, run, template, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("adapter_config.json files differ", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
