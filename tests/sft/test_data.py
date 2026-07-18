from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ksllm4rec_sft.data import (
    expected_dataset_info,
    iter_derived_records,
    prepare_dataset,
    render_qwen3_nothink,
    validate_dataset_info,
)


class DataPreparationTest(unittest.TestCase):
    def test_conversion_preserves_all_three_strings(self) -> None:
        rows = [
            [
                {
                    "system": "sys",
                    "prompt": "p/no_think",
                    "response": "<think>r</think>answer",
                }
            ],
            [{"system": "", "prompt": "item", "response": "<s_a_1>"}],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = prepare_dataset(source, root / "derived")
            converted = list(iter_derived_records(Path(report.derived_path)))
            self.assertEqual(report.records, 2)
            self.assertEqual(report.content_sha256, report.derived_content_sha256)
            self.assertEqual(converted[0].response, rows[0][0]["response"])
            source_text, target_text = render_qwen3_nothink(converted[0])
            self.assertIn(rows[0][0]["prompt"], source_text)
            self.assertEqual(target_text, rows[0][0]["response"] + "<|im_end|>\n")

    def test_conversion_registers_the_selected_dataset_name(self) -> None:
        row = [[{"system": "s", "prompt": "p", "response": "r"}]]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(json.dumps(row[0]) + "\n", encoding="utf-8")
            prepare_dataset(source, root / "derived", dataset_name="frontier_dataset")
            dataset_info = json.loads(
                (root / "derived/dataset_info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(dataset_info), {"frontier_dataset"})

    def test_dataset_info_structure_is_exactly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset_info.json"
            path.write_text(
                json.dumps(expected_dataset_info("frontier_dataset", "train.jsonl")),
                encoding="utf-8",
            )
            validate_dataset_info(
                path,
                dataset_name="frontier_dataset",
                derived_filename="train.jsonl",
            )
            path.write_text('{"frontier_dataset": {}}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly match"):
                validate_dataset_info(
                    path,
                    dataset_name="frontier_dataset",
                    derived_filename="train.jsonl",
                )

    def test_invalid_structure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(
                '{"system":"","prompt":"p","response":"r"}\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                prepare_dataset(source, root / "derived")


if __name__ == "__main__":
    unittest.main()
