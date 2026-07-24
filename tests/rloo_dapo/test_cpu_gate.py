from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from ksllm4rec_rloo_dapo import cpu_gate


class CPUGateReportTest(unittest.TestCase):
    SIGNATURE = {"sha256": "a" * 64}

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / "cpu_tests.log"
        self.log.write_text("all five CPU suites passed\n", encoding="utf-8")
        suites = []
        total_tests = 0
        project_root = Path(__file__).resolve().parents[2]
        for index, suite in enumerate(cpu_gate.CPU_SUITES, start=1):
            test_count = index
            total_tests += test_count
            suites.append(
                {
                    "suite": suite,
                    "returncode": 0,
                    "test_count": test_count,
                    "test_file_count": len(
                        tuple((project_root / "tests" / suite).glob("test_*.py"))
                    ),
                    "elapsed_seconds": float(index),
                }
            )
        self.report = {
            "schema_version": cpu_gate.CPU_GATE_SCHEMA_VERSION,
            "gate": "cpu",
            "passed": True,
            "runtime_signature_sha256": self.SIGNATURE["sha256"],
            "suite_order": list(cpu_gate.CPU_SUITES),
            "suites": suites,
            "total_tests": total_tests,
            "elapsed_seconds": 15.0,
            "log_path": str(self.log.resolve()),
            "log_sha256": hashlib.sha256(self.log.read_bytes()).hexdigest(),
        }
        self.report_path = self.root / "cpu_gate.json"
        self._write(self.report)

    def _write(self, report: dict[str, object], *, allow_nan: bool = False) -> None:
        self.report_path.write_text(
            json.dumps(report, allow_nan=allow_nan) + "\n", encoding="utf-8"
        )

    def test_complete_content_bound_report_is_accepted(self) -> None:
        actual = cpu_gate.validate_cpu_gate_report(
            self.SIGNATURE, self.report_path, expected_log=self.log
        )
        self.assertEqual(actual, self.report)

    def test_missing_suite_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["suites"] = report["suites"][:-1]
        self._write(report)
        with self.assertRaisesRegex(RuntimeError, "every suite"):
            cpu_gate.validate_cpu_gate_report(
                self.SIGNATURE, self.report_path, expected_log=self.log
            )

    def test_non_finite_number_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["elapsed_seconds"] = math.nan
        self._write(report, allow_nan=True)
        with self.assertRaises((ValueError, RuntimeError)):
            cpu_gate.validate_cpu_gate_report(
                self.SIGNATURE, self.report_path, expected_log=self.log
            )

    def test_tampered_log_is_rejected(self) -> None:
        self.log.write_text("changed after publication\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            cpu_gate.validate_cpu_gate_report(
                self.SIGNATURE, self.report_path, expected_log=self.log
            )

    def test_wrong_runtime_signature_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stale"):
            cpu_gate.validate_cpu_gate_report(
                {"sha256": "b" * 64}, self.report_path, expected_log=self.log
            )


if __name__ == "__main__":
    unittest.main()
