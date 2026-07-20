import unittest

from ksllm4rec_grpo import fingerprint


class FingerprintCoverageTest(unittest.TestCase):
    def test_profile_contract_is_bound_to_training_and_probe_signatures(self) -> None:
        relative_path = "ksllm4rec_grpo/profiles.py"

        self.assertIn(relative_path, fingerprint._RUNTIME_CODE_FILES)
        self.assertIn(relative_path, fingerprint._PROBE_CODE_FILES)


if __name__ == "__main__":
    unittest.main()
