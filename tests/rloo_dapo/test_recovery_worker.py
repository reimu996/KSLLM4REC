from __future__ import annotations

from pathlib import Path
import unittest

from ksllm4rec_rloo_dapo import contract
from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.recovery_worker import _approved_output


class RecoveryWorkerTest(unittest.TestCase):
    def test_only_fixed_or_well_formed_staging_output_is_accepted(self) -> None:
        config = approved_config(contract.SFT372_PROFILE)
        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        fixed = profile.pilot_dir.with_name(f"{profile.pilot_dir.name}-resume-check")
        staged = (
            profile.log_dir
            / ".gates.staging.20260724T120102Z.12345"
            / "pilot-resume-check"
        )
        self.assertEqual(_approved_output(config, fixed), fixed.resolve())
        self.assertEqual(_approved_output(config, staged), staged.resolve())
        for invalid in (
            profile.log_dir / ".gates.staging.bad" / "pilot-resume-check",
            profile.log_dir / ".gates.staging.20260724T120102Z.12345" / "pilot",
            Path("/tmp/outside-resume-check"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                _approved_output(config, invalid)


if __name__ == "__main__":
    unittest.main()
