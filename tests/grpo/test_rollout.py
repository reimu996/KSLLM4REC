import hashlib
import unittest

from ksllm4rec_grpo.rollout import rollout_seed


class RolloutSeedTest(unittest.TestCase):
    def test_seed_matches_frozen_payload(self):
        payload = b"rollout|42|2|group-abc|7"
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.assertEqual(rollout_seed("group-abc", 2, 7), expected)

    def test_candidate_seeds_are_independent(self):
        values = {rollout_seed("g", 1, index) for index in range(8)}
        self.assertEqual(len(values), 8)


if __name__ == "__main__":
    unittest.main()
