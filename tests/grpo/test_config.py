import tempfile
import unittest
from pathlib import Path

from ksllm4rec_grpo.config import load_config
from ksllm4rec_grpo.contract import profile_for_config


class ConfigContractTest(unittest.TestCase):
    def test_approved_config_loads(self):
        value = load_config(Path("configs/grpo/onereason_lora_grpo.yaml"))
        self.assertEqual(value["rollout"]["num_generations"], 8)
        self.assertEqual(value["reward"]["normalization"], "zscore")
        self.assertEqual(
            value["trie"]["strategy"],
            "baseline_all_system_prompt_response_sids",
        )

    def test_changed_num_generations_is_rejected(self):
        source = Path("configs/grpo/onereason_lora_grpo.yaml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(source.replace("num_generations: 8", "num_generations: 4"))
            with self.assertRaisesRegex(ValueError, "differs from Spec V3.1"):
                load_config(path)

    def test_unknown_behavior_key_is_rejected(self):
        source = Path("configs/grpo/onereason_lora_grpo.yaml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(source + "\nunknown: true\n")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)

    def test_frontier_profile_loads_without_changing_algorithm(self):
        baseline = load_config(Path("configs/grpo/onereason_lora_grpo.yaml"))
        frontier = load_config(Path("configs/grpo/frontier_sft_epoch2_grpo.yaml"))
        profile = profile_for_config(frontier)
        self.assertEqual(profile.groups, 17_016)
        self.assertEqual(profile.positives, 30_465)
        self.assertEqual(profile.unique_sids, 905_469)
        for section in ("rollout", "reward", "loss", "train", "memory"):
            self.assertEqual(frontier[section], baseline[section])

    def test_frontier_profile_rejects_baseline_data(self):
        source = Path("configs/grpo/frontier_sft_epoch2_grpo.yaml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                source.replace(
                    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl",
                    "/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl",
                )
            )
            with self.assertRaisesRegex(ValueError, "frontier_sft_epoch2_v2"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
