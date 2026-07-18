from __future__ import annotations

import unittest

from ksllm4rec_sft.item_tokens import EXPECTED_ITEM_TOKEN_COUNT, build_item_token_ids
from ksllm4rec_sft.profiles import BASELINE_PROFILE, FRONTIER_PROFILE
from ksllm4rec_sft.trainer import dataset_loss_metric_name


class FakeTokenizer:
    vocab_size = 100

    def __len__(self) -> int:
        return self.vocab_size + EXPECTED_ITEM_TOKEN_COUNT

    def get_added_vocab(self) -> dict[str, int]:
        return {
            f"token_{index}": self.vocab_size + index
            for index in range(EXPECTED_ITEM_TOKEN_COUNT)
        }


class ItemTokenIdsTest(unittest.TestCase):
    def test_uses_the_entire_contiguous_added_vocabulary(self) -> None:
        ids = build_item_token_ids(FakeTokenizer())
        self.assertEqual(len(ids), EXPECTED_ITEM_TOKEN_COUNT)
        self.assertEqual(ids[0], 100)
        self.assertEqual(ids[-1], 100 + EXPECTED_ITEM_TOKEN_COUNT - 1)

    def test_rejects_a_gap_in_added_vocabulary(self) -> None:
        tokenizer = FakeTokenizer()
        added = tokenizer.get_added_vocab()
        added.pop("token_10")
        tokenizer.get_added_vocab = lambda: added  # type: ignore[method-assign]
        with self.assertRaisesRegex(ValueError, "Expected"):
            build_item_token_ids(tokenizer)


class DatasetMetricNameTest(unittest.TestCase):
    def test_metric_uses_the_selected_dataset_instead_of_a_global_constant(
        self,
    ) -> None:
        self.assertEqual(
            dataset_loss_metric_name(BASELINE_PROFILE.dataset_name),
            "loss_ds_hf_kuaishou_llmrec_sft_baseline_0_91",
        )
        self.assertEqual(
            dataset_loss_metric_name(FRONTIER_PROFILE.dataset_name),
            "loss_ds_frontier_feedbackcore_listwise_invariant_v1",
        )


if __name__ == "__main__":
    unittest.main()
