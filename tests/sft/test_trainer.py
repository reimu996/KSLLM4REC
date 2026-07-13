from __future__ import annotations

import unittest

from ksllm4rec_sft.item_tokens import EXPECTED_ITEM_TOKEN_COUNT, build_item_token_ids


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


if __name__ == "__main__":
    unittest.main()
