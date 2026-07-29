from __future__ import annotations

import unittest

from ksllm4rec_dapo_anchor_multitask_v1_1._infra.grpo_constraint import (
    RecommendationGrammar,
)
from ksllm4rec_dapo_anchor_multitask_v1_1._infra.grpo_contract import (
    EMPTY_THINK,
    RESPONSE_PREFIX,
)
from ksllm4rec_dapo_anchor_multitask_v1_1._infra.grpo_trie import SidPrefixTrie
from ksllm4rec_dapo_anchor_multitask_v1_1._infra.orpo_data import Sid


class ToyTokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        fixed = {
            EMPTY_THINK: [1, 2, 3, 4],
            RESPONSE_PREFIX["video"]: [10, 11],
            RESPONSE_PREFIX["prod"]: [10, 13],
            RESPONSE_PREFIX["ad"]: [10, 12],
            RESPONSE_PREFIX["living"]: [10, 14],
        }
        if text in fixed:
            return fixed[text]
        for domain, token_id in {
            "video": 20,
            "prod": 22,
            "ad": 21,
            "living": 23,
        }.items():
            if text == f"<|{domain}_begin|>":
                return [token_id]
        for level, offset in (("a", 1000), ("b", 10_000), ("c", 20_000)):
            prefix = f"<s_{level}_"
            if text.startswith(prefix):
                return [offset + int(text[len(prefix) : -1])]
        if text == "":
            return []
        raise KeyError(text)


class TextToSidGrammarTest(unittest.TestCase):
    def test_training_completion_contains_no_recommendation_renderer(self) -> None:
        sid = Sid("video", 1, 2, 3)
        trie = SidPrefixTrie.from_sids(
            [sid, Sid("video", 1, 2, 9), Sid("ad", 9, 9, 9)]
        )
        grammar = RecommendationGrammar(
            ToyTokenizer(), trie, mode="train_text_to_sid"
        )

        encoded = grammar.encode_sid(sid)
        self.assertEqual(encoded, [1, 2, 3, 4, 20, 1001, 10002, 20003, 999])
        self.assertEqual(grammar.parse(encoded), sid)
        mask = grammar.decision_mask(encoded)
        self.assertFalse(any(mask[:4]))
        self.assertFalse(mask[-1])
        self.assertTrue(mask[4])
        self.assertTrue(mask[7])


if __name__ == "__main__":
    unittest.main()

