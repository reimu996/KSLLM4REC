import unittest

import torch

from ksllm4rec_grpo.constraint import (
    RecommendationGrammar,
    RecommendationLogitsProcessor,
)
from ksllm4rec_grpo.contract import EMPTY_THINK, RESPONSE_PREFIX
from ksllm4rec_grpo.trie import SidPrefixTrie
from ksllm4rec_orpo.data import Sid


class ToyTokenizer:
    eos_token_id = 999

    def __init__(self):
        self.text_ids = {
            EMPTY_THINK: [1, 2, 3, 4],
            '["': [30],
            '"]': [31],
            RESPONSE_PREFIX["video"]: [10, 11],
            RESPONSE_PREFIX["prod"]: [10, 13],
            RESPONSE_PREFIX["ad"]: [10, 12],
            RESPONSE_PREFIX["living"]: [10, 14],
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.text_ids:
            return self.text_ids[text]
        domains = {"video": 20, "prod": 22, "ad": 21, "living": 23}
        for domain, token_id in domains.items():
            if text == f"<|{domain}_begin|>":
                return [token_id]
        for level, offset in (("a", 1000), ("b", 10000), ("c", 20000)):
            prefix = f"<s_{level}_"
            if text.startswith(prefix):
                return [offset + int(text[len(prefix) : -1])]
        raise KeyError(text)


class RecommendationGrammarTest(unittest.TestCase):
    def setUp(self):
        self.sids = [
            Sid("video", 1, 2, 3),
            Sid("video", 1, 2, 9),
            Sid("ad", 9, 9, 9),
        ]
        self.grammar = RecommendationGrammar(
            ToyTokenizer(), SidPrefixTrie.from_sids(self.sids)
        )

    def test_full_completion_includes_empty_think_and_round_trips(self):
        sid = Sid("video", 1, 2, 3)
        encoded = self.grammar.encode_sid(sid)
        self.assertEqual(encoded[:4], [1, 2, 3, 4])
        self.assertEqual(encoded[-4:], [1001, 10002, 20003, 999])
        self.assertEqual(self.grammar.parse(encoded), sid)

    def test_allowed_values_follow_baseline_trie(self):
        prefix = list(self.grammar.prefix_tokens["video"])
        self.assertEqual(self.grammar.allowed_next(prefix), [1001])
        self.assertEqual(self.grammar.allowed_next(prefix + [1001]), [10002])
        self.assertEqual(
            self.grammar.allowed_next(prefix + [1001, 10002]), [20003, 20009]
        )

    def test_non_baseline_sid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-baseline"):
            self.grammar.encode_sid(Sid("video", 1, 2, 4))

    def test_decision_mask_only_marks_multiple_legal_actions(self):
        encoded = self.grammar.encode_sid(Sid("video", 1, 2, 3))
        mask = self.grammar.decision_mask(encoded)
        self.assertEqual(len(mask), len(encoded))
        self.assertTrue(any(mask))
        self.assertFalse(mask[0])
        self.assertFalse(mask[-1])

    def test_logits_processor_is_prompt_independent(self):
        generated = [*self.grammar.prefix_tokens["video"], 1001, 10002]
        input_ids = torch.tensor(
            [[71, 72, *generated], [81, 82, *generated]], dtype=torch.long
        )
        scores = torch.zeros((2, 21010), dtype=torch.float32)
        masked = RecommendationLogitsProcessor(self.grammar, prompt_length=2)(
            input_ids, scores
        )
        self.assertTrue(
            torch.equal(torch.isfinite(masked[0]), torch.isfinite(masked[1]))
        )
        self.assertEqual(
            torch.where(torch.isfinite(masked[0]))[0].tolist(), [20003, 20009]
        )
        torch.testing.assert_close(
            masked[0, [20003, 20009]].exp().sum(), torch.tensor(1.0)
        )

    def test_probe_modes_encode_their_exact_output_wrappers(self):
        sid = Sid("video", 1, 2, 3)
        text = RecommendationGrammar(
            ToyTokenizer(), self.grammar.trie, mode="probe_text_to_sid"
        )
        recommendation = RecommendationGrammar(
            ToyTokenizer(), self.grammar.trie, mode="probe_recommendation"
        )
        self.assertEqual(text.parse(text.encode_sid(sid)), sid)
        self.assertEqual(recommendation.parse(recommendation.encode_sid(sid)), sid)
        self.assertGreater(
            len(recommendation.encode_sid(sid)), len(text.encode_sid(sid))
        )


if __name__ == "__main__":
    unittest.main()
