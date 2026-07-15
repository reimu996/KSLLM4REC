from __future__ import annotations

import unittest

from ksllm4rec_orpo.data import IndexedRecord, PairTask, Sid, sha256_text
from ksllm4rec_orpo.miner import _assert_audit_row, _longest_common_prefix
from ksllm4rec_orpo.pairs import (
    SidCatalog,
    SidCatalogReport,
    _ceval_candidates,
    _user_rejected_map,
)
from ksllm4rec_sft.data import SourceRecord


def empty_sid_report() -> SidCatalogReport:
    return SidCatalogReport(0, 0, 0, 0, 0, 0, 0, 0)


class SidCandidateSelectionTest(unittest.TestCase):
    def test_uses_hardest_available_tier_and_excludes_positives(self) -> None:
        chosen = Sid("prod", 1, 2, 3)
        positive = Sid("prod", 1, 2, 4)
        negative = Sid("prod", 1, 2, 5)
        catalog = SidCatalog(
            by_ab={chosen.ab_key: [chosen, positive, negative]},
            by_a={chosen.a_key: [Sid("prod", 1, 9, 8)]},
            by_domain={"prod": [Sid("prod", 7, 8, 9)]},
            report=empty_sid_report(),
        )
        tier, reason, values = catalog.candidates(
            chosen, {chosen.render(), positive.render()}
        )
        self.assertEqual(tier, 1)
        self.assertEqual(reason, "same_ab_wrong_c")
        self.assertEqual(values, [negative])

    def test_falls_back_to_same_domain(self) -> None:
        chosen = Sid("video", 1, 2, 3)
        fallback = Sid("video", 8, 9, 10)
        catalog = SidCatalog(
            by_ab={},
            by_a={},
            by_domain={"video": [chosen, fallback]},
            report=empty_sid_report(),
        )
        tier, reason, values = catalog.candidates(chosen, {chosen.render()})
        self.assertEqual((tier, reason, values), (3, "same_domain_wrong_sid", [fallback]))


class UserRejectedSelectionTest(unittest.TestCase):
    def test_cross_user_rejected_has_disjoint_sids(self) -> None:
        sid1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
        sid2 = "<|video_begin|><s_a_4><s_b_5><s_c_6>"
        records = [
            IndexedRecord(
                1,
                "a",
                SourceRecord("", sid1, f'<think></think>{{"event":"{sid1}"}}'),
                PairTask.USER_INTEREST,
            ),
            IndexedRecord(
                2,
                "b",
                SourceRecord("", sid2, f'<think></think>{{"event":"{sid2}"}}'),
                PairTask.USER_INTEREST,
            ),
        ]
        mapping = _user_rejected_map(records)
        self.assertEqual(mapping[1][1], 2)
        self.assertEqual(mapping[2][1], 1)


class CandidateMiningUtilitiesTest(unittest.TestCase):
    def test_finds_first_divergent_token(self) -> None:
        self.assertEqual(_longest_common_prefix([[1, 2, 3], [1, 2, 4]]), 2)

    def test_ceval_candidates_only_change_correct_letter(self) -> None:
        chosen = "<think>\n</think>\n正确答案是 (B)"
        values = _ceval_candidates(chosen)
        self.assertEqual(len(values), 3)
        self.assertEqual({value[-2] for value in values}, {"A", "C", "D"})

    def test_audit_rejects_known_positive(self) -> None:
        sid = "<|prod_begin|><s_a_1><s_b_2><s_c_3>"
        response = f"<think>\n</think>\n{sid}"
        row = {
            "pair_id": "pair",
            "task": PairTask.TEXT_TO_SID.value,
            "chosen": response,
            "rejected": response.replace("s_c_3", "s_c_4"),
            "positive_set": {response.replace("s_c_3", "s_c_4").splitlines()[-1]},
            "original_final_suffix_sha256": sha256_text("\n" + sid),
            "instruction": "caption/no_think",
        }
        with self.assertRaisesRegex(RuntimeError, "known positive"):
            _assert_audit_row(row)


if __name__ == "__main__":
    unittest.main()
