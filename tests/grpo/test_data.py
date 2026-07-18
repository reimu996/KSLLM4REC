import unittest

from ksllm4rec_grpo.data import group_recommendation_records
from ksllm4rec_orpo.data import IndexedRecord, PairTask
from ksllm4rec_sft.data import SourceRecord


class RecommendationGroupingTest(unittest.TestCase):
    @staticmethod
    def row(line, mode, sid):
        record = SourceRecord(
            system="recommend system",
            prompt=f"same prompt/{mode}",
            response=f"<think>reason</think>\nanswer: {sid}",
        )
        return IndexedRecord(line, f"sha-{line}", record, PairTask.RECOMMEND)

    def test_mode_variants_merge_and_keep_all_positives(self):
        rows = [
            self.row(1, "think", "<|video_begin|><s_a_1><s_b_2><s_c_3>"),
            self.row(2, "no_think", "<|video_begin|><s_a_1><s_b_2><s_c_4>"),
        ]
        groups = group_recommendation_records(rows, enforce_contract=False)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].positive_sids), 2)
        self.assertEqual(groups[0].source_lines, (1, 2))
        self.assertNotIn("completion", groups[0].to_dict())

    def test_multiple_systems_for_one_prompt_are_rejected(self):
        rows = [self.row(1, "think", "<|video_begin|><s_a_1><s_b_2><s_c_3>")]
        other = self.row(2, "no_think", "<|video_begin|><s_a_1><s_b_2><s_c_4>")
        rows.append(
            IndexedRecord(
                other.line_number,
                other.source_sha256,
                SourceRecord(
                    "other system", other.record.prompt, other.record.response
                ),
                other.task,
            )
        )
        with self.assertRaisesRegex(ValueError, "multiple systems"):
            group_recommendation_records(rows, enforce_contract=False)


if __name__ == "__main__":
    unittest.main()
