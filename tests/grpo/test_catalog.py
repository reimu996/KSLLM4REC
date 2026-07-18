import unittest

from ksllm4rec_grpo.data import extract_sids_from_records
from ksllm4rec_sft.data import SourceRecord


class BaselineSidExtractionTest(unittest.TestCase):
    def test_extracts_complete_sids_from_all_three_fields_and_deduplicates(self):
        first = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
        second = "<|ad_begin|><s_a_4><s_b_5><s_c_6>"
        records = [
            SourceRecord(system=first, prompt=f"p {second}", response=f"r {first}"),
            SourceRecord(system="", prompt="none", response=second),
        ]
        result = extract_sids_from_records(records)
        self.assertEqual({sid.render() for sid in result}, {first, second})

    def test_ignores_incomplete_sid_fragments(self):
        records = [SourceRecord("<s_a_1><s_b_2>", "", "answer")]
        with self.assertRaisesRegex(RuntimeError, "No complete SID"):
            extract_sids_from_records(records)


if __name__ == "__main__":
    unittest.main()
