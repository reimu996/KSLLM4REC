from __future__ import annotations

import unittest

from ksllm4rec_orpo.data import (
    EMPTY_THINK,
    PairTask,
    Sid,
    classify_record,
    direct_response,
    final_answer_suffix,
    normalize_prompt_mode,
)
from ksllm4rec_sft.data import SourceRecord


class OrpoDataNormalizationTest(unittest.TestCase):
    def test_direct_response_preserves_post_think_bytes(self) -> None:
        original = "<think>reason\nwith sid</think>  \nanswer"
        converted = direct_response(original)
        self.assertEqual(converted, EMPTY_THINK + "  \nanswer")
        self.assertEqual(
            final_answer_suffix(converted), final_answer_suffix(original)
        )

    def test_prompt_mode_is_normalized_only_at_the_suffix(self) -> None:
        self.assertEqual(normalize_prompt_mode("question/think"), "question/no_think")
        self.assertEqual(
            normalize_prompt_mode("question/no_think  "), "question/no_think"
        )
        self.assertEqual(normalize_prompt_mode("question"), "question/no_think")

    def test_sid_round_trip_and_hierarchy_keys(self) -> None:
        value = "<|prod_begin|><s_a_168><s_b_4799><s_c_5345>"
        sid = Sid.parse(value)
        self.assertEqual(sid.render(), value)
        self.assertEqual(sid.ab_key, ("prod", 168, 4799))
        self.assertEqual(sid.a_key, ("prod", 168))

    def test_task_classification_uses_output_contract(self) -> None:
        recommend_system = "recommend-system"
        recommend = SourceRecord(
            recommend_system,
            "history",
            "<think></think>target <|video_begin|><s_a_1><s_b_2><s_c_3>",
        )
        text_to_sid = SourceRecord(
            "item",
            "caption",
            "<think></think><|prod_begin|><s_a_1><s_b_2><s_c_3>",
        )
        sid_to_text = SourceRecord(
            "item",
            "<|prod_begin|><s_a_1><s_b_2><s_c_3>",
            "<think></think>caption",
        )
        self.assertEqual(
            classify_record(recommend, {recommend_system}), PairTask.RECOMMEND
        )
        self.assertEqual(
            classify_record(text_to_sid, {recommend_system}), PairTask.TEXT_TO_SID
        )
        self.assertEqual(
            classify_record(sid_to_text, {recommend_system}), PairTask.SID_TO_TEXT
        )


if __name__ == "__main__":
    unittest.main()
