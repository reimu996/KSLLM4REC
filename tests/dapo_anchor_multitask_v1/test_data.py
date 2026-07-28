from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from ksllm4rec_dapo_anchor_multitask_v1.data import (
    FRONTIER_TEXT_TO_SID_DUPLICATE_ROWS,
    FRONTIER_TEXT_TO_SID_GROUPS,
    FRONTIER_TEXT_TO_SID_ROWS,
    FRONTIER_TEXT_TO_SID_UNIQUE_SIDS,
    SidTask,
    build_frontier_text_to_sid_groups,
    canonical_group_id,
    group_text_to_sid_records,
    load_text_to_sid_target_groups,
    recommendation_target_group,
    write_text_to_sid_artifact,
)
from ksllm4rec_dapo_anchor_multitask_v1._infra.frontier_data import (
    FrontierIndexedRecord,
)
from ksllm4rec_dapo_anchor_multitask_v1._infra.grpo_data import RecommendationGroup
from ksllm4rec_dapo_anchor_multitask_v1._infra.sft_data import SourceRecord


SID_A = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
SID_B = "<|video_begin|><s_a_1><s_b_2><s_c_4>"
SOURCE = Path("/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl")
PROVENANCE = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/provenance.jsonl"
)


def row(line: int, mode: str, sid: str, *, variant: str) -> FrontierIndexedRecord:
    record = SourceRecord(
        system="item system",
        prompt=f"same item description/{mode}",
        response=f"<think>variant-specific reasoning</think>\nanswer: {sid}",
    )
    return FrontierIndexedRecord(
        line_number=line,
        source_sha256=f"sha-{line}",
        record=record,
        task="item_text_to_sid",
        variant=variant,
        route_id="test",
        provenance={},
    )


class TextToSidGroupingTest(unittest.TestCase):
    def test_recommendation_artifact_adapts_to_the_shared_group_type(self) -> None:
        source = RecommendationGroup(
            group_id=canonical_group_id("system", "prompt/no_think"),
            system="system",
            prompt="prompt/no_think",
            positive_sids=(SID_A, SID_B),
            source_lines=(1, 2),
        )
        group = recommendation_target_group(source)
        self.assertEqual(group.task, SidTask.RECOMMENDATION)
        self.assertEqual(group.group_id, source.group_id)
        self.assertEqual(group.positive_sids, (SID_A, SID_B))

    def test_direct_and_thinking_variants_collapse_to_one_single_gt_group(self) -> None:
        result = group_text_to_sid_records(
            [
                row(2, "no_think", SID_A, variant="direct"),
                row(1, "think", SID_A, variant="thinking"),
            ],
            enforce_contract=False,
        )
        self.assertEqual(result.raw_rows, 2)
        self.assertEqual(result.duplicate_rows, 1)
        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertEqual(group.task, SidTask.ITEM_TEXT_TO_SID)
        self.assertEqual(group.prompt, "same item description/no_think")
        self.assertEqual(group.positive_sids, (SID_A,))
        self.assertEqual(group.source_lines, (1, 2))
        self.assertEqual(group.to_dict()["task"], "item_text_to_sid")

    def test_conflicting_targets_for_normalized_prompt_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "maps to multiple SIDs"):
            group_text_to_sid_records(
                [
                    row(1, "think", SID_A, variant="thinking"),
                    row(2, "no_think", SID_B, variant="direct"),
                ],
                enforce_contract=False,
            )

    @unittest.skipUnless(SOURCE.is_file() and PROVENANCE.is_file(), "Frontier data absent")
    def test_artifact_roundtrip_is_complete_and_refuses_overwrite(self) -> None:
        build = build_frontier_text_to_sid_groups(SOURCE, PROVENANCE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "text-groups"
            manifest = write_text_to_sid_artifact(
                build, output, source=SOURCE, provenance=PROVENANCE
            )
            loaded = load_text_to_sid_target_groups(output / "groups.jsonl")
            self.assertEqual(loaded, build.groups)
            self.assertEqual(manifest["groups"], 10_597)
            self.assertEqual(manifest["positive_edges"], 10_597)
            self.assertEqual(manifest["multi_gt_groups"], 0)
            with self.assertRaises(FileExistsError):
                write_text_to_sid_artifact(
                    build, output, source=SOURCE, provenance=PROVENANCE
                )

    @unittest.skipUnless(SOURCE.is_file() and PROVENANCE.is_file(), "Frontier data absent")
    def test_full_frontier_text_contract_is_13353_to_10597(self) -> None:
        result = build_frontier_text_to_sid_groups(SOURCE, PROVENANCE)
        self.assertEqual(result.raw_rows, FRONTIER_TEXT_TO_SID_ROWS)
        self.assertEqual(len(result.groups), FRONTIER_TEXT_TO_SID_GROUPS)
        self.assertEqual(result.duplicate_rows, FRONTIER_TEXT_TO_SID_DUPLICATE_ROWS)
        self.assertEqual(result.unique_positive_sids, FRONTIER_TEXT_TO_SID_UNIQUE_SIDS)
        self.assertTrue(all(len(group.positive_sids) == 1 for group in result.groups))
        self.assertEqual(
            dict(result.variant_counts),
            {"direct": 5597, "forward": 5000, "thinking": 2756},
        )


if __name__ == "__main__":
    unittest.main()
