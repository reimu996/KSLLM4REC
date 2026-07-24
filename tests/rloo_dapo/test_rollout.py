from __future__ import annotations

import unittest

from ksllm4rec_rloo_dapo.rollout import (
    PromptRequest,
    PromptRollout,
    ProposalDecision,
    SampledCandidate,
    _canonicalize_aligned_rollout,
    candidate_wave_pairs,
    collect_forced_span,
)


class _Grammar:
    eos_token_id = 99

    def allowed_next(self, prefix):
        transitions = {
            (): [1],
            (1,): [2],
            (1, 2): [10, 11],
            (1, 2, 10): [12],
            (1, 2, 10, 12): [99],
        }
        return transitions.get(tuple(prefix), [])

    def parse(self, tokens):
        if tokens != [1, 2, 10, 12, 99]:
            raise ValueError("invalid completion")
        return tokens


class CachedRolloutShapeTest(unittest.TestCase):
    def test_eight_waves_contain_two_candidates_per_prompt(self) -> None:
        waves = candidate_wave_pairs(8)
        self.assertEqual(len(waves), 8)
        self.assertTrue(all(len(wave) == 16 for wave in waves))
        self.assertEqual(waves[0][:4], ((0, 0), (0, 1), (1, 0), (1, 1)))

    def test_forced_span_stops_before_a_decision(self) -> None:
        span = collect_forced_span(_Grammar(), [], maximum_tokens=8)
        self.assertEqual(span.token_ids, (1, 2))
        self.assertFalse(span.finished)

    def test_forced_span_includes_terminal_eos(self) -> None:
        span = collect_forced_span(
            _Grammar(), [1, 2, 10], maximum_tokens=8
        )
        self.assertEqual(span.token_ids, (12, 99))
        self.assertTrue(span.finished)

    def test_aligned_samples_are_promoted_only_after_full_metadata_validation(self) -> None:
        grammar = _Grammar()
        candidates = tuple(
            SampledCandidate(
                candidate_index=index,
                sid=[1, 2, 10, 12, 99],
                token_ids=(1, 2, 10, 12, 99),
                sample_log_probs=(0.0, 0.0, -0.25, 0.0, 0.0),
                decision_mask=(False, False, True, False, False),
                proposal_decisions=(
                    ProposalDecision(
                        position=2,
                        allowed_ids=(10, 11),
                        log_probs=(-0.25, -1.5),
                    ),
                ),
            )
            for index in range(16)
        )
        rollout = PromptRollout(
            request=PromptRequest("group", (7, 8)),
            candidates=candidates,
            policy_aligned=True,
        )

        canonical = _canonicalize_aligned_rollout(rollout, grammar)

        self.assertEqual(canonical.target_forward_calls, 0)
        self.assertEqual(canonical.corrected_candidate_count, 0)
        self.assertEqual(canonical.candidates[0].old_log_probs[2], -0.25)

    def test_aligned_samples_reject_a_forged_chosen_logp(self) -> None:
        candidate = SampledCandidate(
            candidate_index=0,
            sid=[1, 2, 10, 12, 99],
            token_ids=(1, 2, 10, 12, 99),
            sample_log_probs=(0.0, 0.0, -0.5, 0.0, 0.0),
            decision_mask=(False, False, True, False, False),
            proposal_decisions=(
                ProposalDecision(
                    position=2,
                    allowed_ids=(10, 11),
                    log_probs=(-0.25, -1.5),
                ),
            ),
        )
        rollout = PromptRollout(
            request=PromptRequest("group", (7, 8)),
            candidates=tuple(
                candidate
                if index == 0
                else SampledCandidate(
                    candidate_index=index,
                    sid=candidate.sid,
                    token_ids=candidate.token_ids,
                    sample_log_probs=(0.0, 0.0, -0.25, 0.0, 0.0),
                    decision_mask=candidate.decision_mask,
                    proposal_decisions=candidate.proposal_decisions,
                )
                for index in range(16)
            ),
            policy_aligned=True,
        )

        with self.assertRaisesRegex(RuntimeError, "logp is inconsistent"):
            _canonicalize_aligned_rollout(rollout, _Grammar())


if __name__ == "__main__":
    unittest.main()
