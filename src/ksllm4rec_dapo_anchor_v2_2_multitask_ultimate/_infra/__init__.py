"""Vendored infrastructure for DAPO-Anchor (fully self-contained).

Every module in this sub-package is a *bit-for-bit* copy of the source noted
below, with `from ksllm4rec_<sibling>.<module>` rewritten to `from ._infra.<name>`
(or `from .<name>` inside this package) and nothing else changed.

Vendoring point:
    origin_commit = 213a1002a88c754c7c0989ac87a0d241dbeb36f9

Vendor manifest (26 files):

    sft_profiles.py       ← ksllm4rec_sft/profiles.py
    sft_data.py           ← ksllm4rec_sft/data.py
    orpo_data.py          ← ksllm4rec_orpo/data.py
    grpo_profiles.py      ← ksllm4rec_grpo/profiles.py
    grpo_contract.py      ← ksllm4rec_grpo/contract.py
    grpo_probability.py   ← ksllm4rec_grpo/probability.py
    grpo_trie.py          ← ksllm4rec_grpo/trie.py
    grpo_prompt.py        ← ksllm4rec_grpo/prompt.py
    grpo_constraint.py    ← ksllm4rec_grpo/constraint.py
    grpo_data.py          ← ksllm4rec_grpo/data.py
    rloo_contract.py      ← ksllm4rec_rloo/contract.py
    rloo_integrity.py     ← ksllm4rec_rloo/integrity.py
    rloo_fingerprint.py   ← ksllm4rec_rloo/fingerprint.py
    rloo_probability.py   ← ksllm4rec_rloo/probability.py
    rloo_scoring.py       ← ksllm4rec_rloo/scoring.py
    rloo_rollout.py       ← ksllm4rec_rloo/rollout.py
    rloo_modeling.py      ← ksllm4rec_rloo/modeling.py
    rloo_checkpoint.py    ← ksllm4rec_rloo/checkpoint.py
    rloo_objective.py     ← ksllm4rec_rloo/objective.py
    dapo_contract.py      ← ksllm4rec_rloo_dapo/contract.py
    dapo_kv_cache.py      ← ksllm4rec_rloo_dapo/kv_cache.py
    dapo_speculative.py   ← ksllm4rec_rloo_dapo/speculative.py
    dapo_sampling.py      ← ksllm4rec_rloo_dapo/sampling.py
    dapo_scoring.py       ← ksllm4rec_rloo_dapo/scoring.py
    dapo_rollout.py       ← ksllm4rec_rloo/rloo_dapo/rollout.py
    dapo_fingerprint.py   ← ksllm4rec_rloo_dapo/fingerprint.py
    dapo_checkpoint.py    ← ksllm4rec_rloo_dapo/checkpoint.py

To detect drift after further changes in the sibling packages:
    git diff 213a1002 -- src/ksllm4rec_rloo_dapo/rollout.py
"""
