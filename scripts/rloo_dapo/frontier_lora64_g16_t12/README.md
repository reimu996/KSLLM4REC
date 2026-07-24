# RLOO-DAPO T1.2 K=1 KV-cache run

## Why

The run must spend less time regenerating the same prompt prefix while keeping
the cached sampler and differentiable training replay on one probability
definition. The implementation therefore shares the exact same per-layer
prompt/completion operations between collection and training.

## Sampling relationship

Eight raw prompts remain one collection batch. For each prompt:

1. The prompt is evaluated once and its 28 layers of BF16 K/V are retained.
2. Sixteen candidates remain logically active. GPU probability evaluation uses
   two fixed eight-row chunks because B16 changes BF16 results.
3. At each `domain/a/b/c` decision, the completion-only forward reads the saved
   prompt K/V and its own causal 19-token completion row.
4. Forced grammar tokens are appended without a model call.
5. The differentiable replay scorer builds the same per-layer prompt K/V inside
   its call and then runs the same fixed B8 completion forward with gradients;
   prompt and completion layers use non-reentrant gradient checkpointing.
6. The sampled log probabilities therefore become `old_logp` directly; no
   second policy, residual correction, reference model, or replay buffer exists.

The 64-group probability gate independently replays all 1024 candidates and
requires cached sample, canonical `old_logp`, and training replay to agree.

## Window relationship

Each logical window collects exactly 32 effective prompt groups. Reward-equal
groups are excluded, and extra effective groups from the final raw batch are
audited then dropped. The 32 groups are deterministically shuffled into four
disjoint 8-group minibatches. Each group is used once, so K=1 and there are four
optimizer updates per window.

## Commands

```bash
scripts/rloo_dapo/frontier_lora64_g16_t12/test_cpu.sh
scripts/rloo_dapo/frontier_lora64_g16_t12/run_gates.sh
scripts/rloo_dapo/frontier_lora64_g16_t12/run_full.sh
scripts/rloo_dapo/frontier_lora64_g16_t12/verify.sh
```

`run_gates.sh` refuses to overwrite an existing pilot directory. The pilot also
runs one complete dense baseline window and requires at least 1.2x end-to-end
window speedup. Formal training accepts recovery only from an atomic complete-
window checkpoint whose runtime signature matches the current config, inputs,
software versions, and code bytes.

The old `ksllm4rec_rloo`, old configs, old scripts, and old artifacts are not
modified by this workflow.
