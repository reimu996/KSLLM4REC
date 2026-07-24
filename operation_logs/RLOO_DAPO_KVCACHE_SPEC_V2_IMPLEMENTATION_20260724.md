# RLOO-DAPO T1.2 K=1 KV-cache Spec v2 implementation record

Date: 2026-07-24

## Why

The run must collect 32 reward-varying prompt groups per logical window and use
each rollout once, while reducing repeated prompt-prefix computation. The final
candidate distribution must remain the same distribution scored by the
no-cache training path.

## Frozen run contract

| Item | Value |
|---|---:|
| Effective prompt groups/window | 32 |
| Candidates/prompt | 16 |
| Final rollouts/window | 512 |
| Optimizer minibatch | 8 groups / 128 rollouts |
| Optimizer updates/window | 4 |
| K | 1 |
| Ratio clip | [0.8, 1.28] |
| Training temperature | 1.2 |
| Learning rate | 10-window linear warmup to 1e-6, then constant |
| Total windows | 1064 |
| Total optimizer updates | 4256 |
| GT anchor/reference/KL | absent |

Config:
`configs/rloo/frontier_sft_epoch2_lora64_g16_rloo_dapo_t12_k1_kvcache.yaml`

Implementation:
`src/ksllm4rec_rloo_dapo/`

Scripts:
`scripts/rloo_dapo/frontier_lora64_g16_t12/`

## KV and probability mechanism

The target is one policy distribution shared by cached sampling and gradient
replay. The implementation achieves this without a second proposal policy:

1. Eight raw prompts form one collection batch.
2. Each prompt is evaluated once. Its 28 layers of BF16 prompt K/V are retained.
3. Sixteen candidates are logically active. Probability computation is split
   into two fixed B8 chunks; a B16 probe changed hidden states by as much as 3.0.
4. Each `domain/a/b/c` decision runs only the fixed 19-token completion rows,
   which attend to the retained prompt K/V and their own causal suffix.
5. Forced grammar tokens require no model forward.
6. Training replay builds the same prompt K/V inside a differentiable call and
   invokes the same B8 completion operation. Prompt and completion layers use
   non-reentrant gradient checkpointing; no K/V survives a training call.
7. Cached chosen-token log probabilities become canonical `old_logp` directly.

`policy_aligned` samples are promoted only after validating every grammar token,
decision mask, legal-ID set, chosen-token log probability, SID, and candidate
index. The separately loaded 64-group probability gate then independently
replays every final candidate through the differentiable scorer.

## Verified evidence

### CPU and legacy isolation

- New RLOO-DAPO tests: 41 passed.
- Existing `tests/rloo`: 85 passed.
- Existing `tests/sft`: 59 passed.
- Existing `tests/grpo`: 62 passed.
- Existing `tests/orpo`: 19 passed.
- Total across the five suites: 266 passed.
- `git diff` for old `src/ksllm4rec_rloo`, old RLOO config/script/test paths:
  empty.

### Structure

- Recommendation groups: 17016.
- Positive edges: 30465.
- Trie leaves: 905469.
- Trie a nodes: 10744.
- Trie ab nodes: 429540.
- All locked input SHA256 values matched.

### Probability gate

Measured on 64 groups / 1024 final candidates:

- all legal: true;
- all finite: true;
- corrected candidates: 0;
- maximum cached-sample/canonical difference: 0.0;
- maximum final-sample/canonical difference: 0.0;
- maximum independent training replay difference: 0.0.
- runtime signature SHA256:
  `d019cfc17c2d68f8163ae3c64b312360fed5e88d6266f591f725bf980107f1b2`.

### Memory gate

Synthetic worst case: eight copies of the real 2929-token longest prompt, 16
logical candidates per prompt, 128 final samples, then one deterministic
8-candidate backward chunk.

- aligned prompt KV bytes: 335921152;
- peak reserved: 3.064453125 GiB;
- limit: 20 GiB;
- gradients finite: true.

### Throughput

Measured on the same 64 raw prompts / 1024 final candidates:

| Path | Seconds | Speedup vs dense no-cache |
|---|---:|---:|
| Dense no-cache rollout | 75.7422 | 1.000 |
| Aligned KV sampling | 27.9642 | 2.7085 |
| Canonical validation | 0.5762 | - |
| Final canonical rollout | 28.5404 | 2.6539 |

The required final-rollout speedup is 1.5x.

The pilot additionally reruns one complete gate-only dense implementation
window using the current code with aligned cache and shared scoring disabled:

- dense window: 97.3057674 seconds;
- aligned window: 75.1390077 seconds;
- end-to-end speedup: 1.2950x;
- required end-to-end speedup: 1.2x.

### Two-window pilot

Continuous deterministic run:

- 2 windows;
- 64 effective group occurrences;
- 8 optimizer updates;
- peak reserved: 3.34765625 GiB;
- start adapter SHA256:
  `c31f1e67cd46b02a27a1ed44136eccb1f83b7055f247c84d4a6924cafa4c1d70`;
- final adapter SHA256:
  `b6b5607dd1c46b30a9240ad29101dd9519b8a421854117fe37f8033ed004a92a`.

Window 1 used LR 1e-7. Its first minibatch had replay delta 0 and ratio 1.
After optimizer updates, later minibatches produced finite ratios outside the
clip interval and activated clipping; no `new approximately old` gate blocked
them.

### Recovery determinism

Compared:

- one process running two windows continuously;
- one window, process exit, checkpoint restore, then one window.

Results were identical for all deterministic window fields, all 64 group rows,
candidate SIDs/token IDs/rewards, every loss and ratio, source cursor, final
adapter bytes, and serialized optimizer/scheduler/RNG state.

- final adapter SHA256 on both paths:
  `b6b5607dd1c46b30a9240ad29101dd9519b8a421854117fe37f8033ed004a92a`;
- `training_state.pt` SHA256 on both paths:
  `c8fcdde1e0fb6134f481a580dea132a23d3e9e14e500d8366fd30dfcc31ef6c0`;
- `groups.jsonl` SHA256 on both paths:
  `cccc28e698c8bf491078751a9635267844ab882409faec148fd79533150e211b`.

Every deterministic window field, candidate token/SID/reward row, loss, ratio,
clip count, source cursor, adapter byte, and optimizer/scheduler/RNG byte was
identical. Wall-clock and allocator-peak fields were excluded from the JSON
field comparison.

## Rejected alternatives

The following probes were rejected before the aligned design was selected:

| Probe | Result | Rejection reason |
|---|---:|---|
| One 16-row canonical forward versus two 8-row forwards | `0.8934x`; max decision-logp delta `0.38974` | slower and violates 8-row replay parity |
| Fixed 8-row same-prompt KV replay versus fixed 8-row no-cache replay | max delta `0.23880`; 25/29 decisions above `0.005` | cache/no-cache operator paths remain different |
| Disable input-gradient hook and gradient checkpointing during scoring | `1.0113x` | negligible |
| Per-batch legal-ID/lm-head projection cache | total `1.0255x`; output exact; +`0.0743 GiB` | insufficient |
| One-prompt prefill with all 16 candidates active | total `1.0551x` on the same B8 | insufficient |
| Split KV branches by forced-span length | 63 decode calls, 27 forks, peak `21.2129 GiB` | exceeds 20 GiB and adds decode/fork work |
| Full-backbone `torch.compile` | hidden delta `8.125`; DynamicCache graph overwrite | violates replay/cache lifetime |
| Direct token-at-a-time `flash_attn_with_kvcache` | `0.724x` versus current single-prompt path | 17 decode calls lose forced-span batching |

No full 1064-window training was run as part of implementation validation.
