# DAPO-Anchor V2.2 SFT372

## why

V1 repeatedly optimized reward-bearing prompts while discarding reward-flat
prompts, and its anchor branch could take several large optimizer steps per
window. V2.2 keeps the existing 32-group RL window but gives every
reward-flat/no-exact prompt a GT-set teacher-forcing signal. All such prompts
are averaged into at most one independent anchor step, whose pre-clip gradient
norm is capped at 10% of the median RL pre-clip norm for that window.

## Frozen identity

- Start: SFT372 LoRA r=64, alpha=64.
- RL: P=32 effective groups, G=16, four 8-group optimizer minibatches, K=1.
- Rollout: temperature 1.2, top-p 1.0, asymmetric ratio clip [0.8, 1.28].
- Anchor candidate: reward max equals min and exact count equals zero.
- Anchor target: `-log(sum_m P(GT_m | prompt))` over every legal GT SID.
- Anchor execution: dense no-cache teacher forcing, at most eight GT
  completions per forward, all candidate groups, zero or one optimizer step.
- Anchor enablement: W0 onward; it is independent of LR warmup.
- LR: W0=0.1e-6, linearly increasing to 1e-6 at W9, then constant.
- Memory gate: peak reserved CUDA memory <=20 GiB.

Config:

`configs/rloo/frontier_sft372_g16_dapo_anchor_v2_2.yaml`

## Execution

```bash
# CPU/unit gates
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2/test_cpu.sh

# 16 unchanged-policy calibration windows; optimizer_updates must remain zero
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2/run_calibration.sh

# One isolated W0 structural window, not a formal recovery point
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2/run_pilot.sh
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2/verify_pilot.sh

# Formal 1064-window run; validates/reuses calibration first
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2/run_full.sh

# Read-only low-frequency status
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2/status.sh
```

## Counters and recovery

Every complete window records `rl_optimizer_update_step`,
`anchor_optimizer_update_step`, and their exact sum `optimizer_update_step`.
AdamW's actual state step is checked against that sum before every atomic
complete-window recovery checkpoint. A crash before checkpoint commit is
reconciled by truncating JSONL logs back to the latest complete cursor.

The 10% rule constrains the anchor gradient handed to clipping and AdamW. It
does not claim that the later AdamW parameter displacement or function change
is at most 10% of RL.
