# DAPO-Anchor V2.2 Multitask SFT372 E2

## why

Keep the approved V2.2 update order and reward unchanged, while training both
recommendation and item-text-to-SID source groups. There is no offline anchor
calibration: each complete window derives its anchor cap from that window's
four RL gradient norms.

## Frozen identity

- Start: SFT372 LoRA r=64, alpha=64.
- Source tasks: 17,016 recommendation groups plus 10,597 item-text-to-SID groups.
- RL: P=32 effective groups, G=16, four 8-group optimizer minibatches, K=1.
- Rollout: temperature 1.2, top-p 1.0, asymmetric ratio clip [0.8, 1.28].
- Anchor candidate: reward max equals min and exact count equals zero.
- Anchor target: `-log(sum_m P(GT_m | prompt))` over every legal GT SID.
- Anchor execution: dense no-cache teacher forcing, at most eight GT
  completions per forward, all candidate groups, zero or one optimizer step.
- Anchor enablement: W0 onward; it is independent of LR warmup.
- Update order: four RL optimizer steps, then at most one independent Anchor
  optimizer step, then one scheduler step.
- Gradient budget: `min(0.05, 0.1 * median(RL norms) / Anchor norm)` per window.
- LR: W0=0.1e-6, linearly increasing to 1e-6 at W9, then constant.
- Training length: 1,726 windows (two legacy effective epochs of 863 windows).

Config:

`configs/rloo/frontier_sft372_g16_dapo_anchor_v2_2_multitask_ultimate_e2.yaml`

## Execution

```bash
# CPU/config/data checks only; no model forward or calibration
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2_multitask/test_cpu.sh

# Formal training starts directly
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2_multitask/run_full.sh

# Read-only low-frequency status
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2_multitask/status.sh
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
