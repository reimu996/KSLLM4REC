# DAPO-Anchor SFT372 T1.2 K=1 KV-cache 训练入口

## why

这组脚本把 dapo_anchor 训练身份固定下来,保证同一份配置、同一块 GPU。它不
包含 GPU 门禁(与 rloo_dapo 不同)——dapo_anchor 不做 gate/cpu/verify 报告,
仅跑训练(Phase 1 RL + Phase 2 anchor loss)。

## 冻结起点

- **Base model**: `/home/lyc/models/OneReason-0.8B-pretrain-competition`
- **SFT adapter**(checkpoint-372, r=64, alpha=64, dropout=0.05→闭合):
  `/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/sft/platform_exports/frontier_LORA_6464_000015_正则0001/checkpoint-372/train-task-tw1g09-1784715567-epoch2`
- **adapter SHA256**(来自 grpo profiles.py 记录): `699826c7a276b463f7fe8195a0ee6297083d109a6afe23f161f0985b65684ecb`
- **adapter config SHA256**: `55969781f9f855ca35ad5133cb5d0db50f6f8466ce7b078603f26258523e5363`
- **数据**: frontier_feedbackcore_listwise_invariant_v1(63700 rows, 30902 recommend, 905469 unique SIDs)
- **配置**: `configs/rloo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_g16_dapo_anchor005_t12_k1_kvcache.yaml`

## 固定运行身份

`CONFIG`、`LOG_ROOT`、`RUN_DIR`、`TRAIN_LOG`、`DEVICE` 和
`CUDA_VISIBLE_DEVICES=0` 都由 `common.sh` 固定;同名环境变量不会覆盖它们。
只有 `PYTHON` 允许调用者覆盖,默认值为
`/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python`。

## 命令

```bash
# 完整训练 (1064 windows, ~4256 optimizer.steps)
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v1/run_full.sh

# 状态检查 (低频,推荐每 15-30 min)
scripts/dapo_anchor/frontier_sft372_dapo_anchor_v1/status.sh

# Dry-run: 1 window smoke test (不锁,不 tee)
python3 scripts/dapo_anchor/frontier_sft372_dapo_anchor_v1/train.py \
  --config configs/rloo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_g16_dapo_anchor005_t12_k1_kvcache.yaml \
  --output-dir /tmp/dapo_anchor_dry \
  --device cuda:0 --no-resume --stop-after-windows 1
```

## 训练产物

`${RUN_DIR}`(默认 `artifacts/rloo/runs/dapo_anchor_sft372_2effective_epochs/`):

- `windows.jsonl` — 每 window 一行日志(含 anchor_groups / anchor_loss / anchor_grad_norm)
- `groups.jsonl` — 每组详情(rewards, advantages, reward tiers)
- `resolved_config.json`, `runtime_signature.json` — 启动快照
- `recovery/latest.json` — 每 4 optimizer.steps 自动保存
- `epoch-01-adapter/`, `epoch-02-adapter/` — 阶段 adapter
- `run_summary.json` — 训练完成写

`${LOG_ROOT}`(默认 `operation_logs/rloo/dapo_anchor_sft372/`):

- `full_train.stdout.log` — 全 stdout
- `full_train.lock` — flock 互斥文件

## 恢复机制

`run_full.sh` 默认启用了 resume。若训练中断(Ctrl-C / OOM / 宕机),再次执行
`run_full.sh` 会自动从最新 recovery 点恢复。resume 通过 `train.py --no-resume`
跳过。

## 设计差异(与 rloo_dapo)

| 方面 | rloo_dapo SFT372 | dapo_anchor |
|---|---|---|
| 门禁 | 5 份 GPU gate + CPU gate | **无**(跳过) |
| 优势公式 | (r - mean) / std (RLOO) | r - mean (仅居中,不除 std) |
| 附加 loss | 无 | anchor loss λ=0.05 on filter groups |
| verification | verify.sh 跑 post-training check | 无 |
| batch 脚本 | run_all.sh 串联 4 步 | run_full.sh 直跑训练 |

## 互斥

`run_full.sh` 利用 `flock` 独占 `LOG_ROOT/full_train.lock`;第二个进程会即
时拒绝并退出(exit 1),不会覆盖日志。
