# Frontier SFT Epoch 2 -> Online GRPO 执行摘要

## 目标与结论

本实验从官方平台导出的 Frontier SFT Epoch 2 LoRA 继续训练两轮在线
GRPO。正式训练、两轮固定 probe 和全量 verifier 均已自然结束并通过；未发生
CUDA OOM，也未启用 chunk 降级。这里记录的是本地诊断结果，不代表官方比赛
分数，也不宣称一定提分。

## 冻结输入

- Base：`/home/lyc/models/OneReason-0.8B-pretrain-competition`
  - `model.safetensors` SHA256：
    `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90`
- 起点 Frontier SFT Epoch 2 adapter：
  `artifacts/sft/platform_exports/frontier_feedbackcore_listwise_invariant_epoch2_20260719/extracted`
  - 权重 SHA256：
    `b7a50ca748f83059ddf27c5a8053fda3095a541619ee67a27460f9535b865f63`
- 数据：`/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl`
  - 63,700 行；SHA256：
    `9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a`
- 来源记录：同目录 `provenance.jsonl`
  - SHA256：
    `e56dddff865780273573e06e5a44d4c1dd9a27013c78c166aa6565f7b9ad8908`
- 最终运行合同签名：
  `68d54ad8f9351f30b02b9e74858f43eda34dc17b347879fe86e7aa60da5ed15b`

## 数据合同

- 原始 recommendation 行：30,902。
- 统一将 prompt 末尾 `/think` 或 `/no_think` 归一化为 `/no_think`。
- reward 只使用 `</think>` 后的唯一最终 SID；原始 thinking 文本不进入
  rollout、reward 或 loss。
- 归一化训练组：17,016；唯一 `(group, GT SID)`：30,465；跨组不同 GT
  SID：29,414。
- 437 个重复正确边均为 direct/thinking 双份；合并后不重复计权，原始行号
  仍保留在 `source_lines`。
- 合法 SID 前缀树扫描全部 63,700 行的 system、prompt、response；叶子数
  905,469。
- `groups.jsonl` SHA256：
  `a2571d97e4ac231d61d1f468bc23b8e65bebfd5ad7b1a2dd566f1bfd86a9658e`
- trie manifest SHA256：
  `f51414b13a8982605ebe1f7fec75929245db60c854e3860d3b6c1625a3f33e4a`

## 训练合同与结果

- 在线 G=8，重复候选保留；合法前缀树约束；无 beam、无 replay buffer。
- 无在线 exact GT 时，以确定性 0.5 概率将候选索引 7 替换为一个 GT。
- reward：exact 1.0、同 domain+a+b 0.20、同 domain+a 0.05、同 domain
  0.01、否则 0；组内 sample Z-score，correction=1，epsilon=1e-4。
- PPO clip=0.2；sampled-token reference KL beta=0.02；无 CE、Focal Loss
  或 item-token weight。
- LoRA r=32、alpha=32、运行及保存 dropout=0；仅 policy LoRA 可训练。
- 2 epoch；每轮 17,016 组；梯度累计 8 组；每轮 2,127 optimizer steps；
  总计 4,254 steps；LR 5e-6、cosine、128 warmup steps。
- 正式训练用时 151,180.738 秒；平均 0.2251 groups/s。
- 最终状态：`epoch_index=3`、`global_step=4254`、
  `groups_completed=34032`、`next_group_offset=0`、chunk=`8/8`。
- 峰值显存：allocated 5.7291 GiB，reserved 5.9766 GiB；OOM retry 为空。
- signal groups：22,219；forced-GT groups：17,148；live exact groups：154。
- Epoch 1 权重 SHA256：
  `51787cf83fea5bfa11b1015acaf42236cbc3eec5bda22511e59f6e1ffb6f842c`
- Epoch 2 权重 SHA256：
  `6541a936c982af332c969da83e867fcad556e9125ac2db6947dd18082acc1724`
- 两轮 `adapter_config.json` SHA256：
  `28ab07197f42ec5b551622437a008721e4b7296bd12f79dd347e028c4d93de3e`

## 固定 probe

固定输入 SHA256：
`86e050a00506efbaef0e49997ed138462aaddc322eba9fc66227a25654a75e5e`。
每轮均使用 constrained beam=16；512 条 recommendation 与 512 条
text-to-SID；所有 target 可达，所有预测 SID 合法。

| Adapter | recommendation exact | text-to-SID exact | 合法预测 |
|---|---:|---:|---:|
| Epoch 1 | 1/512 | 70/512 | 1,024/1,024 |
| Epoch 2 | 1/512 | 75/512 | 1,024/1,024 |

probe 是本地代理诊断：recommendation probe 使用旧 JSON wrapper 并只比较解析
后的 SID，不能替代官方评测。Frontier 原始推荐响应同时存在裸 SID 和带文字前缀
的表面格式；官方解析器是否只提取 SID 缺少本地证据。

## 验证证据

- `train_audit.jsonl`：34,032 行；Epoch 1/2 各 17,016 行。
- `train_progress.jsonl`：4,254 行；每 step 精确对应 8 个组。
- 所有 progress/audit 关键数值有限；chunk 全程 `8/8`。
- 最终 recovery 为 `checkpoint-step-004254`；manifest 内所有文件的大小和
  SHA256 已独立重算通过。
- `operation_logs/grpo/frontier_sft_epoch2_v2/final_verification.json`
  的 `passed=true`；文件 SHA256：
  `864c1ff5370e5c4f73349233828303716e81b7500179333b5497bf1f8b7042c6`。

## 复现入口

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
bash scripts/grpo/frontier_sft_epoch2/prepare.sh
bash scripts/grpo/frontier_sft_epoch2/test_cpu.sh
bash scripts/grpo/frontier_sft_epoch2/run_gates.sh
bash scripts/grpo/frontier_sft_epoch2/run_full.sh
bash scripts/grpo/frontier_sft_epoch2/run_probes.sh
bash scripts/grpo/frontier_sft_epoch2/verify.sh
```

上传合同由 `configs/grpo/hf_upload_frontier_sft_epoch2.json` 固定；远端上传结果
单独记录，避免把“本地训练通过”和“远端仓库可用”混为一个状态。
