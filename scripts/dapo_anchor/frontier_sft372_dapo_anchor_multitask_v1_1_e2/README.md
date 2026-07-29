# DAPO-Anchor-Multitask V1.1 两轮精确训练入口

## why

目标是让每个**归一化 GT-SID 集合 group**在每个 epoch 恰好产生一次在线候选、
恰好获得一次 Anchor 监督；两轮结束后，每个 group 总共被覆盖两次。这里的
group 是 `(system, prompt, GT-SID set)`，不是原始 JSONL 行，因此同一语义的
direct/thinking 重复行不会被当作额外训练样本。

该目录只写入 V1.1 的独立配置、pilot、运行和评测目录；不会改写旧 V1、
V2.2 或其运行产物。

## 固定训练合同

- 起点：SFT372，LoRA `r=64`、`alpha=64`、`dropout=0.0`。
- 每个 epoch：recommendation 17,016 个 group 加 text-to-SID 10,597 个 group，
  共 27,613 个 group、41,062 条 GT-SID 边。
- 两个 epoch：共 55,226 个 group、82,124 条 GT-SID 边。每个 group 每轮恰好
  一次，跨轮的顺序独立但可复现：
  `sha256("dapo-anchor-v1.1|seed|epoch|task|group_id")`。
- 分块：每轮 532 个 source block，共 1,064 个全局 block。块只负责切分和恢复，
  不改变上述 group 覆盖定义。
- 候选：每个 group 固定采样 16 条候选；每轮 441,808 条、全程 883,616 条。
  不因奖励相同而丢弃 group，不补采，不以 overflow 截断 group。
- 采样：两个任务都使用 `T=1.2`、`top-p=1.0`、`top-k=0`、KV cache；每次最多
  并行处理 8 个 prompt，每 prompt 两个候选 wave。
- 奖励：D 臂的 exact / same-AB / same-A / same-domain / other-domain 分别为
  `1.0 / 0.40 / 0.15 / 0.01 / 0.0`。C 臂保留历史新档位
  `1.0 / 0.15 / 0.05 / 0.01 / 0.0`，仅用于对照。
- RL：仅在 grammar decision token 上计算 RLOO/DAPO loss，`K=1`，非对称 ratio
  clip 为 `[0.8, 1.28]`；一个 policy minibatch 最多 8 个 RL group。
- Anchor：每个归一化 group 都进入 GT-SID 集合负对数似然分支。候选奖励有差异的
  group 同时进入 RL 与 Anchor；奖励全相同的 group 只进入 Anchor。Anchor 仅在
  decision token 上计算，所有 GT-SID 都保留，不做挑选、截断或强制注入。
- 更新：Anchor 梯度合并到同一窗口的最后一个 RL policy step，不产生常规独立
  optimizer step；缩放后其 L2 范数不超过该窗口 RL 原始梯度范数中位数的 10%。
  若 epoch 末尾仅剩 Anchor，则最多执行一次不推进 scheduler 的 epoch-end flush。
- 学习率：按 policy optimizer step 计数。第 1--4 步 `0.1e-6`，第 5--8 步
  `0.2e-6`，每 4 个 policy step 增加 `0.1e-6`，第 37--40 步为 `1e-6`，之后
  恒定 `1e-6`。Anchor-only flush 不消耗 warmup step。
- 显存门：单窗口 GPU pilot 的 peak reserved memory 不超过 20 GiB。

配置文件：

`configs/rloo/frontier_sft372_g16_dapo_anchor_multitask_v1_1_e2_arm_d.yaml`

默认执行 D 臂，即旧 reward 的完整多任务训练。A/B/C/D 使用互不共享的 config、run
和 log 路径；通过 `KSLLM4REC_ARM=A|B|C|D` 可显式选择实验臂。

## 执行关系

以下命令按顺序执行。校准和 pilot 已存在时拒绝覆盖；正式训练目录非空时，仅允许
从同配置、同运行指纹且带原子 recovery 的目录恢复。

```bash
# 1. 构建或校验固定 text-to-SID group artifact；已有 artifact 只校验 SHA256
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/run_prepare_data.sh

# 2. CPU 单测与 V1.1 配置门禁
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/test_cpu.sh

# 3. 一个完整 policy 优化窗口的 GPU pilot
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/run_pilot.sh

# 4. GPU 恢复审计：连续 4 block 与 2 block 后恢复的轨迹必须逐字节一致
/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python \
  scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/run_recovery_audit.py \
  --config configs/rloo/frontier_sft372_g16_dapo_anchor_multitask_v1_1_e2_arm_d.yaml \
  --device cuda:0

# 5. 正式完整训练入口；仅显式执行此命令才会开始训练
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/run_full.sh

# 6. 两轮完成后，验证精确覆盖、Anchor 覆盖、恢复光标与训练汇总
/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python \
  scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/verify_run.py \
  --config configs/rloo/frontier_sft372_g16_dapo_anchor_multitask_v1_1_e2_arm_d.yaml \
  --run-dir artifacts/rloo/runs/dapo_anchor_multitask_sft372_v1_1_e2_arm_d \
  --require-complete

# 7. 训练结束后人工触发真实 1,024 行 Probe64；正式训练不会自动调用它
KSLLM4REC_PROBE_ADAPTER=/absolute/path/to/adapter \
  scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/run_probe64.sh

# 8. 严格验收 Probe64 的固定 probe、Trie 与 adapter 字节
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/verify_probe64.sh
```

## 正式训练期间的检查频率

为什么：训练状态由原子 recovery 和审计 JSONL 自身保证可恢复；频繁读取日志既不
改变训练结果，也会增加人工和 token 开销。

正式 `run_full.sh` 不启动轮询、健康监控或自动停训逻辑。稳定运行时只在以下时点做
人工检查：启动后确认进程已进入训练、第一轮结束（全局 block 532）和两轮完成；发生
异常退出时才立即检查 recovery。不要按 window、reward、梯度、显存或进度高频轮询。

## 输出边界

- 数据：`artifacts/grpo/data/text_to_sid_groups_frontier_v1/`
- pilot：`artifacts/rloo/pilots/dapo_anchor_multitask_sft372_v1_1_e2_arm_d_one_window/`
- 恢复审计：`artifacts/rloo/audits/dapo_anchor_multitask_v1_1_e2/<audit-id>/`；每次
  审计按唯一 `audit-id` 建目录，报告内绑定运行签名与 arm。
- 正式运行：`artifacts/rloo/runs/dapo_anchor_multitask_sft372_v1_1_e2_arm_d/`
- 离线 Probe64：`artifacts/rloo/evaluations/dapo_anchor_multitask_sft372_v1_1_e2_arm_d/`

正式运行不持久化 stdout，也不创建 `health.jsonl`。可恢复状态来自原子
`recovery/`，覆盖审计来自 source、RL group、Anchor group、window 和 epoch-end
flush JSONL，最终结果来自 `run_summary.json`。
