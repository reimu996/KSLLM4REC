# DAPO-Anchor-Multitask V1.0 可复刻入口

## why

这组脚本要保证新版本只消费一次完整源数据，并让 recommendation 与
`item_text_to_sid` 在同一冻结策略下采样、在同一策略优化窗口中训练，同时不覆盖
V2.2 的代码、配置、运行产物或日志。脚本只写入 Multitask V1 的新目录。

## 固定训练合同

- 起点：SFT372，LoRA `r=64`、`alpha=64`。
- 任务：recommendation 17,016 个源 group；text-to-SID 10,597 个源 group。
- 数据：532 个 source block；每个源 group 恰好采样 16 条候选；完整 C 组共
  27,613 个源 group、441,808 条候选；不补采、不重复、不丢末批 overflow。
- 采样：两个任务都使用 `T=1.2`、`top-p=1.0`、`top-k` 关闭。
- 奖励：exact/same-AB/same-A/same-domain/other-domain 分别为
  `1.0/0.15/0.05/0.01/0.0`。
- RL：只在 grammar decision token 上计算，`K=1`，非对称 clip
  `[0.8,1.28]`，每个 optimizer minibatch 最多 8 个 group。
- Anchor：无 exact 的 varied group 同时进入 RL 与 Anchor；flat group 只进入
  Anchor。Anchor 梯度合并到当前窗口最后一个策略 optimizer step，不产生独立
  optimizer step；缩放后范数不超过本窗口 RL 梯度范数中位数的 10%。
- 学习率：按策略 optimizer step 计数；第 1-4 步 `0.1e-6`，第 5-8 步
  `0.2e-6`，到第 37-40 步达到 `1e-6`，之后保持 `1e-6`。
- 显存门：单窗口 pilot 的 peak reserved memory 不超过 20 GiB。
- 正式训练不写健康指标、不提供状态脚本，也不按 reward、梯度、显存或进度触发
  停训；只保留恢复与离线验收所需的原子记录。

配置文件：

`configs/rloo/frontier_sft372_g16_dapo_anchor_multitask_v1.yaml`

A/B/C 三个实验臂使用互不共享的 config、run、log 和 calibration 路径。默认执行
C；A 或 B 在同一命令前加 `KSLLM4REC_ARM=A` 或 `KSLLM4REC_ARM=B`。

## 执行关系

以下命令串行执行。校准和 pilot 已存在时会拒绝覆盖；正式训练目录为空时新建，
非空时仅允许从同配置、同运行指纹且含原子 recovery 的目录恢复。

```bash
# 1. 构建数据；若固定 artifact 已存在，只校验 SHA256，不改写
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/run_prepare_data.sh

# 2. CPU 单测与配置门禁
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/test_cpu.sh

# 3. 前 16 个 source block 只读校准；optimizer step 必须为 0
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/run_calibration.sh

# 4. 一个完整策略优化窗口 GPU pilot
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/run_pilot.sh

# 5. 正式完整训练入口；本仓库只提供入口，不会自动启动
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/run_full.sh

# 6. 完整运行结构验收
/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python \
  scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/verify_run.py \
  --config configs/rloo/frontier_sft372_g16_dapo_anchor_multitask_v1.yaml \
  --run-dir artifacts/rloo/runs/dapo_anchor_multitask_sft372_v1 \
  --require-complete

# 7. 训练结束后人工触发真实1024行Probe64；正式训练不会调用它
KSLLM4REC_PROBE_ADAPTER=/absolute/path/to/adapter \
  scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/run_probe64.sh

# 8. 重新读取固定probe、固定Trie和adapter字节，严格验收完整报告
scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1/verify_probe64.sh
```

## 输出边界

- 数据：`artifacts/grpo/data/text_to_sid_groups_frontier_v1/`
- 校准：`operation_logs/rloo/dapo_anchor_multitask_sft372_v1/`
- pilot：`artifacts/rloo/pilots/dapo_anchor_multitask_sft372_v1_one_window/`
- 正式运行：`artifacts/rloo/runs/dapo_anchor_multitask_sft372_v1/`
- 离线Probe64：`artifacts/rloo/evaluations/dapo_anchor_multitask_sft372_v1/`

`run_full.sh` 只在用户明确执行时开始完整训练；本次实现和语法验证不会运行它。
脚本不持久化 stdout，也不创建 `health.jsonl`；正式运行状态只由原子 recovery、
三个审计 JSONL 和最终 `run_summary.json` 表示。
