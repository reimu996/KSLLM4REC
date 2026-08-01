# KSLLM4REC 官方竞赛成绩（StreamLake 官方评测）

> 生成日期：2026-08-02。**官网已删除数据，本文件与 `KSLLM4REC_RESULT/` 是唯一留存。**
> 来源：`/home/lyc/REC_PROJECTS/KSLLM4REC_RESULT/streamlake_data/`（streamlake-experiment-analyst skill 同步，最后同步 07-28）+ 用户确认（官网删除后记忆）。

## 总分演进链（按时间）

| 日期 | 模型 | anchor | 官方分 | 说明 |
|---|---|---|---|---|
| 07-22 | SFT372（frontier_LORA_6464_000015_正则0001-epoch2） | — | **0.9744** | SFT + 正则 0.001；R1=0.1078 R2=0.4973 R3=0.1546 |
| 07-25 | RLOO-DAPO-T12 窗口级（无 anchor） | ❌ | **0.9844 / 0.9871** | 继承 SFT372；DAPO 非对称 clip + KV cache |
| 07-26 | RLOO-DAPO DAPOW200 | ❌ | 0.9651 | 窗口 200 |
| 07-27 | DAPO-Anchor V2.2 单任务（4 个 checkpoint 评测：1.0079 / 0.9732 / 0.9713 / 0.9600） | ✅ | **1.0079**（最优） | 最高 1.0079（R1=0.1257 R2=0.4804 R3=0.1565） |
| **07-28+** | **DAPO-Anchor multitask（W450，1 epoch 附近）** | ✅ | **1.0191** | **全程最高**（用户确认，官网删除后记忆） |

## Top 10（本地 metrics.csv 留存，73 个评估实验）

| # | 总分 | R1 | R2 | R3 | 实验（日期） |
|---|---|---|---|---|---|
| 1 | 1.0079 | 0.1257 | 0.4804 | 0.1565 | RLOO-DAPO-Anchor（07-27） |
| 2 | 0.9871 | 0.1250 | 0.4607 | 0.1561 | RLOO-DAPO-T12（07-25） |
| 3 | 0.9844 | 0.1226 | 0.4906 | 0.1565 | RLOO-DAPO-T12（07-25） |
| 4 | 0.9744 | 0.1078 | 0.4973 | 0.1546 | frontier_LORA_6464_000015_正则0001-epoch2（07-22）= SFT372 |
| 5 | 0.9732 | 0.1313 | 0.4397 | 0.1569 | RLOO-DAPO-Anchor（07-27） |
| 6 | 0.9720 | 0.1154 | 0.4693 | 0.1420 | 正则0001_val0（07-25） |
| 7 | 0.9718 | 0.1117 | 0.4587 | 0.1561 | frontier GRPO epoch1（07-22） |
| 8 | 0.9713 | 0.1309 | 0.4682 | 0.1576 | RLOO-DAPO-Anchor（07-27） |
| 9 | 0.9709 | 0.1083 | 0.4933 | 0.1546 | 正则0001 Dropout001（07-23） |
| 10 | 0.9651 | 0.1117 | 0.5159 | 0.1535 | DAPOW200（07-26） |

> multitask 1.0191 未在本地 metrics.csv（07-28 后未同步）；R2 指标权重最大，近似命中（same_ab/same_a）策略在 R2 上受益。

## 关键结论

1. **官方总分口径下 RL/DAPO 系列明确上涨**：0.9744 → 0.9871 → 1.0079 → **1.0191**；
2. 此前"RL 未超起点"结论仅适用于**本地 probe（exact 口径）**——官方 pass@64 总分口径与 probe exact 是两把尺子；
3. **multitask W450（1 epoch 附近）是当前最强 checkpoint**，超 SFT372 4.6%；
4. 官方分本地留存位置：`/home/lyc/REC_PROJECTS/KSLLM4REC_RESULT/streamlake_data/`（experiments.sqlite / metrics.csv / raw/）。

## 待确认

- multitask 1.0191 的精确评测日期与 R1/R2/R3 细分（官网已删，用户记忆仅有总分）
