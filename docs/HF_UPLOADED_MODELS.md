# HuggingFace 已上传模型清单（reimu996 账号）

> 生成日期：2026-08-02。来源：本地上传记录（artifacts/*/hf_upload*）+ operation_logs 上传摘要 + 用户确认。
> 说明：HF API 网络不可达，repo ID 基于本地 README/摘要提取；个别未确认项已标注。

## 演进主线

```
基座 ──ORPO──→ GRPO ──→ Frontier GRPO ──→ RLOO ──→ RLOO-DAPO ──→ DAPO-Anchor 单任务 ──→ multitask
07-16      07-18       07-21            07-23     07-25          07-27             07-28+
```

每代改进对应上一代的一个实测失效点；anchor 系列的引入对应"有效组率跌到 20% 以下"（RLOO 实测 21.2%、后期 0-12.5%）的探索不足问题。

## 完整清单（按时间）

| # | 日期 | 模型 repo ID（reimu996/ 下） | 批次/说明 | 本地证据 |
|---|---|---|---|---|
| 1 | 2026-07-16 | `OneReason-0.8B-ORPO-Epoch1` | ORPO E1 | `artifacts/orpo/hf_upload/…20260716…` |
| 2 | 2026-07-16 | `OneReason-0.8B-ORPO-Epoch2` | ORPO E2 | 同上 |
| 3 | 2026-07-18 | `OneReason-0.8B-GRPO-Epoch1` | GRPO E1 | `artifacts/grpo/hf_upload/…20260718…` |
| 4 | 2026-07-18 | `OneReason-0.8B-GRPO-Epoch2` | GRPO E2 | 同上 |
| 5 | 2026-07-19 | ⚠️ repo ID 未记录 | Frontier SFT epoch1（本地 `artifacts/sft/hf_upload/frontier_epoch1`，README 无 title） | `artifacts/sft/hf_upload/` |
| 6 | 2026-07-21 | `OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch1` | Frontier GRPO E1 | `artifacts/grpo/hf_upload_frontier_sft_epoch2/…20260721…` |
| 7 | 2026-07-21 | `OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch2` | Frontier GRPO E2 | 同上 |
| 8 | 2026-07-23 | `OneReason-0.8B-Frontier-SFT-Epoch2-RLOO-Epoch1`（revision `805a410…`） | Frontier RLOO E1 | 摘要 `operation_logs/HF_FRONTIER_RLOO_UPLOAD_SUMMARY_20260723.md` |
| 9 | 2026-07-23 | `OneReason-0.8B-Frontier-SFT-Epoch2-RLOO-Epoch2`（revision `d3fce548…`） | Frontier RLOO E2 | 同上 |
| 10 | 2026-07-25 | `OneReason-0.8B-Frontier-SFT372-RLOO-DAPO-Window-150` | RLOO-DAPO 窗口级 | `artifacts/rloo/hf_upload_sft372_dapo/…20260725…` |
| 11 | 2026-07-25 | `OneReason-0.8B-Frontier-SFT372-RLOO-DAPO-Window-175` | 同上 | 同上 |
| 12 | 2026-07-25 | `OneReason-0.8B-Frontier-SFT372-RLOO-DAPO-Window-200` | 同上 | 同上 |
| 13 | 2026-07-25 | `OneReason-0.8B-Frontier-SFT372-RLOO-DAPO-Window-400` | 同上 | 同上 |
| 14 | 2026-07-25 | `OneReason-0.8B-Frontier-SFT372-RLOO-DAPO-Window-532` | 同上（≈单任务 1 epoch） | 同上 |
| 15 | 2026-07-25 | `OneReason-0.8B-Frontier-SFT372-RLOO-DAPO-Window-625` | 同上 | 同上 |
| 16 | 2026-07-27 | `OneReason-0.8B-Frontier-SFT372-DAPO-Anchor-V2.2-First-Source-Pass` | DAPO-Anchor 单任务完整 1 遍历 | `artifacts/rloo/hf_upload/first_source_pass_20260727_035732` |
| 17 | 2026-07-27 | `OneReason-0.8B-Frontier-SFT372-DAPO-Anchor-V2.2-Second-Source-Pass` | DAPO-Anchor 单任务完整 2 遍历 | `artifacts/rloo/hf_upload/second_source_pass_20260727_100554` |
| 18 | 2026-07-27 | `OneReason-0.8B-Frontier-SFT372-DAPO-Anchor-V2.2-W150` | DAPO-Anchor 窗口候选 | `artifacts/rloo/hf_upload/dapo_anchor_candidates_W150_W175_20260727_172653` |
| 19 | 2026-07-27 | `OneReason-0.8B-Frontier-SFT372-DAPO-Anchor-V2.2-W175` | 同上 | 同上 |
| 20 | 2026-07-27 | `OneReason-0.8B-Frontier-SFT372-DAPO-Anchor-V2.2-W215` | 同上 | `…/dapo_anchor_candidates_W215_W249_20260727_082702` |
| 21 | 2026-07-27 | `OneReason-0.8B-Frontier-SFT372-DAPO-Anchor-V2.2-W249` | 同上 | 同上 |
| 22 | 2026-07-28+ | `OneReason-0.8B-Frontier-SFT372-DAPO-Anchor-Multitask-W450` | **multitask**（first pass 后第一 checkpoint ≈ 完整 1 epoch） | 用户确认；本地无 staging，上传时间未记录 |

## 逐代差异与主要改进

| 代次 | 训练起点 | 核心机制 | 相对上一代的主要改进 | 结果 |
|---|---|---|---|---|
| **ORPO**（07-16） | 基座直训 | 离线偏好对齐：SFT loss + 0.1×odds ratio（32,705 对） | 第一代 | 负面：text 合法率 97%→72% |
| **GRPO**（07-18） | OneReason SFT epoch1 | G=8 在线、forced-GT（50%）、Z-score advantage、clip+KL | 偏好对齐→在线策略优化 | 中性 |
| **Frontier SFT**（07-19） | 基座 | frontier 63,700 条 SFT（focal loss + item 3× 加权） | 数据升级：32,705→63,700（后续 RL 新起点） | 工程过，未评测 |
| **Frontier GRPO**（07-21） | Frontier SFT E2 lora64 | 同 GRPO，数据换 frontier（17,016 组） | 起点升级 | 中性：rec exact 1/512 |
| **Frontier RLOO**（07-23） | Frontier SFT E2 lora64 | G=16 纯在线、留一 advantage、去 KL/clip、**GT-set anchor 兜底**、reward 档 1.0/0.4/0.15/0.01/0 | ① 去 forced-GT（污染分布）② advantage 换留一 ③ 引入 GT-set anchor（专治无梯度组，实测 78.7% 组走 anchor） | 中性偏负 |
| **RLOO-DAPO**（07-25） | **SFT372**（0.9744） | DAPO 非对称 clip [0.8,1.28]、T=1.2、**KV cache**（2.71×）、speculative 修正、只训有效组 | ① clip_higher 防熵坍缩 ② KV cache 提速 | 工程全过，全量未跑完 |
| **DAPO-Anchor 单任务**（07-27） | SFT372 | DAPO + **anchor 独立 Phase 2**（每窗口 4 RL + 1 有界 anchor 步，预算 min(0.05, 0.1×RL/anchor)） | anchor 从"并入最后一步"升级为独立有界步 | 中断（313/1063 窗口） |
| **multitask**（07-28+） | SFT372 | 双任务（recommendation 17,016 + text_to_sid 10,597 组） | 单任务→双任务联合训练 | W450 = 最贴近 1 epoch，在线评测分最高 |

## 三条横向升级线

| 维度 | 演进 |
|---|---|
| 训练数据 | baseline 32,705（ORPO/GRPO）→ frontier 63,700（07-19 起） |
| SFT 起点 | 无 → rank32 → lora64 jrxg7q → **SFT372 tw1g09（0.9744）** |
| reward 档 | 1.0/0.2/0.05（GRPO）→ 1.0/0.4/0.15（RLOO 起） |

## 未确认项

1. 07-19 Frontier SFT epoch1 的 repo ID（本地 README 无 title）
2. multitask W450 的上传时间（本地无 staging）
