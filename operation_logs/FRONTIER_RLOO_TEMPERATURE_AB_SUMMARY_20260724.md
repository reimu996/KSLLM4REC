# Frontier RLOO Temperature 1.0 vs 1.2 执行结果

## why

本实验要判断：固定最终 RLOO Epoch 2 权重后，把在线候选采样温度从 `1.0` 提高到 `1.2`，能否增加同组 reward 差异和 SID 多样性，同时不损害正确 SID 命中能力。

预先声明的最终判定为 `not_promising`。`T=1.2` 明显增加 SID 多样性，但 informative RLOO 组率只增加 `3.5156` 个百分点，未达到 `5` 个百分点门槛；全组落在 `same_domain` 的比例只下降 `1.3672` 个百分点，未达到下降 `5` 个百分点门槛。本实验没有自动启动后续训练。

## 判定层级

判定层级：同一固定 policy 上的合法在线 rollout 分布。

固定项：Epoch 2 adapter、512 个 prompt 组及顺序、每组 GT SID 集合、905,469 叶子的合法前缀树、候选编号 `0..15`、每个候选 seed、`G=16`、`8+8` chunk、最大输出 32 token、五档 reward 和重放算法全部相同。

有且仅有 1 个本质差异：合法动作 logits 在 softmax 前分别除以 `1.0` 或 `1.2`；每个 arm 的采样与重放均使用自己的温度。

## 固定输入

- policy adapter：`artifacts/rloo/runs/frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor_2epoch/epoch_002`
- adapter SHA256：`aeebde52b58efd3685a2925de9b66163d40a91e62467fac7e233b68e11c8fdb2`
- 512 个 group ID SHA256：`53af7f0ac42a8ee0914e2ea8e4a2eac15fe63c432be007e808ab7351ea098189`
- seed：`uint64_be(SHA256("rollout|42|3|group_id|candidate_index")[:8])`
- reward：`exact=1.0, same_ab=0.40, same_a=0.15, same_domain=0.01, other_domain=0.0`
- 实现提交：`a38a714`；CUDA 显存 API 修复提交：`18ee171`
- 两个正式 arm 的执行 Git HEAD：`18ee1710026a1d01313f57048d024205df91cc22`
- 共同运行签名：`7b39088411235e810ed8a858a0ec07bf9df73f1f5a9534eb75b9d0a7c694d163`

## 正式结果

每个 arm 都处理 512 组、8,192 个候选；下表中的百分比均同时保留原始计数。

| 指标 | T=1.0 | T=1.2 | T=1.2 - T=1.0 |
|---|---:|---:|---:|
| informative RLOO 组 | 79/512 = 15.4297% | 97/512 = 18.9453% | +3.5156 pp |
| reward 全相等组 | 433/512 = 84.5703% | 415/512 = 81.0547% | -3.5156 pp |
| 全组 same_domain | 376/512 = 73.4375% | 369/512 = 72.0703% | -1.3672 pp |
| 每组平均唯一 SID | 12.1035 | 13.4023 | +1.2988 |
| 重复候选槽位 | 1,995/8,192 = 24.3530% | 1,330/8,192 = 16.2354% | -8.1177 pp |
| 平均合法动作熵 | 1.2821 nats | 1.5523 nats | +0.2702 nats |
| 平均归一化合法动作熵 | 0.2648 | 0.3177 | +0.0529 |
| 至少一个 exact 的组 | 8/512 = 1.5625% | 9/512 = 1.7578% | +0.1953 pp |
| exact 候选槽位 | 45/8,192 = 0.5493% | 38/8,192 = 0.4639% | -0.0854 pp |
| 平均组内最高 reward | 0.054219 | 0.057305 | +0.003086 |
| 平均候选 reward | 0.034204 | 0.032200 | -0.002004 |
| other_domain 槽位 | 710/8,192 = 8.6670% | 710/8,192 = 8.6670% | 0 pp |

五档候选槽位原始计数：

| reward 档 | T=1.0 | T=1.2 |
|---|---:|---:|
| exact | 45 | 38 |
| same_ab | 127 | 112 |
| same_a | 795 | 769 |
| same_domain | 6,515 | 6,563 |
| other_domain | 710 | 710 |

## 配对解释

8,192 个 `(group_id, candidate_index)` 全部使用相同 seed。`T=1.0` 与 `T=1.2` 的结果中：

- 5,799/8,192 = 70.7886% 的候选 SID 完全相同，即 29.2114% 的 SID 被温度改变。
- 8,086/8,192 = 98.7061% 的候选仍处于相同 reward 档，只有 1.2939% 跨 reward 档。
- `T=1.0` 的 79 个 informative 组在 `T=1.2` 下全部仍 informative；另有 18 个组从 reward 全相等变为 informative。
- 每组 SID 集合的平均 Jaccard 相似度为 0.5745。

因此，`T=1.2` 的主要作用是增加同一 reward 档内部的 SID 多样性。它确实减少重复候选并新增 18 个 RLOO 组，但没有产生足够多的跨 reward 档变化来达到预先声明的信号门槛。`any-exact` 组数从 8 增至 9，但 exact 槽位从 45 降至 38，说明更高温度减少了少数组内对 exact SID 的重复采样。

## 门槛判定

| 预先声明条件 | 实际值 | 结果 |
|---|---:|---:|
| informative 组率至少 +5 pp | +3.5156 pp | 未通过 |
| 每组平均唯一 SID 至少 +0.5 | +1.2988 | 通过 |
| 全组 same_domain 至少 -5 pp | -1.3672 pp | 未通过 |
| any-exact 组率不下降 | +0.1953 pp | 通过 |
| 平均组内最高 reward 不下降 | +0.003086 | 通过 |
| other_domain 槽位率最多 +2 pp | 0 pp | 通过 |
| sampled/replayed log-prob 差 <= 1e-5 | 0.0 | 通过 |
| 所有候选合法、所有数值有限 | 16,384/16,384 | 通过 |

只有全部条件同时通过才允许标记 `promising`，所以最终结论必须是 `not_promising`。这不等于 `T=1.2` 一定降低官方分数；它只表示本次固定 policy、固定 512 组诊断不足以支持把训练温度改为 1.2。

## 安全与复现证据

- SFT、ORPO、GRPO、RLOO 四套 CPU 回归共 225 项通过，`py_compile`、shell 语法检查和 `git diff --check` 通过。
- 两臂均为 `torch.inference_mode()`；backward、optimizer update、scheduler step、anchor evaluation、GT injection、reward 条件重采样均为 0。
- 392 个 LoRA tensor、40,370,176 个参数的运行前后 SHA256 均为 `cd34d515341a91fb26e4faca1279707b9a6a2500ddc53365424867b0af5f9ea0`。
- `T=1.0`：1434.662 秒，peak allocated 2.5874 GiB，peak reserved 2.6797 GiB。
- `T=1.2`：1428.378 秒，peak allocated 2.5874 GiB，peak reserved 2.6797 GiB。
- 两次 8 组 smoke 交换执行顺序后原始审计完全一致；正式运行前 8 组也与 smoke 完全一致。
- 最终 verifier：`passed=true`；合法候选总数 16,384；sample/replay 最大 log-prob 差为 0.0。

第一次 smoke 在模型加载前因 PyTorch 2.7 的显存统计接口不接受 `torch.device("cuda:0")` 而停止，没有生成候选或改动权重。`operation_logs/rloo/frontier_epoch2_temperature_t100_t120_smoke_v1_failed_cuda_device_20260724_013417/` 只保留了启动记录，异常 traceback 没有重定向落盘；失败原因由报错位置和修复 diff 支撑，不把该目录当作完整失败证据。修复提交为 `18ee171`，随后完整 CPU 回归和所有 GPU 验证通过。

## 复现命令

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/rloo/frontier_temperature_ab/test_cpu.sh
scripts/rloo/frontier_temperature_ab/run_smoke.sh
scripts/rloo/frontier_temperature_ab/run_full.sh
scripts/rloo/frontier_temperature_ab/verify.sh
```

核心原始文件位于已忽略目录 `operation_logs/rloo/frontier_epoch2_temperature_t100_t120_v1/`：

- `t100_audit.jsonl`：`ffa676fc83ada4e61dfd6b0915c4bc35cbc7a9653bbf20f3e83385cdf0ee2064`
- `t120_audit.jsonl`：`8e07731d2303f83c6b87b1f995f544313ace499ae0957d6f9e433e1958e398ef`
- `t100_summary.json`：`9e0001a68dfb63330c3cada082e04d277c8b32f40327b37232aa24e139de0ae9`
- `t120_summary.json`：`570bf4c5fd7951ab22fc6e4b231d1ea1b38cc99fb22401f52c430daa1da5375d`
- `comparison.json`：`fc4b254f0b6d5e9496eb5c5b2e7d70f69a9f05b0958bf516d9137151be2dedfa`
- `verification.json`：`fc503b6b83c34ae70d2b3d0afb66c4ab62f2ad81f6c611798fda1c71aae21930`
- `first8_determinism.json`：`3c816762ed3dfd0854739bb68efc79a2ea34e1ea8813479c77cbf81333d84892`

原始 JSONL 和 JSON 不提交 Git；配置、实现、测试、脚本和本摘要提交 Git。
