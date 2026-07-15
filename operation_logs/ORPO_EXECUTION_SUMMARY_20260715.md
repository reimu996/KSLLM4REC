# OneReason-0.8B 从 Base 直接 LoRA ORPO 两轮执行总结（2026-07-15）

## 1. 目标与结论

why：本次实验要验证“从干净 OneReason-0.8B 的第一个优化步开始，直接联合 chosen-SFT 与 chosen/rejected 偏好训练”能否在单张 RTX 4090 上完整执行，并在未进入训练的固定 Explorer probe 上优于干净 base。

工程结论：执行成功。`32,705` 条偏好对完成两轮 ORPO，共 `8,178` 个优化步；四档显存门、双 checkpoint 独立加载、输入/环境/实现哈希校验全部通过。训练峰值 reserved 显存为 `4.869140625 GiB`，没有 OOM、NaN 或 Inf。

性能结论：本地证据为负。epoch 2 的训练 loss、pair accuracy 和 margin 都优于 epoch 1，但 epoch 1/2 在固定 Explorer text-to-SID probe 上都明显弱于干净 base；recommendation exact 也没有超过 base。因此本次两个 ORPO adapter 都不建议直接作为当前最佳提交。

本结论只覆盖本地固定 probe，不等价于平台隐藏测试集分数；本次没有上传或提交比赛平台。

## 2. 受控实现与输入

| 对象 | 值 |
| --- | --- |
| ORPO 实现提交 | `b19b41345e7ed008d3acdecb3e487be0cd271efe` |
| 实现提交 tree | `4bdacfd136834a4c7161c74917618fca06669acf` |
| 训练时 Git parent | `6c3fd4d58d0c0d8e984f5d7c930fd73996f628ab` |
| 受控实现指纹 | `e02af8e86072c4db612f111b6b4768ca3758909a295db063536912acf1337dfd` |
| base | `/home/lyc/models/OneReason-0.8B-pretrain-competition` |
| base weights SHA256 | `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90` |
| source dataset | `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl` |
| source SHA256 | `4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8` |
| source records | `32,705` |
| LLaMA-Factory | commit `0b7aaf8f6a624bd89a01a155d4265ec861cbdf38`；tree `02158cfe15d2019e3353901d61440b5a320993ab` |
| Python / Torch | `3.11.15` / `2.7.1+cu126` |
| Conda 环境 | `/home/lyc/miniconda3/envs/onereason_lora_sft` |
| Python package lock | 123 个包，逐行校验通过 |

训练启动时 ORPO 文件尚未进入 Git，所以运行 manifest 的 `git` 字段记录 parent `6c3fd4d`。manifest 同时逐文件记录了全部 ORPO 配置、脚本和源码，并得到实现指纹 `e02af8...`。训练后对准备提交的文件重新运行配置检查，指纹仍为 `e02af8...`；这些完全相同的文件随后提交为 `b19b413`。因此复现身份以实现指纹为主，不能把 parent commit 单独当成训练代码版本。

## 3. 实际使用的 loss

从第一个优化步到最后一个优化步都使用同一个 ORPO loss；不存在先 SFT、再 ORPO 的阶段，也没有使用 focal loss。

```text
chosen_logp   = chosen response 的平均 token log-prob
rejected_logp = rejected response 的平均 token log-prob

chosen_log_odds   = chosen_logp   - log(1 - exp(chosen_logp))
rejected_log_odds = rejected_logp - log(1 - exp(rejected_logp))
log_odds = chosen_log_odds - rejected_log_odds

sft_loss = -chosen_logp
preference_loss = -log(sigmoid(log_odds))
pair_loss = sft_loss + 0.1 * preference_loss
loss = batch 内 pair_loss 的平均值
```

`log(1-exp(x))` 使用稳定分段计算。sequence average log-prob 使用 `512` token 分块的冻结 LM Head，不物化完整 `[2B,L,176253]` FP32 logits。只更新 LoRA A/B；reference model 未加载。

## 4. 偏好对生成结果

### 4.1 prompt 组口径修正

第一次 `prepare_pairs` 在写出最终候选前失败，退出码为 1：

```text
RuntimeError: Recommendation positive groups do not match the frozen contract:
groups=6378, pairs=18651
```

原因是初始 Spec 把原始 prompt 字符串分组数 `9,829` 错写成规范化分组数。`/think` 与 `/no_think` 后缀规范到同一个训练 prompt 后，实际是 `6,378` 组，仍包含 `18,651` 个正例。最终实现按 `6,378` 个规范化组构建完整 positive set，避免把同一训练输入的另一个合法 SID 当成 rejected。

失败记录：`operation_logs/orpo/prepare_pairs_20260715_033212_621627478/console.log`。

修正后的成功记录：`operation_logs/orpo/prepare_pairs_20260715_040310_891584018/console.log`。

### 4.2 最终 pair

| 任务 | pair | Tier 1 | Tier 2 | Tier 3 |
| --- | ---: | ---: | ---: | ---: |
| recommendation | 18,651 | 17,707 | 935 | 9 |
| text-to-SID | 5,197 | 3,929 | 1,212 | 56 |
| SID-to-text | 4,487 | 183 | 4,275 | 29 |
| user-interest | 2,792 | 2,792 | 0 | 0 |
| CEval | 1,578 | 1,578 | 0 | 0 |
| 合计 | 32,705 | 26,189 | 6,422 | 94 |

SID-to-text 的 Tier 2 中，baseline 同 `s_a` 为 `2,879` 条，Explorer 同 `s_a` 为 `1,396` 条。构造过程只读扫描了 `35,914,095` 行 Pid2Sid 和 `19,701,886` 行 Explorer caption；无非法 Pid2Sid 行。干净 base 共执行 `24,351` 次 hard-negative mining forward。

产物：

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| `candidates.jsonl` | 279,204,035 | `519fa7020d931e44787efc3293d4c59fec1468a024e8a6cc7fff12a7ce2a8516` |
| `audit.jsonl` | 267,227,013 | `561295e08554e9c19899a5484f2e25c4a32832d8da4f0199a2e21a1c313b1c3e` |
| `train.jsonl` | 227,109,426 | `ab96da904229473dfd1a1898013fef7692cb879be78e33f17dc187ca3cb97c94` |
| `pair_manifest.json` | 1,447 | `5764ec5bbdb786310e577c46bd7bcbd767cffcc5325ddecf805f4086a299c9ba` |

exactly `32,705` 个 pair_id 全部唯一。第二次从空输出目录重新 mining 后，`audit.jsonl`、`train.jsonl`、`pair_manifest.json` 与第一次逐字节相同；第二次日志为 `operation_logs/orpo/repro_pair_002/console.log`。

## 5. 数据预检与固定 probe

| 项目 | 结果 |
| --- | ---: |
| 最长原始 pair | `9,752` token |
| 最长 source | `9,453` token |
| 最长 chosen response | `538` token |
| 最长 rejected response | `538` token |
| `cutoff_len=16384` source 裁剪 | `0` |
| chosen/rejected 裁剪 | `0` |

固定 probe 由未进入 baseline/pair 训练的 Explorer 行组成：512 条 recommendation 加 512 条 text-to-SID。probe SHA256 为 `8b8c64bcf9f4590dc006f900ded27d7008816d56ea1b3df6ccfd55a4e63b5fb1`。

本次只运行 greedy `max_new_tokens=64`，报告生成文本中的第一个完整 SID 是否存在、是否等于单一目标 SID。没有运行未定义候选集的 Recall@8/32，也没有运行随机采样 Pass@8/32。

## 6. 四档显存门

每档都真实执行 8 个 microbatch、1 个 backward 和 1 个 optimizer step；不是只做前向或只改配置名。

| 门禁 | 实际序列长度 | 峰值 reserved 显存 | 结果 |
| --- | ---: | ---: | --- |
| 512 | 512 | `2.060546875 GiB` | passed |
| 2,048 | 2,048 | `2.5078125 GiB` | passed |
| 8,192 | 8,192 | `4.365234375 GiB` | passed |
| 16,384 | 16,384 | `6.845703125 GiB` | passed |

四档 manifest 的实现指纹都等于 `e02af8...`。`artifacts/orpo/gates_report.json` SHA256 为 `0f541d94edf8023984f775fed639db97998077bda7eec5338175a991a968d02f`。

## 7. 完整训练

### 7.1 固定配置

| 项目 | 值 |
| --- | --- |
| 起点 | clean base；没有 adapter |
| LoRA | rank 32；alpha 32；dropout 0；q/k/v/o/gate/up/down |
| 可训练参数 | `20,185,088 / 821,618,688 = 2.4567%` |
| batch / accumulation | `1 / 8`；effective batch 8 |
| cutoff / packing | `16384 / false` |
| epoch / seed | `2.0 / 42` |
| optimizer | AdamW；LR `1e-4`；weight decay `0.001` |
| schedule | cosine；warmup ratio `0.03` |
| precision | BF16、TF32、FlashAttention 2 |
| ORPO | beta `0.1`；LM chunk `512`；无 reference model |

### 7.2 运行结果

| 指标 | 结果 |
| --- | ---: |
| run_id | `full_orpo_epoch_002_20260715_044814_407422461` |
| optimizer steps | `8,178` |
| micro steps | `65,410` |
| completed epoch | `2.0` |
| 最长实际序列 | `9,752` |
| Trainer train loss | `1.000548892074822` |
| runtime | `20,067.7213` 秒，即 `5:34:27.72` |
| samples/s | `3.259` |
| optimizer steps/s | `0.408` |
| 峰值 allocated | `4.727444648742676 GiB` |
| 峰值 reserved | `4.869140625 GiB` |
| resume checkpoint | 无，从 clean base 启动 |
| reference model | 未使用 |

完整 manifest：`operation_logs/orpo/full_orpo_epoch_002_20260715_044814_407422461/manifest.json`。

### 7.3 两轮训练窗口

下表是 `trainer_state.json` 中每 5 个 optimizer step 日志窗口的等权平均，不是验证集指标。

| 指标 | epoch 1 | epoch 2 |
| --- | ---: | ---: |
| 日志窗口 | 817 | 818 |
| loss | `1.103721` | `0.897628` |
| SFT loss | `1.036923` | `0.846768` |
| odds-ratio loss | `0.668007` | `0.509399` |
| pair accuracy | `49.5838%` | `75.7648%` |
| reward margin | `0.015003` | `0.044900` |
| chosen logp | `-1.036923` | `-0.846768` |
| rejected logp | `-1.186957` | `-1.295769` |

训练目标内部明显改善：chosen 概率提高、rejected 概率降低、pair accuracy 和 margin 增加。但第 9 节显示，该变化没有转化为固定 Explorer probe 提升。

## 8. Adapter 验证

| checkpoint | epoch | adapter SHA256 | bytes | LoRA B 总 L2 norm |
| --- | ---: | --- | ---: | ---: |
| `checkpoint-4089` | 1.0 | `c367c92cd09f5b8d8ee896a8928977767cbf544a0e261f5d318aba7bc2cb5c68` | 80,792,456 | `16.669607752838843` |
| `checkpoint-8178` | 2.0 | `593a117229c11f8bc488acb4e6402c6ec879be116b6d063dd1901974c6fe6a27` | 80,792,456 | `18.284580902129466` |

两者的 `adapter_config.json` 均为 1,082 bytes，SHA256 为 `b8847f943c2f2050181c4ae2db76b189d5c387f9fafb2c3e4aa7bdbbf44dead7`；每个 adapter 含 196 个 LoRA A 和 196 个 LoRA B 张量，全部有限。

独立加载结果：

- epoch 1 相对 base 的最大绝对 logit 差为 `11.3125`。
- epoch 2 相对 base 的最大绝对 logit 差为 `12.25`。
- 两者都能生成有限 token，验证状态均为 passed。

验证报告：`operation_logs/orpo/verify_full_20260715_102318_101101984/verification.json`，SHA256 `615bbf670e05fbfcd3f9c8739516095b19aac0e3a5124d662429d06f5d324e3e`。

最终输出目录 `artifacts/orpo/runs/orpo_from_base_epoch2` 是 epoch 2 adapter 的副本。基础模型没有合并进 adapter，也没有生成全参数 `model.safetensors`。

## 9. 固定 probe 结果

### 9.1 汇总

| 模型状态 | recommendation 有效 SID | recommendation exact | text-to-SID 有效 SID | text-to-SID exact |
| --- | ---: | ---: | ---: | ---: |
| clean base | `512/512 = 100%` | `1/512 = 0.1953%` | `498/512 = 97.2656%` | `5/512 = 0.9766%` |
| ORPO epoch 1 | `512/512 = 100%` | `0/512 = 0%` | `371/512 = 72.4609%` | `1/512 = 0.1953%` |
| ORPO epoch 2 | `512/512 = 100%` | `0/512 = 0%` | `438/512 = 85.5469%` | `2/512 = 0.3906%` |

报告与预测哈希：

| 状态 | report SHA256 | predictions SHA256 |
| --- | --- | --- |
| base | `bfcf1471fab8505d3aabfc5ba061ad616ee9c9cddfd0a164862a903e206264ff` | `2bb5d4a9bba65f65a9f697ac24a0d7d87cf8adea9b51eec0ade85a4d2a76ff30` |
| epoch 1 | `3cfd7f069f9ecbac3547469f1e85a9687571b6511de3024e265bf1d1a7bd0702` | `6bc55763fd33b4c79b3a20c113b365636e7fee53915a1ec02c4b7c1f3dca060d` |
| epoch 2 | `ffe605bc25d2ea533a02262ed69ca414cc138963725d0b14383936c20a5aec52` | `fe097a4293586acc4ecc52b19d8ad87e4c8409b573718c4d91a9447f2e0bf305` |

### 9.2 逐样本变化

- recommendation：base 唯一 1 条 exact 在 epoch 1 和 epoch 2 都变错；两个 adapter 没有新增 exact。
- text-to-SID epoch 1：base 的 5 条 exact 中保留 1 条、丢失 4 条，没有新增 exact。
- text-to-SID epoch 2：base 的 5 条 exact 全部丢失，另有 2 条原本错误的样本变为 exact。
- text-to-SID epoch 1：127 条“base 有合法 SID”的样本变成无 SID。
- text-to-SID epoch 2：63 条“base 有合法 SID”的样本变成无 SID，另有 3 条 base 无 SID 样本恢复合法 SID。

recommendation 的 exact 计数只有 0 或 1，统计量过稀，不能单独支持强结论。text-to-SID 的合法 SID 率从 `97.27%` 降到 `72.46%/85.55%`，差异更大，是本次否决 adapter 的主要本地证据。

### 9.3 失败形态

epoch 1 的 text-to-SID 有 `141/512` 条无完整 SID，epoch 2 有 `74/512` 条，base 只有 `14/512` 条。adapter 的无 SID 输出并非空文本，而是中文内容描述，例如以“该视频为一段……”开头，表现为把 text-to-SID 输入执行成了相反方向的 SID-to-text/内容描述任务。

这是可观测失败形态，不足以单独证明根因。合理但尚未受控验证的根因候选包括：

1. 五任务联合训练中的方向干扰，尤其 text-to-SID 与 SID-to-text 同时更新相同 LoRA 参数。
2. recommendation 占 `18,651/32,705 = 57.03%`，逐 pair 等权时任务权重不平衡。
3. direct no-think 删除了 11,771 条 teacher CoT，可能降低任务区分线索。
4. `1e-4`、两轮的累计更新对当前 base 过强。

没有 CE-only、本地 focal adapter 或任务分离 ORPO 的同 probe 对照，因此不能从本次实验中确定四者谁是主因。

## 10. 提交决策

当前不建议提交 `checkpoint-4089` 或 `checkpoint-8178` 作为最佳模型。

判定依据：

1. 两个 adapter 都未超过 base 的 recommendation exact。
2. 两个 adapter 都显著降低 text-to-SID 合法 SID 率和 exact。
3. epoch 2 虽比 epoch 1 恢复一部分格式能力，但仍明显低于 base。
4. 训练 pair 指标改善与 probe 退化同时发生，不能使用训练 loss 下降作为提分证据。

不删除两个 adapter：它们是完整、可加载的负实验产物，可用于后续受控对照。

下一步的最高信息量实验不是直接再跑一个更复杂目标，而是先用同一 probe 评估现有 CE/focal adapter，确认 probe 与已有路线的方向一致；再比较“降低 LR/只训 1 epoch”“任务均衡采样”“只对 SID 输出任务做 ORPO”。这些都属于新 Spec，本次未执行。

## 11. 验收条件结果

| AC | 状态 | 证据 |
| --- | --- | --- |
| AC-001：32,705 direct no-think chosen，最终答案保持 | passed | pair audit 与 source hash |
| AC-002：32,705 合法且唯一 pair | passed | `pair_manifest.json` |
| AC-003：recommendation rejected 不在规范化 positive set | passed | 全量 audit；6,378 组 |
| AC-004：user-interest rejected SID 不相交优先规则 | passed | 2,792/2,792 Tier 1 |
| AC-005：pair 独立重生成逐字节一致 | passed | `repro_pair_002` 与三份相同 SHA256 |
| AC-006：chunked/full logp、loss、hidden gradient 对齐 | passed | 19 个单元测试中的数值等价测试 |
| AC-007：极端 logp 的 loss/gradient 有限 | passed | 单元测试与完整训练 |
| AC-008：四档真实显存门通过 | passed | `gates_report.json` |
| AC-009：clean base 连续完成 2 epoch ORPO | passed | full manifest；8,178 steps |
| AC-010：双 checkpoint 可独立加载、LoRA B 非零 | passed | verification report |
| AC-011：输入、配置、命令、日志和输出可定位 | passed | 锁文件、run_id、manifest 与本总结 |
| AC-012：未把本地结果表述成线上提分 | passed | 本总结明确记录负结果和边界 |

## 12. 复现命令

前置模型、数据、LLaMA-Factory 和 Conda 环境仍位于固定路径时：

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC

scripts/orpo/freeze_sources.sh
scripts/orpo/prepare_pairs.sh
scripts/orpo/run_config_check.sh
scripts/orpo/run_gates.sh
scripts/orpo/run_full.sh
scripts/orpo/run_probes.sh
```

全新产物根目录也可执行 `scripts/orpo/run_all.sh`。当前 `run_probes.sh` 使用防覆盖语义；同名 probe 输出目录已存在时会拒绝覆盖，复跑前必须先归档旧结果。

最终静态验证：

- ORPO 单元测试：19/19 passed。
- Ruff：passed。
- 全部 `scripts/orpo/*.sh`：`bash -n` passed。
- 配置、输入、环境合同：passed。
- 训练后重新计算实现指纹：仍为 `e02af8...`。

## 13. 未执行事项

- 未上传 Hugging Face、ModelScope 或比赛平台。
- 未运行平台隐藏测试集，不能报告比赛分数。
- 未评估现有 focal adapter 或公开 SFT adapter 的同 probe 指标。
- 未执行 CE vs ORPO、不同 beta、不同 LR、任务均衡或 CoT 保留的受控消融。
- 未合并 adapter 到 base；提交 LoRA 时仍需 base 与标准 PEFT 文件。
- 大模型、偏好数据、预测明细和原始日志不进入 Git；本机删除后需按锁与命令重建。

一句话结论：单张 RTX 4090 可以用约 `4.87 GiB` 峰值 reserved 显存从干净 OneReason-0.8B 直接完成两轮 rank-32 LoRA ORPO，但本次构造在固定 Explorer probe 上损伤 text-to-SID 泛化，因此工程验收通过、性能门禁不通过，两个 adapter 均不建议直接提交。
