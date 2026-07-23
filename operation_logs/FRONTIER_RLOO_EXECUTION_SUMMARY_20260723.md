# Frontier SFT Epoch 2 r64 -> Reference-Free G16 RLOO 执行摘要

## 目标与结论

本实验要在不向 rollout 强制塞入 GT、也不加载 reference model 的前提下，检验 G=16 在线 RLOO 加条件 GT 集合锚点能否继续优化 Frontier SFT Epoch 2 adapter。两轮训练、三组固定 probe 和最终 verifier 均已自然结束；没有 CUDA OOM、chunk 降级或 GT 注入。

训练分布内的平均 reward 和 exact 候选率在第二轮上升，但固定 recommendation probe 没有超过训练起点。因此本次结果证明流程可复现、权重确实更新，不证明官方比赛性能提升。

## 冻结输入

- Base model：`/home/lyc/models/OneReason-0.8B-pretrain-competition`
  - `model.safetensors` SHA256：`28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90`
- 初始 Frontier SFT Epoch 2 r64 adapter：`artifacts/sft/platform_exports/frontier_feedbackcore_listwise_invariant_epoch2_lora64_20260722/extracted`
  - 权重 SHA256：`c31f1e67cd46b02a27a1ed44136eccb1f83b7055f247c84d4a6924cafa4c1d70`
  - 配置 SHA256：`041e3f79eaa6acfa181612838c03e275f0fa7883e0b286c82fbb6f164714731e`
- Frontier 数据：`/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl`
  - SHA256：`9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a`
- 推荐组：`artifacts/grpo/data/recommend_groups_frontier_v1/groups.jsonl`
  - 17,016 个 prompt 组、30,465 条去重正例边
  - SHA256：`a2571d97e4ac231d61d1f468bc23b8e65bebfd5ad7b1a2dd566f1bfd86a9658e`
- 合法 SID 前缀树：`artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1`
  - 905,469 个合法 SID；manifest SHA256：`f51414b13a8982605ebe1f7fec75929245db60c854e3860d3b6c1625a3f33e4a`
- 固定 probe：`artifacts/grpo/data/frontier_probe_v1/fixed_probe_1024.jsonl`
  - SHA256：`86e050a00506efbaef0e49997ed138462aaddc322eba9fc66227a25654a75e5e`
- 校准组 ID：`configs/rloo/frontier_sft_epoch2_lora64_calibration_ids.json`
  - SHA256：`2dccbfc3e5298b416168b0af3d2bed5c5ae63beb415af9201b4b03d3966b94d6`
- `tokenizer.json` SHA256：`cd4d15f596979aecbc11ba12668cb21c9fb8452ce1235cd9d7878cc950170421`
- 实现 commit：`3a1a2fa feat: add reference-free Frontier RLOO workflow`
- 正式运行签名：`315c1a90a5a0c97c4fa0dd99dce50696726b408a89aaf8f919b74d0c260e0bb2`

这套源数据、推荐组、正例边、合法 SID 前缀树和固定 probe 与紧邻的 Frontier GRPO 实验完全相同。它与更早使用 `HF_kuaishou-llmrec-sft-baseline-0.91` 的 baseline GRPO 不同。

文中术语固定如下：`GT` 是数据给出的正确 SID 集合；`group` 是归一化后 prompt 相同的一组训练目标；`policy` 是当时正在训练并负责现场采样的模型；一个“候选槽位”是 G=16 中的一次采样位置，两个槽位允许得到同一个 SID；`RLOO` 是“每个候选的 reward 减去其余 15 个候选 reward 均值”的组内基线算法。

## 训练合同

- 每个 prompt 由当前 policy 现场采样 16 个候选；保留重复候选，无 beam、无 replay buffer。
- 采样只能沿完整合法 SID 前缀树前进；分支节点只对合法 token 做 FP32 softmax。
- 一个 SID 由 `domain + a + b + c` 四段组成，例如 `<|video_begin|><s_a_6207><s_b_5619><s_c_4704>`。每个候选与该组任一 GT 比较并取最高档：四段相同为 exact `1.00`；domain/a/b 相同但 c 不同为 same_ab `0.40`；domain/a 相同但未命中前两档为 same_a `0.15`；只有 domain 相同为 same_domain `0.01`；domain 不同为 other_domain `0.00`。优先级固定为 `exact > same_ab > same_a > same_domain > other_domain`。
- 奖励不全相同时使用 RLOO：每个候选减去其余 15 个候选的平均 reward，不除以标准差。
- 16 个 reward 全为 1 时跳过；其他 reward 全相同时使用全部唯一 GT SID 计算集合锚点 loss：每条 GT 路径只累加合法分支位置的 token log-prob，再对所有路径分数计算一次全局 `-logsumexp`。这不是标准逐 token SFT CE。
- GT 只用于 reward 和锚点目标；不进入 16 个在线候选，正式训练 `gt_injection_count=0`。
- 每 8 个 prompt 组成一个更新窗口；窗口目标为 RLOO 组均值加 `lambda0` 乘锚点组均值。
- `lambda0=0.0001500929949371556`，由固定 512 组的两类 FP32 梯度范数校准得到。
- LoRA r64、alpha64、运行期 dropout 0；LR `5e-6`，128 窗口 warmup，cosine，max grad norm 1。
- 两轮共 4,254 个更新窗口；cutoff 16,384；BF16、TF32、gradient checkpointing 开启。
- reference model、KL、PPO clipping、GT 注入和 reward Z-score 全部禁用。

## 开跑前门禁

六份报告使用同一运行签名；除只检查冻结结构的 structure 报告外，其余五份均绑定校准得到的同一 `lambda0`。六项均通过当前实现的开跑前门禁。

| 门禁 | 固定规模 | 关键实测值 | 结果 |
|---|---:|---:|---:|
| 结构合同 | 冻结输入与参数 | 结构、文件哈希、运行签名一致 | 通过 |
| 梯度校准 | 512 组 | RLOO/anchor=`440/72`；梯度范数=`0.0182787/12.1782` | 通过 |
| 采样/重放概率 | 1 组、16 候选 | 最大 decision log-prob 差 `0` | 通过 |
| 最长 prompt 显存 | 32 组、512 候选 | peak reserved `10.0527 GiB` | 通过 |
| 训练信号 | 512 组 | RLOO/anchor=`409/103`；RLOO 率 `79.88%` | 通过 |
| 端到端耗时 | 256 组 | `1,367.793 s`；两轮投影 `60.6103 h` | 通过 |

`min_rloo_group_rate=25%` 只由 calibration、signal、timing 的固定预跑样本执行。它用于判断起点是否有足够的非零 RLOO 训练信号，不是要求训练后的 policy 在两轮全量数据上保持 25%。

## 正式训练结果

正式训练耗时 `125,953.079 s = 34.98697 h`，吞吐为 `0.270196 groups/s`。两轮共访问 34,032 个 group；“访问”包含同一 17,016 个组在两轮各出现一次，不表示 34,032 个不同 prompt。

| 指标 | Epoch 1 | Epoch 2 | 两轮合计 |
|---|---:|---:|---:|
| group 访问 | 17,016 | 17,016 | 34,032 |
| 在线候选槽位 | 272,256 | 272,256 | 544,512 |
| RLOO / anchor / skip 组 | 4,316 / 12,698 / 2 | 2,910 / 14,089 / 17 | 7,226 / 26,787 / 19 |
| RLOO 分支率 | 25.3644% | 17.1016% | 21.2330% |
| 平均 reward | 0.0226136 | 0.0309773 | 0.0267954 |
| exact 候选 | 689 | 1,798 | 2,487 |
| exact 槽位率 | 0.2531% | 0.6604% | 0.4567% |
| 至少一个 exact 的组 | 287 | 373 | 660 |
| 每组平均唯一候选数 | 13.4778 | 12.4662 | 12.9720 |
| 重复候选槽位率 | 15.7638% | 22.0862% | 18.9250% |

五档 reward 的完整候选计数：

| Reward 档位 | Epoch 1 | Epoch 2 | 两轮合计 |
|---|---:|---:|---:|
| exact | 689 | 1,798 | 2,487 |
| same_ab | 1,922 | 2,961 | 4,883 |
| same_a | 16,360 | 21,560 | 37,920 |
| same_domain | 224,488 | 221,735 | 446,223 |
| other_domain | 28,797 | 24,202 | 52,999 |

第二轮相对第一轮：平均 reward 上升 `36.99%`，exact 槽位率变为 `2.61` 倍；同时 RLOO 分支率下降 `8.26` 个百分点，重复候选槽位率上升 `6.32` 个百分点。平均唯一候选下降和重复槽位上升与 policy 分布变得更集中一致；这是采样统计支持的推断。直接观测事实是 exact 采样增加、产生奖励差异的组变少，anchor 分支更占主导。

正式训练的总体 RLOO 率 `21.23%`，Epoch 2 为 `17.10%`。它们低于开跑前的 `25%` 抽样阈值，但不是当前实现的训练后失败条件。最终 verifier 的 `passed=true` 不能表述为“正式两轮 RLOO 率均达到 25%”。

其余工程指标：

- 4,254 个窗口全部发生 optimizer update；正式峰值 reserved 显存 `10.681640625 GiB`。
- 所有 544,512 个候选均在合法前缀树中；所有 reward 和 advantage 有限，所有参与反向的 loss 有限；19 个 skip 组按设计记录 `loss=null`。
- sampled/replayed decision log-prob 最大绝对差为 `0`。
- 两轮所有组的 live/final reward 与 exact 率一致；GT 注入数为 `0`。
- 裁剪前梯度范数最大 `19.723486`；860/4,254 个窗口超过 1，随后按配置裁剪。
- 按真实窗口公式重构的目标均值为 Epoch 1 `-0.0057793`、Epoch 2 `-0.0092069`。RLOO advantage 同时有正值和负值，因此该策略目标允许小于零；数值更负不等价于普通 CE 下降，也不能单独证明性能改善。
- `run_summary.loss_mean=12.4886` 混合了未乘 `lambda0` 的 anchor 原始 loss，不是实际窗口目标，不能用于跨 epoch 或跨方法比较。

## 固定 Probe

三版均使用同一组 1,024 条输入和 constrained beam=16。每版包含 512 条 recommendation、512 条 text-to-SID；每个 target 都可由前缀树到达，每个预测 SID 都合法。

| Adapter | recommendation exact | text-to-SID exact | 合法预测 |
|---|---:|---:|---:|
| 训练起点 | 2/512 | 78/512 | 1,024/1,024 |
| Epoch 1 | 0/512 | 79/512 | 1,024/1,024 |
| Epoch 2 | 1/512 | 79/512 | 1,024/1,024 |

Epoch 2 的 text-to-SID 比起点多命中 1 条，recommendation 比起点少命中 1 条。本地固定 probe 没有证据支持 recommendation 整体提升，也不能替代官方比赛评测。

probe 报告 SHA256：

- 起点：`d1dce3bd5784f19fe2b54ec8ae50155fd197eefc8787f5ebdf1d5a4864223016`
- Epoch 1：`5167b5a6c1ccee5d6bb7e6675e73891029fe2958632dde038d4435b13fa876dc`
- Epoch 2：`3817dc6fce4b3a3d35ffae576bf5c9c1035b7c46170d12d875d6973b88abbb19`

## Adapter 产物

| Adapter | 路径 | 大小 | `adapter_model.safetensors` SHA256 |
|---|---|---:|---|
| Epoch 1 | `artifacts/rloo/runs/frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor_2epoch/epoch_001` | 161,533,160 B | `796a960c1759770f19cdde632bafeac56d848f6aac64436de413933cc3c66dc2` |
| Epoch 2 | `artifacts/rloo/runs/frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor_2epoch/epoch_002` | 161,533,160 B | `aeebde52b58efd3685a2925de9b66163d40a91e62467fac7e233b68e11c8fdb2` |

两轮 `adapter_config.json` 均为 1,081 B，SHA256 均为 `32e96266efa21abaebd076e9287ca108c3a5abab0c4be3aa1e13fe3427bb109e`。

逐张量 CPU 差分确认两个输出 adapter 都不是初始 adapter 的原样复制：392/392 个 LoRA tensor 均发生变化，三版 tensor 的 key、shape 和 dtype 一致且数值有限。初始到 Epoch 1 的 L2 变化为 `1.701786`，初始到 Epoch 2 为 `1.799072`，Epoch 1 到 Epoch 2 为 `0.373253`。

## 验证证据

- CPU 回归：SFT 59 项、ORPO 19 项、GRPO 62 项、RLOO 72 项，共 212 项全部通过；RLOO 子套件耗时 1.459 秒。
- `train_audit.jsonl`：34,032 行；`train_progress.jsonl`：4,254 行。
- 最终状态：`epoch_index=3`、`next_group_offset=0`、`window_step=4254`、`optimizer_update_step=4254`。
- 最终 recovery：`checkpoint-window-004254-update-004254`；manifest SHA256：`87cd7a8e5a61ac2e7988cd42e088e4664eb90bb2f8e87346d9633c4c08bad3fb`。
- `run_summary.json` 与 `full_train_result.json` 字节一致；SHA256：`fbe87c3aa83350f85286b332770d16d2af4d43f24777c56e037ca74404b51f75`。
- `final_verification.json` 的 `passed=true`；SHA256：`cc4af20336ad0d505aad0a93d3e42c83f89426c58a4892bcaba031e7b9859569`。

六份门禁报告 SHA256：

| 报告 | SHA256 |
|---|---|
| structure | `4ed8cfb50c31d40581d93e61431b5931ec19ba44652c79700b0b22dcb1774b3f` |
| calibration | `e0e08082bd43b123b68c99c81d164d2a9e684cf02b1c3e0577a667b48dc845ed` |
| probability | `fb4ecfeff90d65faa27426471301cecde90046d7db5e9d6d9cfba0cd8f99a02e` |
| memory | `0291746e29952cb8093e786f0bafa3b659ec50e659edbb2cf595ca89cf3b81fb` |
| signal | `9bafd0d181e31cdc5c36790503f05693e5ebdcad00afe13b0bc24e3fcae644bf` |
| timing | `0d74d4f0bb1367db2497e345eee7306f645e8f73e43d36b7116681c010972b1a` |

## 复现入口

唯一运行环境是 `/home/lyc/miniconda3/envs/onereason_lora_sft`；本实验没有安装或升级依赖。运行签名记录的关键版本为 Python 3.11.15、PyTorch 2.7.1+cu126、Transformers 4.57.1、PEFT 0.18.1、FlashAttention 2.7.4.post1。

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
bash scripts/rloo/frontier_lora64_g16/prepare.sh
bash scripts/rloo/frontier_lora64_g16/test_cpu.sh
bash scripts/rloo/frontier_lora64_g16/run_gates.sh
bash scripts/rloo/frontier_lora64_g16/run_epoch0_probe.sh
bash scripts/rloo/frontier_lora64_g16/run_full.sh
bash scripts/rloo/frontier_lora64_g16/run_trained_probes.sh
bash scripts/rloo/frontier_lora64_g16/verify.sh
```

两轮 adapter 目前只保存在上述本地路径。本次已确认范围不包含 Hugging Face 上传，不能把本地训练完成写成“已上传”。

一句话结论：本次与 Frontier GRPO 使用完全相同的数据资产，RLOO 两轮训练工程验收完成且在线 reward 上升，但固定 recommendation probe 没有超过起点。
