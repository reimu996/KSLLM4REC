# OneReason-0.8B LoRA GRPO V3.1

## 目标

这条流程要解决的问题是：从已经完成的非 ORPO SFT adapter 继续做在线策略训练，让同一个推荐 prompt 生成的 8 个合法 SID 之间形成可学习的相对奖励，同时保持全部输入、随机选择、恢复点和最终 adapter 可审计、可复跑。

训练不会读取 Explorer 目录来扩大合法 SID 集合。合法集合只来自 baseline `train.jsonl` 的 `system`、`prompt`、`response` 三个字符串字段。

## 固定输入

| 对象 | 路径 | SHA256 |
|---|---|---|
| base 权重 | `/home/lyc/models/OneReason-0.8B-pretrain-competition/model.safetensors` | `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90` |
| 起始 SFT adapter | `artifacts/sft/runs/full_epoch_001/adapter_model.safetensors` | `77ac75d6ad558bdafc97cfb096f9608c59f4382c2884f17dceaf0fd943e0b710` |
| 起始 SFT adapter 配置 | `artifacts/sft/runs/full_epoch_001/adapter_config.json` | `5128629f9fe624a5807a8e8e728d3a8763936a01f8c9bc9df1ce4b9d00a9c5ec` |
| baseline 数据 | `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl` | `4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8` |
| 推荐训练组 | `artifacts/grpo/data/recommend_groups_v3_1/groups.jsonl` | `1c85ad933db46284d1417b7c2e83e31c2d379997a4e966439435f04ca75a243f` |
| baseline 前缀树 manifest | `artifacts/grpo/catalog/baseline_all_sids_v3_1/manifest.json` | `737b2f69a568ba21d707f88b9fbd52a2d3c53ebb1b3aab6f2478e87530dbc066` |

## 数据对象

`RecommendationGroup` 表示一个训练环境：

- `group_id`：`system + 归一化 prompt` 的 SHA256；字符串；决定所有候选随机种子。
- `positive_sids`：这个 prompt 的全部正确 SID；`tuple[str, ...]`；每组 1 到 23 个。
- `source_lines`：对应 baseline 原始行号；用于追溯。

冻结计数：

- baseline 总行数：32,705。
- 推荐行数：18,651。
- 合并重复 prompt 后：6,378 组。
- 推荐任务唯一正确 SID：18,408。
- baseline 三字段全部唯一 SID：768,593。

压缩前缀树使用 6 个 NumPy 数组保存 768,593 条合法路径；它回答“当前已经生成 domain/a/b 后，下一步允许哪些 token”，不保存 prompt，也不预生成候选。

## 单组执行关系

执行关系：串行。改写对象是每组 8 个候选及其 token 概率。

输入示例：

```text
group_id = abc...
positive_sids = [video/1/2/3, video/1/2/4]
epoch = 1
G = 8
```

主链：

1. 当前 policy 按 8 个独立 SHA256 种子在线采样 8 条完整 completion；保留重复候选。每生成一个 token，都对“prompt + 当前已生成前缀”做一次 `use_cache=False` 的完整前向。
2. 每一步只在前缀树允许的 token 中做 FP32 softmax，并记录生成引擎的 token logp 作为数值诊断。
3. 若 8 条里没有任何正确 SID，则按 `SHA256(add_gt|42|epoch|group_id) < 0.5` 决定是否把第 8 条替换为一个正确 SID。
4. 对最终 8 条分别计算五档奖励：`1.0 / 0.20 / 0.05 / 0.01 / 0.0`。
5. 在组内用样本标准差 `correction=1` 做 Z-score；分母加 `1e-4`。
6. 参数更新前，当前 policy 立即用无 cache 整段 scorer 重算最终 8 条 completion 并冻结 live 候选的 `old_logp`；forced GT 的重算值会被丢弃，因此 forced GT 没有 `old_logp`。
7. 冻结 reference adapter 对完整 completion 做一次 `use_cache=False` 的 teacher forcing，计算 `reference_logp`。
8. 可训练 policy 用同一套无 cache 整段 scorer 计算 `new_logp`；反向传播前，live 候选的决策 mask 必须逐项相等，且 `max(abs(new_logp-old_logp)) <= 1e-5`，否则立即停止。
9. 通过一致性门禁后，计算 clipped surrogate 与 `beta=0.02` 的 reference KL。
10. 每个候选先按有多个合法选择的 token 求平均，再按 8 个候选求平均。

候选必须由当前 policy 在本组现场生成，禁止固定候选、回放缓冲或离线偏好对。PPO ratio 使用参数更新前冻结的整段 `old_logp` 和同一整段 scorer 计算的 `new_logp`，所以 live 候选必须满足 `new/old=1`；`max(abs(new_logp-old_logp)) <= 1e-5` 是反向传播前的硬门禁。逐前缀生成 logp 与整段 old 的差值单独记录为诊断，不进入 PPO ratio，因为 ragged FlashAttention 的 kernel 打包会造成浮点实现差异。

## 模型与优化器

- 同一个 base 模型挂两套同源 adapter。
- `default`：继续训练的 policy；只有它的 LoRA A/B 参数可训练。
- `reference`：冻结参考；每次切换 adapter 后重新强制 `requires_grad` 白名单。
- LoRA：继承 SFT 的 `r=32`、`alpha=32`、7 类 target module；运行时与保存配置的 dropout 都改为 0。
- AdamW：`lr=5e-6`、`weight_decay=0`、`betas=(0.9,0.999)`、`eps=1e-8`。
- 每 8 组更新一次；每轮最后 2 组单独更新；每轮 798 步，两轮 1,596 步。
- 前 48 步线性 warmup，之后 cosine。
- policy 的整段 `use_cache=False` loss forward 在 `train` 模式运行，实际启用 Transformers gradient checkpointing；reference 和在线采样在 `no_grad + eval` 下运行。dropout 已固定为 0，因此三条路径表示同一个 policy。真实显存由最长样本门禁测量，不根据配置名估算。

## 门禁与运行签名

- 最长提示词门禁依次测试统一 chunk `8/4/2/1`，PyTorch allocator 硬限制为 20 GiB；rollout 和带 Adam 状态的两次 loss 都通过才可选中。
- 512 组信号门禁必须产生有限 loss、非零梯度、非零参数变化和至少一组非零 advantage。
- 32 组耗时门禁把每组实测训练时间外推到 12,756 组；结果必须不超过 48 小时。
- full 入口同时读取 memory、signal、timing 三份报告。任一报告缺失、失败或运行签名不一致，正式训练都不会启动。
- 运行签名绑定冻结数据、base/SFT/tokenizer 实体、19 个训练源码文件，以及 Python、Torch、Transformers、PEFT、FlashAttention 等依赖版本。恢复点和最终验收使用同一签名。

## 一键执行

环境固定为：

```text
/home/lyc/miniconda3/envs/onereason_lora_sft
```

完整命令：

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/grpo/run_all_v3_1.sh
```

分阶段命令：

```bash
scripts/grpo/prepare_v3_1.sh
scripts/grpo/test_cpu_v3_1.sh
scripts/grpo/run_gates_v3_1.sh
scripts/grpo/run_full_v3_1.sh
scripts/grpo/run_probes_v3_1.sh
scripts/grpo/verify_v3_1.sh
```

`run_full_v3_1.sh` 默认读取 `recovery/latest.json` 恢复。训练在第一个窗口前先保存 step-0 恢复点，之后每 100 步和每轮末保存；恢复时先把 audit/progress 截断到游标再重放。最新恢复点损坏时直接失败，不会静默退回更旧权重。

## 输出

```text
artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch/
  epoch_001/adapter_model.safetensors
  epoch_001/adapter_config.json
  epoch_002/adapter_model.safetensors
  epoch_002/adapter_config.json
  recovery/
  train_audit.jsonl
  train_progress.jsonl
  run_summary.json

operation_logs/grpo/v3_1/
  config_check.json
  prepare_data.json
  build_trie.json
  tokenizer_gate.json
  memory_gate.json
  signal_gate.json
  timing_gate.json
  probes/
  final_verification.json
```

固定 probe 的 1,024 条标签来自既有 Explorer probe，但合法树按本方案只含 baseline SID。因此标签可达上限是：text-to-SID 256/512、recommendation 111/512。报告必须同时看 `reachable_targets` 和 `exact`，不能把不可达标签造成的上限当成训练退化。

## 2026-07-18 实际执行结果

正式运行签名：

```text
acd7f47e30862022e52df795f37bd9126a6812d08512c172876295ed9454057a
```

训练自然完成且严格 verifier 通过：

| 项目 | 实测结果 |
|---|---:|
| optimizer steps | 1,596 |
| 组级 audit 行 | 12,756 = 2 x 6,378 |
| 实际训练时间 | 79,665.77 秒（22.13 小时） |
| signal groups | 10,430 |
| forced-GT groups | 6,358 |
| live exact groups | 93 |
| loss mean | 0.0016136334 |
| reward mean | 0.0724215052 |
| 首个 LoRA 张量最大变化（训练日志诊断） | 0.0005372721 |
| 全部 392 个 LoRA 张量全局最大变化：epoch 1 / epoch 2 | 0.0020553945 / 0.0025138753 |
| 梯度范数范围 | 0.8020775 至 6.1031985 |
| 峰值 allocated / reserved | 5.733 / 9.375 GiB |
| OOM 重试 | 0 |
| old/current 最大 logp 差 | 0.0 |
| 生成路径/整段重算最大诊断差 | 0.623046875 |

最终 adapter：

| 轮次 | `adapter_model.safetensors` SHA256 | 大小 |
|---|---|---:|
| epoch 1 | `104b1ac5193525836413f50ac2ba996caaf5b7106e81c97cbd28d699fcd16115` | 80,792,456 B |
| epoch 2 | `42003b20033693db48e2168983a41916a6c34014a3d4db7ebb1d843ddde39e51` | 80,792,456 B |

固定 beam-16 probe：

| adapter | recommendation exact / valid | text-to-SID exact / valid |
|---|---:|---:|
| epoch 1 | 0/512 / 512/512 | 3/512 / 512/512 |
| epoch 2 | 0/512 / 512/512 | 4/512 / 512/512 |

两轮的 2,048 个预测全部属于 baseline 合法 SID 集。epoch 2 在本地 text-to-SID exact 上比 epoch 1 多 1 条，recommendation exact 都是 0；这个固定 probe 不是官方评测，不能据此宣称官方分数提升。

当前 Transformers 4.57.1 的 beam-search 重构不支持 `low_memory=True`。首次 probe 在 0 条预测时触发该兼容性错误；最终实现不再向 `generate` 传这个参数，同时保留 `num_beams=16`、KV cache 和合法前缀约束。失败现场保留在 `operation_logs/grpo/v3_1/probe_failures/low_memory_transformers_4_57_1/`，回归测试位于 `tests/grpo/test_probe.py`。

完整执行记录见 `operation_logs/GRPO_EXECUTION_SUMMARY_20260718.md`。
