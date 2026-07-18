# OneReason-0.8B LoRA GRPO V3.1 执行总结

## 目的与结论

本次执行要验证的目标是：从既有非 ORPO SFT adapter 出发，只使用 baseline 数据定义训练组和合法 SID 集，完成两轮在线 G=8 LoRA GRPO，并交付可恢复训练记录、两轮 adapter、固定约束 probe、严格 verifier 结果和可复现命令。

结论：2026-07-18 已全部完成。两轮训练自然结束，1,596 个优化步覆盖 12,756 个组级环境；三项 GPU 门禁、54 个 CPU 回归、tokenizer gate、两套固定 probe 和最终严格 verifier 均通过。没有 OOM 重试。

## 固定输入

| 对象 | 路径 | SHA256 |
|---|---|---|
| base 权重 | `/home/lyc/models/OneReason-0.8B-pretrain-competition/model.safetensors` | `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90` |
| 起始 SFT adapter | `artifacts/sft/runs/full_epoch_001/adapter_model.safetensors` | `77ac75d6ad558bdafc97cfb096f9608c59f4382c2884f17dceaf0fd943e0b710` |
| 起始 SFT adapter 配置 | `artifacts/sft/runs/full_epoch_001/adapter_config.json` | `5128629f9fe624a5807a8e8e728d3a8763936a01f8c9bc9df1ce4b9d00a9c5ec` |
| baseline `train.jsonl` | `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl` | `4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8` |
| 冻结训练组 | `artifacts/grpo/data/recommend_groups_v3_1/groups.jsonl` | `1c85ad933db46284d1417b7c2e83e31c2d379997a4e966439435f04ca75a243f` |
| 合法 SID 前缀树 manifest | `artifacts/grpo/catalog/baseline_all_sids_v3_1/manifest.json` | `737b2f69a568ba21d707f88b9fbd52a2d3c53ebb1b3aab6f2478e87530dbc066` |
| 固定 probe | `artifacts/orpo/data/all_tasks_32705/fixed_probe_1024.jsonl` | `8b8c64bcf9f4590dc006f900ded27d7008816d56ea1b3df6ccfd55a4e63b5fb1` |

冻结数据计数：baseline 32,705 行；推荐 18,651 行；合并后 6,378 个训练组；训练任务唯一正 SID 18,408；baseline 三字段合法 SID 768,593。

正式训练运行签名：

```text
acd7f47e30862022e52df795f37bd9126a6812d08512c172876295ed9454057a
```

该签名绑定配置、冻结 groups/trie、base/SFT/tokenizer 文件、训练源码及依赖版本。训练期间没有修改签名覆盖的对象。

## 环境

```text
Conda prefix: /home/lyc/miniconda3/envs/onereason_lora_sft
Python:       3.11.15
Torch:        2.7.1+cu126
Transformers: 4.57.1
PEFT:         0.18.1
FlashAttention: 2.7.4.post1
GPU:          NVIDIA GeForce RTX 4090, 23.99 GiB
```

训练配置：LoRA `r=32`、`alpha=32`、7 类投影、dropout 0；G=8；每 8 组更新；AdamW `lr=5e-6`；warmup 48 步；cosine；两轮共 1,596 步；BF16、TF32、FlashAttention 2、gradient checkpointing。

## 可复现命令

完整顺序：

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/grpo/prepare_v3_1.sh
scripts/grpo/test_cpu_v3_1.sh
scripts/grpo/run_gates_v3_1.sh
scripts/grpo/run_full_v3_1.sh
scripts/grpo/run_probes_v3_1.sh
scripts/grpo/verify_v3_1.sh
```

等价一键入口：

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/grpo/run_all_v3_1.sh
```

`run_full_v3_1.sh` 会从 `recovery/latest.json` 恢复。恢复点在 step 0、每 100 步、每轮末和最终 step 1,596 原子保存；恢复包含 adapter、optimizer、scheduler、随机状态、游标、manifest 和配置。

## 门禁结果

### tokenizer / 数据

- 训练组：6,378。
- prompt token 范围：72 至 2,929。
- cutoff：16,384。
- empty-think token：`[151667, 198, 151668, 198]`。
- 四类 SID 示例均可 encode/decode round trip。

### 显存门禁

- 最长 prompt：2,929 token。
- 选定 `rollout_chunk=8`、`loss_chunk=8`。
- rollout：20.291 秒；peak allocated 2.777 GiB；reserved 3.609 GiB。
- 含两套 scorer 与 Adam 状态的 loss：44.660 秒；peak allocated 5.655 GiB；reserved 6.338 GiB。
- allocator 硬上限：20 GiB。

### 512 组信号门禁

- passed：true。
- 训练时间：3,364.81 秒。
- signal groups：452；forced groups：243。
- 梯度范数：1.4833 至 3.8611。
- 首个可训练 LoRA 张量最大变化（门禁诊断值）：0.0001209136。
- peak allocated / reserved：5.434 / 7.236 GiB。

### 32 组耗时门禁

- passed：true。
- 实测：207.94 秒。
- 两轮投影：23.0253 小时，小于 48 小时上限。
- 首个可训练 LoRA 张量最大变化（门禁诊断值）：0.0000006268。

## 正式训练结果

| 项目 | 结果 |
|---|---:|
| optimizer steps | 1,596 |
| progress 行 | 1,596 |
| audit 行 | 12,756 |
| epoch 1 / epoch 2 组数 | 6,378 / 6,378 |
| 实际训练时间 | 79,665.77 秒（22.13 小时） |
| groups/s | 0.160119 |
| loss mean | 0.0016136334 |
| reward mean | 0.0724215052 |
| reward std mean | 0.1806242496 |
| signal groups | 10,430 |
| forced-GT groups | 6,358 |
| live exact groups | 93 |
| 首个可训练 LoRA 张量最大变化（训练日志诊断） | 0.0005372721 |
| 梯度范数最小 / 最大 | 0.8020775 / 6.1031985 |
| peak allocated / reserved | 5.733 / 9.375 GiB |
| OOM 重试 | 0 |

最终一步是第二轮最后 2 组，`window_groups=2`；没有丢弃尾批。最终游标：`epoch_index=3`、`global_step=1596`、`groups_completed=12756`、`next_group_offset=0`。

`parameter_max_change_this_invocation` 字段只比较训练器捕获的第一个可训练 LoRA 张量，不能当成全部参数的全局最大变化。对起始 SFT adapter 与两轮最终 adapter 的全部 392 个同名张量做 CPU 重算后：epoch 1 全局最大绝对变化为 `0.0020553944632411`，epoch 2 为 `0.0025138752534986`；两者都出现在 `base_model.model.model.layers.27.mlp.down_proj.lora_B.weight`。

审计重算结果：

- `maximum_on_policy_logp_delta = 0.0`，通过 `<=1e-5` 硬门禁。
- `maximum_generation_rescore_logp_delta = 0.623046875`，仅是生成路径与整段 scorer 的数值诊断，不进入 PPO ratio。

## Adapter 产物

| 轮次 | 文件 | 大小 | SHA256 |
|---|---|---:|---|
| epoch 1 | `artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch/epoch_001/adapter_model.safetensors` | 80,792,456 B | `104b1ac5193525836413f50ac2ba996caaf5b7106e81c97cbd28d699fcd16115` |
| epoch 1 | `epoch_001/adapter_config.json` | 1,082 B | `ab0a7166ee06530101d16f95b1736189cbe300d4b74138fd4110a15e72a9da60` |
| epoch 2 | `artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch/epoch_002/adapter_model.safetensors` | 80,792,456 B | `42003b20033693db48e2168983a41916a6c34014a3d4db7ebb1d843ddde39e51` |
| epoch 2 | `epoch_002/adapter_config.json` | 1,082 B | `ab0a7166ee06530101d16f95b1736189cbe300d4b74138fd4110a15e72a9da60` |

最终恢复点：`recovery/checkpoint-step-001596`；`recovery/latest.json` 指向 step 1,596。

## 固定 probe

执行规则：每轮 adapter 各 1,024 条；text-to-SID 512、recommendation 512；beam=16；合法前缀树约束；max completion 32。

| adapter | task | rows | reachable targets | valid predictions | exact |
|---|---|---:|---:|---:|---:|
| epoch 1 | recommendation | 512 | 111 | 512 | 0 |
| epoch 1 | text-to-SID | 512 | 256 | 512 | 3 |
| epoch 2 | recommendation | 512 | 111 | 512 | 0 |
| epoch 2 | text-to-SID | 512 | 256 | 512 | 4 |

probe 签名：

- epoch 1：`55ac36a2a043d11a2e1cca56365cd9d55fc4a79b999ef74b8ed9a50c7d6b3a38`。
- epoch 2：`e5f6b517077d560f37da15c8a026672aaeec64c5d833fb99c485dc03113c4aa4`。

两轮所有预测都合法。epoch 2 的 text-to-SID exact 比 epoch 1 多 1 条，但 recommendation exact 均为 0。固定 probe 不是官方评测，而且只有部分 GT 位于本方案的 baseline 合法树，因此不能据此宣称官方分数提升。

## Probe 兼容性故障与修复

首次运行 `scripts/grpo/run_probes_v3_1.sh` 时，epoch 1 在写出任何预测前失败：

```text
ValueError: `low_memory=True` is not supported after the beam search refactor.
```

根因：`probe.py` 显式向 Transformers 4.57.1 的 beam search 传入 `low_memory=True`；该版本源码会主动拒绝这个参数。PEFT 转发、beam=16、generation config 和约束 processor 不是根因。

修复：只移除 `low_memory=True`，保留 `num_beams=16`、`use_cache=True`、early stopping、renormalize logits 和合法前缀约束。新增 `tests/grpo/test_probe.py`，静态锁定真实 `.generate(...)` 调用不得再次传入该参数。

证据：

- 修复前回归测试失败并明确发现 `low_memory` keyword。
- 修复后该测试通过。
- 原始 GPU probe 重跑后完成 2,048/2,048 条预测。
- 失败现场保存在 `operation_logs/grpo/v3_1/probe_failures/low_memory_transformers_4_57_1/`，其中预测文件为 0 行。
- 无 `[DEBUG-*]`、`breakpoint()` 或 `pdb.set_trace` 残留。

probe 代码不属于正式训练运行签名，因此该修复不改变已完成训练和门禁的签名；probe 自身签名包含修复后的 `probe.py` SHA256。

## 最终验收

`scripts/grpo/verify_v3_1.sh` 返回 0，`operation_logs/grpo/v3_1/final_verification.json` 中 `passed=true`，`final_verification.stderr.log` 为 0 B。verifier 重新验证：

- 固定输入、groups、trie 与运行签名。
- 三项 GPU 门禁与统一 chunk。
- 1,596 步、两轮各 6,378 组、最终恢复游标。
- 12,756 条组级候选、reward、样本 Z-score、forced-GT 分支和 loss。
- old/current 硬差值与 generation-rescore 诊断分离。
- 两轮 adapter 文件大小与 SHA256。
- 两套 probe 的输入签名、1,024 行游标、SID 合法性和重新计算指标。

其余验证：

```text
CPU unittest: 54/54 passed
Ruff format:  35 files already formatted
Ruff check:   All checks passed
compileall:   passed
bash -n:      all scripts/grpo/*.sh passed
debug scan:   no matches
```

## 已知维护风险

以下两项不影响本次结果，但后续修改流程时需要补强：

- 当前最终 verifier 会核对 recovery 的运行签名和游标，但不会在 verifier 路径中重新读取 manifest 并校验所有恢复文件的 SHA256，也不会加载 `training_state.pt`。本次 `checkpoint-step-001596` 的 adapter、配置、训练状态和 cursor 已在提交前独立逐项与 manifest 核对一致，因此本次恢复产物有效；后续可为 verifier 增加完整恢复文件校验和损坏回归测试。
- 当前 probe 签名绑定 adapter、trie、固定 probe、beam/长度、`probe.py`、约束/模型/概率/提示/前缀树代码和依赖版本，但没有单独绑定 `contract.py`、`config.py` 或全部配置字段。本次两轮预测都是当前代码真实运行得到的完整 1,024 行，结果有效；以后修改 empty-think、cutoff 或模型加载行为时，应扩充 probe 签名并重新生成预测。

## 原始记录位置

原始记录和大产物不进入 Git，但保留在本机：

```text
operation_logs/grpo/v3_1/
artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch/
```

最关键的机器可读文件：

- `operation_logs/grpo/v3_1/memory_gate.json`
- `operation_logs/grpo/v3_1/signal_gate.json`
- `operation_logs/grpo/v3_1/timing_gate.json`
- `operation_logs/grpo/v3_1/full_train_result.json`
- `operation_logs/grpo/v3_1/probes/epoch_001/probe_report.json`
- `operation_logs/grpo/v3_1/probes/epoch_002/probe_report.json`
- `operation_logs/grpo/v3_1/final_verification.json`
