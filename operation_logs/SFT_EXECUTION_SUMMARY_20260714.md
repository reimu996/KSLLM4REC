# OneReason-0.8B LoRA SFT 执行总结（2026-07-14）

## 1. 目标与结论

why: 本次执行要证明固定的 OneReason-0.8B、固定比赛数据、固定 focal + item 加权损失，可以在单张 RTX 4090 上按可复现合同完成 1 epoch LoRA SFT，并产出可独立加载、确实改变模型输出的 adapter。

结论: 执行成功。四档 GPU 显存门全部通过，完整训练完成 `325/325` 个优化步和 `1.0 epoch`，自定义 loss 的 CE 安全回退次数为 `0`，训练峰值 reserved 显存为 `4.7109375 GiB`。最终 adapter 通过输入、环境、配置、文件哈希、张量结构、非零 LoRA B 和实际前向增量验证。

本结论只证明训练流程和产物正确，不证明比赛指标提升。本次没有构造或运行本地验证集，也没有提交线上评测。

## 2. 固定输入与受控实现

| 对象 | 值 |
| --- | --- |
| 项目训练提交 | `5f52be414cad7d39aade02583858d08f3c27bd6f` |
| 实现指纹 | `11e29d023eb19d851d3d212f492951d43ced96da21bed7dae0a82a889d734091` |
| base model | `/home/lyc/models/OneReason-0.8B-pretrain-competition` |
| base weights SHA256 | `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90` |
| source dataset | `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl` |
| source dataset SHA256 | `4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8` |
| source records | `32,705` |
| derived dataset SHA256 | `313007f231b464a893314a79f572437c5955994d57c99ef47d9a0d2a7b2bf4fa` |
| LLaMA-Factory commit | `0b7aaf8f6a624bd89a01a155d4265ec861cbdf38` |
| LLaMA-Factory Git tree | `02158cfe15d2019e3353901d61440b5a320993ab` |
| Python | `3.11.15` |
| Conda environment | `/home/lyc/miniconda3/envs/onereason_lora_sft` |
| Python package lock | `123` 个包，逐行校验通过 |

数据预检结果:

- 最长完整样本为 `10,553` token，最长 target 为 `1,527` token。
- `cutoff_len=16384` 下，完整样本和 target 的超限数量均为 `0`。
- 因此从 `32768` 改为 `16384` 没有裁剪或丢弃训练样本。
- 梯度累积从 `4` 改为 `8`，每次优化更新的序列位置预算仍为 `131,072`；packing 边界可能变化，因此不声称与旧方案逐 bit 等价。
- 全量数据包含 `486,963` 个 item target token、`6,195,584` 个 target token；全部 `32,705` 条 response 都保留原有 think 文本和 think token。

## 3. 实际训练配置

| 项目 | 实际值 |
| --- | --- |
| cutoff / packing | `16384` / `packing=true` / `neat_packing=true` |
| microbatch / accumulation | `1` / `8` |
| epoch / seed | `1` / `42` |
| precision | BF16，TF32，FlashAttention 2，不量化 |
| LoRA | rank `32`，alpha `32`，dropout `0.05` |
| LoRA target | q/k/v/o/gate/up/down projection |
| optimizer | AdamW，LR `2e-4`，weight decay `0.001` |
| schedule | cosine，warmup ratio `0.03` |
| custom loss | focal gamma `2.0`，item weight `3.0`，LM chunk `512` |
| checkpoint | 每 `256` 步保存，最多保留 `2` 个 |
| GPU 启动门槛 | free memory `>= 20.5 GiB` |
| GPU reserved 上限 | `<= 20.0 GiB` |

训练前验证结果:

- `33` 个单元测试通过。
- Ruff 静态检查通过。
- Shell `bash -n` 语法检查通过。
- `git diff --check` 通过。
- 数据预检、配置合同、输入哈希、环境锁和 LLaMA-Factory 工作区检查全部通过。

## 4. GPU 显存门

四档门禁都执行 `8` 个 microbatch，即 `1` 个优化步。门禁报告: `artifacts/sft/gates_report.json`。

| 门禁 | cutoff | 实际最大序列长度 | 峰值 reserved 显存 | fallback | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gate_00512` | 512 | 511 | `2.87109375 GiB` | 0 | passed |
| `gate_02048` | 2048 | 2047 | `3.193359375 GiB` | 0 | passed |
| `gate_08192` | 8192 | 8191 | `3.5859375 GiB` | 0 | passed |
| `gate_16384` | 16384 | 16383 | `4.701171875 GiB` | 0 | passed |

所有门禁使用同一实现指纹 `11e29d023eb19d851d3d212f492951d43ced96da21bed7dae0a82a889d734091`，不存在用旧实现产物通过当前门禁的情况。

## 5. 完整训练结果

| 指标 | 结果 |
| --- | ---: |
| 运行编号 | `full_epoch_001_20260714_102648_304003647` |
| 开始时间 | `2026-07-14 10:26:48 +08:00` |
| 优化步 / microbatch | `325 / 2,593` |
| 完成 epoch | `1.0` |
| 最长实际序列 | `16,383` |
| Trainer 报告的 train loss | `12.0118896484375` |
| 训练时长 | `4,914.2052` 秒（约 `1:21:54`） |
| 样本吞吐 | `0.528 samples/s` |
| 优化步吞吐 | `0.066 steps/s` |
| 峰值 allocated 显存 | `4.54974889755249 GiB` |
| 峰值 reserved 显存 | `4.7109375 GiB` |
| CE fallback | `0` |

恢复检查点已写入:

- `artifacts/sft/runs/full_epoch_001/checkpoint-256`
- `artifacts/sft/runs/full_epoch_001/checkpoint-325`

全量 manifest:

`operation_logs/sft/full_epoch_001_20260714_102648_304003647/manifest.json`

## 6. Adapter 验证

最终输出目录:

`artifacts/sft/runs/full_epoch_001`

| 验证项 | 结果 |
| --- | --- |
| adapter 文件 | `adapter_model.safetensors`，`80,792,456` bytes |
| adapter SHA256 | `77ac75d6ad558bdafc97cfb096f9608c59f4382c2884f17dceaf0fd943e0b710` |
| LoRA A / B 张量 | `196 / 196` |
| LoRA B 总 L2 norm | `10.933871239810095`，大于 0 |
| base 与 adapter 最大 logits 差 | `9.78125`，大于 0 |
| 验证前 GPU free memory | `22.44921875 GiB` |
| 验证峰值 reserved 显存 | `1.66796875 GiB` |
| 独立加载前向 | 通过，logits 全部有限 |
| 总状态 | `passed` |

验证生成的短文本为 `"<think>\n</think>\n该用户最近喜欢"`。该文本只用于证明 base + adapter 能实际加载并执行前向，不是推荐质量评价。

验证报告:

`operation_logs/sft/verify_full_20260714_114858_059687694/verification.json`

## 7. 复现命令

在固定输入、固定 LLaMA-Factory tree 和固定 Conda 环境仍存在的前提下，从项目根目录执行:

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/sft/run_tests.sh
scripts/sft/run_preflight.sh
scripts/sft/run_config_check.sh
scripts/sft/run_gates.sh
scripts/sft/run_full.sh
```

`run_full.sh` 会先复核四档门禁，再训练或从最新完整 checkpoint 恢复，训练成功后自动执行 `verify_full.sh`。

## 8. 未执行事项与风险边界

- 未构造本地 train/validation 划分，无法报告泛化指标或离线推荐指标。
- 未与标准 CE、不同 focal gamma、不同 item weight 或 think 开关做受控对照实验。
- 未合并 LoRA 到 base model；推理时需要同时加载固定 base model 和本 adapter。
- 未 push 到远端，也未提交比赛平台。
- 大模型产物、checkpoint 和详细日志按设计不进入 Git；本摘要只记录其路径、大小和 SHA256。本机文件被删除后，需要按复现命令重新生成。

一句话结论: 16K cutoff 没有裁剪或丢弃任何原始训练样本，单张 RTX 4090 以约 `4.71 GiB` 峰值 reserved 显存完成了固定方案的 1 epoch LoRA SFT，最终 adapter 已通过非零权重和实际前向增量验证。
