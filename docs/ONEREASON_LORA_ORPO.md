# OneReason-0.8B LoRA ORPO 复现手册

本手册用于在固定模型、固定比赛数据、固定 Explorer 数据、固定 Conda 环境和单张 RTX 4090 上，从干净 OneReason-0.8B 的第一个优化步开始执行两轮 LoRA ORPO。它不包含“先 SFT、后 ORPO”的阶段，也不加载已有 adapter。

## 1. 目标与边界

why：训练需要在同一次参数更新中既提高正确回答 `chosen` 的概率，又让 `chosen` 胜过同一输入下的错误回答 `rejected`，同时避免 `[2B,L,V]` 全量 logits 在 24 GiB 显存上溢出。

目标态：

- 原始模型、baseline 数据和 Explorer 数据只读。
- exactly `32,705` 条 baseline 回答全部成为 chosen，并各有一个可审计 rejected。
- 从干净 base 创建 rank-32 LoRA；连续执行 `2.0 epoch` ORPO。
- epoch 1、epoch 2 分别保存标准 PEFT adapter。
- 四档显存门、完整训练、双 checkpoint 加载和固定 probe 都有哈希及日志。
- 本地 probe 只作为提交门禁，不把本地结果称为比赛线上分数。

范围外：独立 CE/SFT 阶段、focal loss、已有 focal adapter、DPO reference model、SimPO、GRPO、QLoRA、全参数训练、自动提交比赛平台。

## 2. 固定输入

源文件清单和 SHA256 位于：

- `configs/orpo/generation.lock.json`：模型、baseline、Explorer SID/caption/probe 目录和 LLaMA-Factory。
- `configs/orpo/artifacts.lock.json`：完成偏好对生成后，进一步冻结全部训练输入。
- `configs/sft/environment.lock.txt`：Python `3.11.15` 环境的 123 个 Python 包。

关键输入：

| 对象 | 固定值 |
| --- | --- |
| base | `/home/lyc/models/OneReason-0.8B-pretrain-competition` |
| base weights SHA256 | `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90` |
| baseline | `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl` |
| baseline SHA256 | `4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8` |
| baseline 行数 | `32,705` |
| LLaMA-Factory | commit `0b7aaf8f6a624bd89a01a155d4265ec861cbdf38`；tree `02158cfe15d2019e3353901d61440b5a320993ab` |
| Conda 环境 | `/home/lyc/miniconda3/envs/onereason_lora_sft` |

所有阶段先重新计算锁中对象的大小和 SHA256。路径相同但内容变化、LLaMA-Factory tree 变化、Python 包增删或版本变化时，流程拒绝继续。

## 3. 训练对象和 loss

执行关系：chosen 与 rejected 在同一 microbatch 中并行前向；SFT 项与偏好项相加；只执行一次 backward；优化器只更新 LoRA 参数。

变量：

- `hidden_states`：base backbone 输出；shape `[2B,L,1024]`；前 `B` 行 chosen，后 `B` 行 rejected。
- `labels`：next-token 目标；shape `[2B,L]`；response 之外为 `-100`。
- `chosen_logp`：chosen response 的平均 token log-prob；shape `[B]`。
- `rejected_logp`：rejected response 的平均 token log-prob；shape `[B]`。
- `beta`：偏好项权重；标量，固定 `0.1`。
- `loss`：最终交给 LoRA optimizer 的标量。

伪代码：

```text
输入:
  chosen_logp   = chosen response 的平均 token log-prob
  rejected_logp = rejected response 的平均 token log-prob

步骤:
  chosen_log_odds   = chosen_logp   - log(1 - exp(chosen_logp))
  rejected_log_odds = rejected_logp - log(1 - exp(rejected_logp))
  log_odds = chosen_log_odds - rejected_log_odds
  sft_loss = -chosen_logp
  preference_loss = -log(sigmoid(log_odds))
  pair_loss = sft_loss + 0.1 * preference_loss
  loss = mean(pair_loss)

输出:
  loss -> backward -> 只更新 LoRA A/B
```

这是 ORPO 的 chosen-SFT 项与 odds-ratio 偏好项，不是 focal loss，也不是需要 reference model 的 DPO loss。稳定实现位于 `src/ksllm4rec_orpo/loss.py`。

## 4. 偏好对构造

所有 response 被规范成 direct no-think：保留原始 `</think>` 后最终答案，前缀统一为一个空 think。该转换会删除 teacher CoT，不是无损训练变换。

任务及 rejected 规则：

| 任务 | 行数 | rejected 规则 |
| --- | ---: | --- |
| recommendation | 18,651 | 同 domain，优先同 `s_a/s_b` 错 `s_c`，再退到同 `s_a` 或同 domain |
| text-to-SID | 5,197 | 与 recommendation 相同，并排除该文本的全部已知正确 SID |
| SID-to-text | 4,487 | 从其他 SID 取同层级 caption，禁止与 chosen 的已知 caption 相同 |
| user-interest | 2,792 | 另一用户的合法 JSON，优先要求 SID 集与当前 prompt 完全不相交 |
| CEval | 1,578 | 其余选项中由干净 base 选择概率最高的错误选项 |

recommendation 的原始 prompt 字符串有 `9,829` 组；去掉末尾 `/think`、`/no_think` 模式差异后，真正送入训练的规范化 prompt 有 `6,378` 组，共包含 `18,651` 个正例。rejected 必须排除规范化组的完整正例集合，不能只排除当前行 chosen。

当同一难度层有多个合法候选时，干净 base 在 chosen/rejected 首个分歧 token 处选择错误概率最高的候选。生成结果：

- `train.jsonl` SHA256：`ab96da904229473dfd1a1898013fef7692cb879be78e33f17dc187ca3cb97c94`
- `audit.jsonl` SHA256：`561295e08554e9c19899a5484f2e25c4a32832d8da4f0199a2e21a1c313b1c3e`
- `pair_manifest.json` SHA256：`5764ec5bbdb786310e577c46bd7bcbd767cffcc5325ddecf805f4086a299c9ba`
- `32,705` 个 pair_id 全部唯一。

第二次从空目录独立生成的 `train.jsonl`、`audit.jsonl` 和 `pair_manifest.json` 必须与第一次逐字节相同。

## 5. 分块 sequence log-prob

原生 ORPO 若同时物化 `[2,16384,176253]` 的 FP32 logits，仅该张量就约 `21.52 GiB`，还未计入模型、激活、梯度和优化器。

本实现先取得 `[2B,L,1024]` hidden states，只选 `labels != -100` 的 response 位置，然后每 `512` 个 token 计算一次冻结 LM Head：

```text
valid_hidden: [N,1024]
  -> 每 512 行计算 logits: [512,176253]
  -> 收集正确 token log-prob
  -> 按 response row 累加并除以 token 数
  -> 得到 [2B] sequence average log-prob
```

backward 时逐块重算 LM Head，以计算量换显存。LM Head 冻结，不计算其权重梯度。单元测试把分块结果与完整 logits 参考实现对齐，并覆盖极端 log-prob 下的有限 loss/gradient。

## 6. 固定配置

配置文件：`configs/orpo/onereason_lora_orpo.yaml`。

| 项目 | 值 |
| --- | --- |
| 训练起点 | 干净 base；`adapter_name_or_path=null` |
| ORPO | `beta=0.1`；无 reference model |
| LoRA | rank `32`；alpha `32`；dropout `0`；q/k/v/o/gate/up/down |
| cutoff / packing | `16384` / `false` |
| microbatch / accumulation | `1` / `8` |
| epoch | `2.0` |
| optimizer | AdamW；LR `1e-4`；weight decay `0.001`；max grad norm `1.0` |
| schedule | cosine；warmup ratio `0.03` |
| precision | BF16；TF32；FlashAttention 2 |
| memory | gradient checkpointing；LM chunk `512` |
| seed | model/data 都为 `42` |
| checkpoint | 每个 epoch 保存；最多 2 个 |

预检测得真实最长 pair 为 `9,752` token；`cutoff_len=16384` 下 source、chosen、rejected 的裁剪数均为 0。

## 7. 执行顺序

所有命令从项目根目录运行：

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC

scripts/orpo/freeze_sources.sh
scripts/orpo/prepare_pairs.sh
scripts/orpo/run_config_check.sh
scripts/orpo/run_gates.sh
scripts/orpo/run_full.sh
scripts/orpo/run_probes.sh
```

也可以在全新、没有同名输出目录的产物根目录中执行：

```bash
scripts/orpo/run_all.sh
```

阶段关系：

```text
冻结输入
  -> 生成候选
  -> base hard-negative mining
  -> 数据/长度/probe 预检
  -> 512/2048/8192/16384 四档真实优化步门禁
  -> 两轮完整 ORPO
  -> 双 checkpoint 独立加载验证
  -> base/epoch1/epoch2 固定 probe
```

`run_full.sh` 会先检查四档门禁，再训练或从最新完整 checkpoint 恢复，成功后自动运行 `verify_full.sh`。`run_probes.sh` 为防止旧结果被覆盖，要求 `artifacts/orpo/probes/base`、`epoch_001`、`epoch_002` 不存在；复跑 probe 前必须先把旧结果归档到其他路径。

## 8. 输出与判定

大产物保留在本机，不进入 Git：

- 偏好数据：`artifacts/orpo/data/all_tasks_32705`
- 显存门：`artifacts/orpo/runs/gates`、`artifacts/orpo/gates_report.json`
- 两轮训练：`artifacts/orpo/runs/orpo_from_base_epoch2`
- 固定 probe：`artifacts/orpo/probes`
- 原始日志/manifest：`operation_logs/orpo`

每个 epoch checkpoint 必须同时包含非空的 `adapter_model.safetensors`、`adapter_config.json`、`optimizer.pt`、`scheduler.pt`、`trainer_state.json` 和 `rng_state.pth`。验证还要求：

- LoRA A/B 张量数量符合固定 target；LoRA B 非零且全部有限。
- base + adapter 可独立加载；前向 logits 全部有限且不同于禁用 adapter。
- manifest 证明 `reference_model_used=false`、训练从 clean base 开始。
- 实现指纹、输入锁、环境锁和产物 SHA256 全部匹配。
- 峰值 reserved 显存不超过 `20.0 GiB`。

固定 probe 是 512 条未进入训练的 Explorer recommendation 和 512 条未进入训练的 Explorer text-to-SID。当前实现使用 greedy generation、`max_new_tokens=64`，报告第一个完整 SID 的合法率和 exact-match。probe 与平台隐藏测试集不是同一分布；它能否决明显退化的 adapter，不能证明线上一定提分。

实际执行结果和是否建议提交见 `operation_logs/ORPO_EXECUTION_SUMMARY_20260715.md`。
