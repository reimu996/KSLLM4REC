# OneReason-0.8B LoRA SFT 复现手册

本手册面向已经具备固定模型、固定数据、Conda、CUDA 驱动和联网条件的预配置机器。它保证在这些前置条件成立时可复用、可复现；不承诺从一台空白机器自动获取比赛私有模型或数据。

## 1. 目标

why: 这套流程要在单张 RTX 4090 上，用固定的 OneReason-0.8B、固定的 32,705 条比赛 SFT 数据和固定的 focal + item 加权损失完成 1 epoch LoRA 训练，同时避免长序列输出层产生不可承受的显存占用。

目标态:

- 原始模型和原始数据只读，不做原地修改。
- 新建独立 Conda 环境 `onereason_lora_sft`，不修改已有环境。
- 每个输入、依赖、配置、阶段和输出都有可验证记录。
- 训练保留 response 中的完整 `<think>...</think>`；system、user prompt 不计算 loss。
- 先通过四档长度显存门，再运行完整 1 epoch；中断后从最近 checkpoint 恢复。
- 输出标准 PEFT LoRA adapter，并实际加载做前向验证。

## 2. 固定输入

输入锁文件是 `configs/sft/artifacts.lock.json`。预检和每次训练启动都会重新计算 SHA256；路径相同但内容变化时直接拒绝运行。

运行前置条件:

- Linux 或 WSL2 x86_64；当前实测 NVIDIA 驱动 `595.95`，且驱动必须支持 CUDA 12.6 用户态组件。
- 单张 RTX 4090；GPU 支持 BF16 和 FlashAttention 2；启动训练时可用显存至少 20.5 GiB。
- Miniconda 固定安装在 `/home/lyc/miniconda3`；新环境固定使用 Python `3.11.15`。项目磁盘额外可用空间至少 30 GiB，用于新环境、wheel、派生数据、checkpoint 和日志。
- 下表中的模型、数据和 LLaMA-Factory 仓库已经存在；模型与数据不会由脚本联网下载。
- 首次构建环境时可以访问 PyPI、PyTorch 官方 wheel 索引和 FlashAttention 的 GitHub Release；已有校验通过的本地缓存时可以离线复用。

| 对象 | 固定路径 | 关键值 |
| --- | --- | --- |
| base model | `/home/lyc/models/OneReason-0.8B-pretrain-competition` | Qwen3，28 层，hidden size 1024，vocab 176253 |
| model weights | `model.safetensors` | SHA256 `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90` |
| source data | `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl` | 32,705 行；SHA256 `4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8` |
| LLaMA-Factory | `/home/lyc/REC_PROJECTS/LlamaFactory` | commit `0b7aaf8f6a624bd89a01a155d4265ec861cbdf38`；Git tree `02158cfe15d2019e3353901d61440b5a320993ab` |

派生数据位于 `artifacts/sft/data/hf_baseline_091/train_alpaca.jsonl`。转换只把外层结构映射为 LLaMA-Factory 的 Alpaca 字段，`system/prompt/response` 三个字符串逐字节保持不变；转换前后内容摘要必须一致。

LLaMA-Factory 不只检查 HEAD：预检、配置检查、训练和最终验证都要求 Git tree 与锁文件一致、工作区无 tracked/untracked 改动，并要求 Python 实际从锁定目录的 `src/llamafactory` 导入。`LLAMAFACTORY_ROOT` 不允许通过环境变量改写。

## 3. Loss 和显存

### 3.1 数学对象

- `H`: backbone 最后一层隐藏状态；shape `[B, L, 1024]`；来源是 OneReason backbone；用于计算目标 token 的 logits。
- `W`: 冻结的语言模型输出头权重；shape `[176253, 1024]`；来源是 base model 的 `lm_head.weight`。
- `labels`: 目标 token id；shape `[B, L]`；assistant response 和结尾 token 有具体 id，system、user、padding 为 `-100`。
- `item_mask`: 平台定义的 item 目标位置；shape `[B, L]`；规则是目标 id 是否属于 `tokenizer.get_added_vocab()`。
- `CE`: 正确 token 的逐 token 交叉熵；shape `[N]`，`N` 是本 microbatch 有效目标 token 数。

逐 token loss 与比赛示例完全相同:

```text
p = exp(-CE)
focal = (1 - p) ** 2 * CE
weight = 3.0 if 当前目标是 item token else 1.0
loss = mean(focal * weight)
```

本模型 `tokenizer.vocab_size=151643`、`len(tokenizer)=176253`，added vocabulary 是连续 id `[151643, 176252]`，共 24,610 个。它包含 24,576 个 SID token、8 个 domain begin/end token和 26 个 Qwen 特殊 token。本地实现按全部 24,610 个 id 构造 mask；这是对比赛 `ctx["item_mask"]` 的逐字规则复现，不额外按 token 名做主观筛选。

next-token 对齐规则如下。`H[:, t]` 预测 `labels[:, t+1]`，因此 item 判断也必须取目标位置 `t+1`，不能拿输入位置 `t` 的 token 类型代替:

```text
输入:
  H: [B, L, 1024]
  labels: [B, L]
  item_mask: [B, L]

步骤:
  shift_hidden = H[:, :-1, :]       # [B, L-1, 1024]
  shift_labels = labels[:, 1:]      # [B, L-1]
  shift_item = item_mask[:, 1:]     # [B, L-1]
  valid = shift_labels != -100
  H_valid = shift_hidden[valid]     # [N, 1024]
  y_valid = shift_labels[valid]     # [N]
  item_valid = shift_item[valid]    # [N]

输出:
  每个 H_valid[i] 只和同下标 y_valid[i]、item_valid[i] 计算 loss
```

### 3.2 为什么不能直接把完整 logits 转 FP32

在批准的 `B=1, L=16384, V=176253` 时:

- 完整 BF16 logits 约 5.38 GiB。
- 完整 FP32 logits 约 10.76 GiB。

这还没有计入模型、激活、梯度和优化器，所以即使 cutoff 已降为 16384，也不能直接执行示例中的完整 `shift_logits.float()`。

### 3.3 等价的分块实现

执行关系: token 分块串行；每个块的 loss 求和；最终除以全部有效 token 数。改写对象是输出头临时 logits，不改训练样本、不改 labels、不改 focal 公式。

```text
输入:
  H_valid = 只取 labels != -100 对应的隐藏状态
  chunk_size = 512

forward:
  通过 DDP 包装模型执行一次 forward，只让 Qwen3 生成最后 1 个位置的占位 logits
  用 backbone forward hook 取得 H，保证不绕过 DDP reducer
  对 H_valid 每 512 个 token 一块:
    logits_chunk = H_chunk @ W.T
    logits_chunk 转 FP32
    计算该块 CE、focal、item weight 和 loss_sum
    保存 CE 指标，丢弃 logits_chunk
  loss = 所有块 loss_sum / N

backward:
  对每个 H_chunk 重新计算 logits_chunk 和同一份 loss
  得到该块对 H_chunk 的梯度
  拼回 H_valid 的梯度，交给 backbone 继续反向传播

输出:
  与完整 logits 实现数值等价的标量 loss
```

`chunk_size=512` 时，一个块的 BF16 logits 约 0.168 GiB，FP32 logits 约 0.336 GiB。实现用自定义 autograd 在 backward 重算输出头，以计算量换显存；单元测试同时对齐标量 loss、逐 token CE 和隐藏状态梯度。CPU 参考实现只使用绝对误差验收，不使用相对误差: 标量 loss 不超过 `1e-6`，逐 token CE 不超过 `2e-6`，隐藏状态梯度不超过 `1e-5`。CE 的 `2e-6` 来自完整序列 GEMM 与有效 token 分块 GEMM 的浮点归约顺序差异，不是公式差异。

## 4. 固定训练配置

配置文件: `configs/sft/onereason_lora_focal_item.yaml`。

| 项目 | 值 |
| --- | --- |
| 模板 | `qwen3_nothink`；只控制模板格式，不删除数据中已有 think 内容 |
| cutoff / packing | 16384；`packing=true`；`neat_packing=true` |
| LoRA target | q/k/v/o/gate/up/down projection |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| batch / accumulation | 1 / 8，等效 batch 8 个 packed sequence |
| precision | BF16；TF32 开启；不量化 |
| checkpointing / attention | gradient checkpointing；FlashAttention 2 |
| optimizer | AdamW；LR `2e-4`；weight decay `0.001`；max grad norm `1.0` |
| schedule | cosine；warmup ratio `0.03` |
| epoch / seed | 1 epoch；seed 和 data seed 均为 42 |
| custom loss | gamma 2.0；item weight 3.0；LM chunk 512 |

不使用 QLoRA，不允许自动降低 cutoff，不允许把 focal 非有限回退当作成功。代码保留 CE 安全回退用于避免进程直接崩溃，但任一回退都会令阶段门控和最终验收失败。

`src/ksllm4rec_sft/contract.py` 是批准配置的可执行合同。配置检查、每个门禁和全量训练都会逐字段验证模型、数据、模板、LoRA、优化器、学习率、batch、精度、packing、cutoff、随机种子及 custom loss；修改 YAML 并重新生成实现指纹不能绕过合同。训练 CLI 不提供 chunk size 覆盖入口，四档门禁固定使用 512。

packing 只拼接样本，不允许不同样本互相看见。`neat_packing=true` 会为每条被拼接样本重置 position id，并让 LLaMA-Factory 派生 `block_diag_attn=true`；system 和 user 对应 label 仍为 `-100`。LLaMA-Factory 内部把原始 `cutoff_len=16384` 转成数据处理长度 16383，再补结尾 token 得到 16384 长度的 packed sequence；配置预检必须同时验证这两个值。

全量预检测得最长完整样本为 10,553 token，因此 16,384 cutoff 不会裁剪或丢弃任何样本。梯度累积从 4 增至 8，使每次优化更新的序列位置预算保持为 `1 * 16384 * 8 = 131072`，与旧方案的 `1 * 32768 * 4` 相同；packing 边界仍会变化，所以不声称两个方案逐 bit 等价。

## 5. 首次准备

所有命令从项目根目录执行:

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/sft/bootstrap_env.sh
scripts/sft/prepare_data.sh
scripts/sft/run_tests.sh
scripts/sft/run_preflight.sh
scripts/sft/run_config_check.sh
```

`bootstrap_env.sh` 的大 wheel 获取顺序:

1. 读取 PyPI 或 PyTorch 官方索引中的文件名、字节数和发布 SHA256。
2. 先在只读 pip HTTP 缓存中按字节数查找。
3. 缓存文件 SHA256 匹配才复制到项目 wheelhouse。
4. 没有匹配缓存时才联网断点续传。
5. wheel 安装后执行 `pip check`、导出完整 `pip freeze`、Conda explicit spec 和 wheelhouse SHA256。

`configs/sft/environment.lock.txt` 是完整 `pip freeze` 锁，不是仅列顶层包的需求文件。配置预检和每次训练启动都会先验证 Python 恰为 `3.11.15`，再执行 `pip freeze` 并逐行比较；Python patch、缺包、多包或版本漂移都会拒绝运行。`configs/sft/constraints.txt` 固定求解输入；每次 `operation_logs/sft/bootstrap_*` 保存 Conda explicit spec、wheel SHA256、`pip inspect` 和安装报告，便于审计实际安装物。

## 6. 显存门控和全量运行

GPU 启动条件固定为可用显存至少 20.5 GiB。每个阶段完成一个优化步，也就是 8 个 microbatch；峰值 reserved 显存必须不超过 20.0 GiB。

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/sft/run_gates.sh
scripts/sft/run_full.sh
```

`run_gates.sh` 串行验证 cutoff 512、2048、8192、16384。只有四个 manifest 都满足以下条件，`run_full.sh` 才启动:

- `status=passed`
- `optimizer_steps >= 1`
- `fallback_count = 0`
- `max_sequence_length >= cutoff * 0.95`，证明阶段实际跑到了对应长度，而不是只有配置名正确
- manifest 中的实现指纹等于当前锁文件、训练配置、全部 `scripts/sft` 脚本和 `src/ksllm4rec_sft` 源码的组合 SHA256
- `peak_memory_reserved_gib <= 20.0`

全量输出固定在 `artifacts/sft/runs/full_epoch_001`。`save_steps=256`、最多保留 2 个 checkpoint。项目本地 resolver 在训练参数解析前扫描 `checkpoint-<step>`：空目录从 base model 开始；否则只选择 step 最大者，并要求 `adapter_config.json`、`adapter_model.safetensors`、`optimizer.pt`、`scheduler.pt`、`trainer_state.json`、`rng_state.pth` 全部非空，且目录 step 等于 `trainer_state.global_step`。最新 checkpoint 不完整时立即停止，不回退到旧 checkpoint，也不静默从头训练。没有 checkpoint 的非空输出目录同样拒绝运行。

## 7. 输出与验收

每次调用在 `operation_logs/sft/<run_id>/` 生成控制台日志、shell 退出状态和 JSON manifest。大日志、wheel 和 checkpoint 被 `.gitignore` 排除；代码、配置、锁文件、手册和最终轻量摘要进入 Git。

`scripts/sft/run_full.sh` 在训练进程成功后自动调用完整验证。也可以单独重跑:

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/sft/verify_full.sh
```

命令退出码为 0 且验证报告 `status=passed` 才算完成。验证内容:

- `adapter_config.json` 固定为 rank 32、alpha 32、dropout 0.05 和 7 个目标模块。
- `adapter_model.safetensors` 含 196 个 LoRA A 张量和 196 个 LoRA B 张量，全部有限。
- LoRA B 的总 L2 norm 大于 0，证明它已离开全零初始化。
- `trainer_state.json` 的 epoch 至少 0.999，global step 大于 0，train loss 有限。
- 全量 manifest 的 focal 回退次数为 0，峰值 reserved 显存不超过 20.0 GiB。
- 全量 manifest 的实现指纹必须等于当前受控实现，避免用旧代码产物冒充当前结果。
- manifest 的 `resolved_config.output_dir` 必须等于被验证目录；`adapter_config.json`、`adapter_model.safetensors`、`trainer_state.json`、`train_results.json` 的路径、大小和 SHA256 必须逐项匹配训练成功时写入同一 manifest 的快照。
- 实际加载 base + adapter，短前向 logits 全部有限，且 adapter logits 与禁用 adapter 时不相同。

最终 adapter 是比赛训练产物；基础模型仍由固定路径单独提供，没有执行权重合并，也没有 push 到远端。
