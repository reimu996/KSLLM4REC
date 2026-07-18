# Frontier 数据集 OneReason-0.8B LoRA SFT 复现手册

## 1. 目标

why: 这次实验只把上一版本地 SFT 的 32,705 条 baseline 数据替换为 63,700 条 frontier 数据，以固定训练方法、隔离数据变化对结果的影响。旧 baseline 配置、脚本、日志和模型产物不得被覆盖。

执行关系: `准备数据 -> 全量分词预检 -> 配置检查 -> 四档 GPU 门禁 -> 完整 1 epoch -> Adapter 验证` 严格串行。任一步失败，后续步骤都不执行。

## 2. 固定输入

| 对象 | 固定值 |
| --- | --- |
| profile | `frontier_feedbackcore_listwise_invariant_v1` |
| source | `/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl` |
| source records | `63,700` |
| source SHA256 | `9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a` |
| base model | `/home/lyc/models/OneReason-0.8B-pretrain-competition` |
| config | `configs/sft/frontier_feedbackcore_listwise_invariant_v1.yaml` |
| artifact lock | `configs/sft/frontier_feedbackcore_listwise_invariant_v1.artifacts.lock.json` |

派生训练文件固定为 `artifacts/sft/data/frontier_feedbackcore_listwise_invariant_v1/train_alpaca.jsonl`，应有 63,700 条、269,870,172 bytes，SHA256 为 `6994bc8f13a676fb6ce3eb25bad891b9b43fdc6cabf5d24fe578329d77490b8d`。同目录的 `dataset_info.json` 也被锁定：数据集名称、字段映射和 `file_name` 必须精确匹配训练入口实际读取的内容。`prepare_data.sh` 转换完成后立即运行 `create-lock`，从正式派生文件和 `dataset_info.json` 重新读取路径、记录数、大小和 SHA256，并以原 baseline 锁中的模型与 LLaMA-Factory 身份为受控基础写出 frontier 锁；任一实测值不一致，后续预检会拒绝运行。

转换只把 `prompt/response/system` 映射到 `instruction/output/system`。禁止改写 `/think`、`/no_think`、`<think>...</think>`、SID、空白字符或记录顺序。

## 3. 固定训练配置

| 项目 | 值 |
| --- | --- |
| epoch | 1 |
| cutoff / packing | 16384 / true / neat packing true |
| microbatch / gradient accumulation | 1 / 8 |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| LoRA target | q/k/v/o/gate/up/down projection |
| optimizer | AdamW；LR `2e-4`；weight decay `0.001` |
| schedule | cosine；warmup ratio `0.03` |
| precision | BF16、TF32、FlashAttention 2；不量化 |
| custom loss | focal gamma 2.0；item weight 3.0；LM chunk 512 |
| seed / data seed | 42 / 42 |

`qwen3_nothink` 只指定对话格式，不删除训练数据中已有的思维文本。配置请求长度是 16,384，但 LLaMA-Factory 在 packing 预处理内部使用 16,383；因此分词预检只要发现一条完整样本或 response 超过 16,383 token，就终止流程。禁止静默截断、改变 cutoff 或裁剪 prompt；报告同时记录 `requested_cutoff_len=16384` 和 `internal_cutoff_len=16383`。

## 4. 一键复现

从项目根目录执行:

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/sft/frontier/run_all.sh
```

`run_all.sh` 的实际顺序:

```text
prepare_data.sh
run_preflight.sh
run_config_check.sh
run_gates.sh
run_full.sh
verify_full.sh
```

四档门禁 stage 分别是 `frontier_gate_00512`、`frontier_gate_02048`、`frontier_gate_08192`、`frontier_gate_16384`，每档只完成一个 optimizer step。完整训练 stage 是 `frontier_full_epoch_001`。

## 5. 独立产物

| 对象 | 路径 |
| --- | --- |
| config report | `artifacts/sft/frontier_feedbackcore_listwise_invariant_v1_config_check.json` |
| gates report | `artifacts/sft/frontier_feedbackcore_listwise_invariant_v1_gates_report.json` |
| gate outputs | `artifacts/sft/runs/frontier_feedbackcore_listwise_invariant_v1_gates/` |
| full output | `artifacts/sft/runs/frontier_feedbackcore_listwise_invariant_v1_epoch_001/` |
| logs | `operation_logs/sft/frontier_feedbackcore_listwise_invariant_v1/` |

最终目录必须包含 `adapter_model.safetensors`、`adapter_config.json`、`trainer_state.json` 和 `train_results.json`。完整训练中断时，full output 根目录的 `.sft_run_binding.json`、每个 checkpoint 内的 `run_binding.json` 以及来源 manifest 必须同时证明 profile、输出目录、输入锁、完整配置和实现指纹一致，才允许从该 full output 下最新且文件完整的 checkpoint 恢复；不得读取 baseline checkpoint 或手工放入的未绑定 checkpoint。

## 6. 完成判据

- 全量分词报告中 `requested_cutoff_len=16384`、`internal_cutoff_len=16383`，且 `target_over_cutoff=0`、`total_over_cutoff=0`。
- 四档门禁均为 passed，`fallback_count=0`，reserved 显存不超过 20.0 GiB。
- 最终 `epoch>=0.999`、global step 大于 0、train loss 有限。
- LoRA A/B 各 196 个、全部有限、LoRA B 的总 L2 norm 大于 0。
- base + adapter 短前向成功，logits 有限，adapter 相对 base 的 logits 变化非零。

该验证证明训练流程和 Adapter 文件正确，不代表官方评测分数提升。本流程不构造验证集、不运行 beam search、不做 ORPO/GRPO、不合并权重，也不上传 HuggingFace。
