# Frontier FeedbackCore OneReason-0.8B LoRA SFT 执行摘要

## 1. 目标与范围

why：验证把已有的 OneReason-0.8B LoRA SFT 流程切换到
`frontier_feedbackcore_listwise_invariant_v1` 后，能否在本机 4090 上稳定完成
一个 epoch，并留下可以逐项核对的训练产物。此次实验只更换数据集；训练方法、模型、
LoRA 配置、损失函数和数据处理约束保持既有 frontier SFT 方案不变。

本记录只证明训练流程和 Adapter 文件有效，不代表官方隐藏评测分数提升；没有运行
官方评测、beam search、ORPO/GRPO，也没有上传 Hugging Face。

## 2. 输入与数据处理

| 对象 | 实际值 |
| --- | --- |
| profile | `frontier_feedbackcore_listwise_invariant_v1` |
| 原始数据 | `/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl` |
| 原始记录数 | 63,700 |
| 原始文件大小 | 269,105,772 bytes |
| 原始 SHA256 | `9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a` |
| 派生训练文件 | `artifacts/sft/data/frontier_feedbackcore_listwise_invariant_v1/train_alpaca.jsonl` |
| 派生记录数 | 63,700 |
| 派生文件大小 | 269,870,172 bytes |
| 派生 SHA256 | `6994bc8f13a676fb6ce3eb25bad891b9b43fdc6cabf5d24fe578329d77490b8d` |
| `dataset_info.json` SHA256 | `a44fbbf06bf3af75e0a0495c0f5ea8fdd3ea66e977f22aaddf238e465adab46a` |

转换只把每条记录的 `system/prompt/response` 映射为
`system/instruction/output`，不改写 `/think`、`/no_think`、`<think>...</think>`、
SID、空白字符或记录顺序。`qwen3_nothink` 只负责对话模板，不是数据清洗规则。

全量 tokenizer 预检通过：请求 cutoff 为 16,384，packing 的内部 cutoff 为 16,383；
`target_over_cutoff=0`、`total_over_cutoff=0`。总 token 长度统计为 min/p50/p90/p95/p99/max
`66/376/1649/2280/5738/9752`；目标 token 数 2,698,648，item token 数 519,008，
item 比例 0.1923214884。预检报告见
`artifacts/sft/data/frontier_feedbackcore_listwise_invariant_v1/tokenizer_report.json`。

## 3. 固定训练配置

| 项目 | 值 |
| --- | --- |
| 基础模型 | `/home/lyc/models/OneReason-0.8B-pretrain-competition` |
| stage / finetuning | `sft` / `lora` |
| epoch | 1 |
| cutoff / packing | 16,384 / `true`（neat packing `true`） |
| micro-batch / 梯度累积 | 1 / 8 |
| 学习率 / scheduler | `2e-4` / cosine |
| warmup / weight decay | `0.03` / `0.001` |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| LoRA target | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| 精度与显存策略 | BF16、TF32、FlashAttention 2、gradient checkpointing |
| seed / data seed | 42 / 42 |
| 自定义损失 | focal gamma=2.0、item weight=3.0、chunk size=512 |

环境为 conda 环境 `onereason_lora_sft`，Python 3.11.15、torch 2.7.1+cu126、
Transformers 4.57.1。LLaMA-Factory 位于 `/home/lyc/REC_PROJECTS/LlamaFactory`，
commit `0b7aaf8f6a624bd89a01a155d4265ec861cbdf38`，tree
`02158cfe15d2019e3353901d61440b5a320993ab`，工作树干净。

## 4. 显存门禁

四档门禁均通过，`fallback_count=0`，峰值 reserved 均低于 20.0 GiB：

| stage | cutoff | max sequence | optimizer steps | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| `frontier_gate_00512` | 512 | 511 | 1 | 2.890625 GiB |
| `frontier_gate_02048` | 2,048 | 2,047 | 1 | 3.212890625 GiB |
| `frontier_gate_08192` | 8,192 | 8,191 | 1 | 3.29296875 GiB |
| `frontier_gate_16384` | 16,384 | 16,383 | 1 | 4.705078125 GiB |

门禁汇总：`artifacts/sft/frontier_feedbackcore_listwise_invariant_v1_gates_report.json`。

## 5. 完整训练

- run id：`frontier_full_epoch_001_20260719_024610_583505248`
- manifest：`operation_logs/sft/frontier_feedbackcore_listwise_invariant_v1/frontier_full_epoch_001_20260719_024610_583505248/manifest.json`
- 输出目录：`artifacts/sft/runs/frontier_feedbackcore_listwise_invariant_v1_epoch_001/`
- 训练时间：5,163.8005 秒（约 1:26:03.8）
- 有效 packed 样本：3,100
- micro-step / optimizer-step：3,100 / 388
- epoch：1.0
- train loss：18.412148632954075
- item loss / item ratio：1.4533700786530972 / 0.3466028251464464
- text loss：0.5109465127810836
- 数据集 token-weighted loss：0.9992712572216987
- peak allocated / reserved：4.549749374389648 / 4.708984375 GiB
- CE fallback：0

## 6. Adapter 验证

独立执行 `scripts/sft/frontier/verify_full.sh`，结果为 `status=passed`：

- `adapter_model.safetensors`：80,792,456 bytes，SHA256
  `eec36e9cfcea3ddba943224ccb5e1ad10eef049c9e931bb5ca00f5b26333f041`
- `adapter_config.json`：SHA256
  `7150b1f80233333b67a5beadaabb6c9389f289511d75790d5b04798611b5543f`
- LoRA tensor 总数 392，A 张量 196，B 张量 196；B 总 L2 norm 为 9.23282912832723（非零）
- trainer state：epoch=1.0、global_step=388、train_loss=18.412148632954075
- clean base + adapter 短前向成功，logits 有限；相对 clean base 的最大 logit 变化为 10.2421875
- 验证时 GPU 预留峰值 1.67578125 GiB

验证报告写入
`operation_logs/sft/frontier_feedbackcore_listwise_invariant_v1/frontier_verify_full_20260719_041243_941999233/verification.json`。

## 7. 复现入口与产物边界

在项目根目录执行以下命令可按同一顺序重建数据、预检、配置检查、门禁、完整训练和验证：

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/sft/frontier/run_all.sh
```

关键固定文件：

- 配置：`configs/sft/frontier_feedbackcore_listwise_invariant_v1.yaml`
- 输入锁：`configs/sft/frontier_feedbackcore_listwise_invariant_v1.artifacts.lock.json`
- 复现手册：`docs/ONEREASON_LORA_SFT_FRONTIER.md`
- 配置检查报告：`artifacts/sft/frontier_feedbackcore_listwise_invariant_v1_config_check.json`
- 门禁报告：`artifacts/sft/frontier_feedbackcore_listwise_invariant_v1_gates_report.json`

模型权重、checkpoint 和运行日志属于本地实验产物，保持在上述路径，不加入 Git；Git
只记录代码、配置、锁文件、脚本和本摘要。实现代码的基线提交为
`cf66d0e feat: add isolated frontier feedbackcore SFT pipeline`。

## 8. 风险与未验证项

1. 没有官方隐藏测试集结果，不能据此宣称推荐指标提升。
2. 没有执行 beam search 合法性评测，也没有与 baseline 做离线质量对比。
3. 没有合并 LoRA 权重或上传 HF；提交时应使用 full output 中的两个 Adapter 文件。
