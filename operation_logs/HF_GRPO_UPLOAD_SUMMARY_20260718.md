# OneReason-0.8B GRPO LoRA Hugging Face 上传记录（2026-07-18）

## 目标与结果

why：把 GRPO epoch 1、epoch 2 的 LoRA adapter 发布为两个公开 Hugging Face 模型仓库，让比赛平台能够通过 URL 下载；同时确保远端文件与本地正式训练产物逐字节一致。

结果：两个公开仓库均已创建、上传并通过匿名下载校验。

| Epoch | 公开仓库 | HF commit |
|---:|---|---|
| 1 | <https://huggingface.co/reimu996/OneReason-0.8B-GRPO-Epoch1> | `aa802f493eca16a4d1d38fc4713476abba00b79c` |
| 2 | <https://huggingface.co/reimu996/OneReason-0.8B-GRPO-Epoch2> | `fe7e62ef3bd3a73116c57300a2553bbf2b1cdf0f` |

HF 身份为 `reimu996`。远端 preflight 时两个仓库均不存在；上传器未覆盖或删除任何已有仓库。

## 固定输入

| Epoch | 本地目录 | adapter 权重 bytes | adapter 权重 SHA256 |
|---:|---|---:|---|
| 1 | `artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch/epoch_001` | 80,792,456 | `104b1ac5193525836413f50ac2ba996caaf5b7106e81c97cbd28d699fcd16115` |
| 2 | `artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch/epoch_002` | 80,792,456 | `42003b20033693db48e2168983a41916a6c34014a3d4db7ebb1d843ddde39e51` |

两份 `adapter_config.json` 均为 1,082 bytes，SHA256 均为 `ab0a7166ee06530101d16f95b1736189cbe300d4b74138fd4110a15e72a9da60`。上传 staging 从正式训练目录逐字节复制配置，没有修改 `base_model_name_or_path`。

## 远端文件与校验

每个仓库主动上传且仅上传：

```text
README.md
adapter_config.json
adapter_model.safetensors
```

Hugging Face 自动生成 `.gitattributes`。最终远端文件集合有且仅有：

```text
.gitattributes
README.md
adapter_config.json
adapter_model.safetensors
```

两个仓库均满足 `private=false`。上传完成后，工具固定各自 HF commit，以匿名身份重新下载三个业务文件并比较大小和 SHA256；六个下载文件全部与 staging 一致，两个远端 adapter 权重也与正式训练产物一致。

## 可复用操作

固定配置：`configs/grpo/hf_upload.json`。

上传工具：`scripts/orpo/upload_hf_adapters.py`。该工具已通用化为配置驱动，同时保持既有 ORPO 默认路径和 README 输出逐字节不变。

使用环境：`/home/lyc/miniconda3/envs/MiniOneRec_kmeans`；执行前要求 Hugging Face CLI 登录身份与配置中的 namespace 一致。

只做本地 staging 和哈希门禁，不写远端：

```bash
/home/lyc/miniconda3/envs/MiniOneRec_kmeans/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --config configs/grpo/hf_upload.json \
  --run-id <unique_run_id>
```

创建两个公开仓库、上传并做匿名下载校验：

```bash
/home/lyc/miniconda3/envs/MiniOneRec_kmeans/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --config configs/grpo/hf_upload.json \
  --execute \
  --run-id <unique_run_id>
```

安全边界：任何远端写入前验证 HF 身份、本地文件 size/SHA256 和两个目标仓库均不存在；目标仓库已存在时立即退出，不覆盖。staging 只允许三个业务文件；上传后要求远端公开、revision 一致、文件白名单一致并通过匿名下载。

## 本次执行记录

实际命令：

```bash
/home/lyc/miniconda3/envs/MiniOneRec_kmeans/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --config configs/grpo/hf_upload.json \
  --execute \
  --run-id hf_upload_20260718_grpo_epoch1_epoch2
```

原始 manifest：`operation_logs/grpo/hf_upload_20260718_grpo_epoch1_epoch2/manifest.json`。

manifest 最终状态为 `completed`；两个 repository 状态均为 `verified`。运行 staging 与匿名下载副本位于 `artifacts/grpo/hf_upload/hf_upload_20260718_grpo_epoch1_epoch2/`。这些大文件和原始 manifest 不进入 Git。

本地验证：

```text
GRPO stage-only contract: passed
ORPO backward-compatible stage-only contract: passed
GRPO README reproduced byte-for-byte: passed
ORPO README reproduced byte-for-byte: passed
Ruff check / format: passed
Python py_compile: passed
```

## 范围边界

- 只发布两套 LoRA adapter，没有上传或合并基础模型。
- 没有上传 tokenizer、optimizer、scheduler、recovery、训练数据、audit 或 probe 文件。
- 没有修改正式训练权重或 `adapter_config.json`。
- 仓库 README 不声明官方比赛分数提升。
