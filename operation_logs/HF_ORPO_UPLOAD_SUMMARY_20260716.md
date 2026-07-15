# OneReason-0.8B ORPO LoRA Hugging Face 上传记录（2026-07-16）

## 1. 目标与结果

why：把 ORPO epoch 1、epoch 2 的 LoRA adapter 放入两个公开 Hugging Face 模型仓库，供比赛平台通过 URL 获取；同时固定本地源文件、远端提交和可重复执行的上传命令。

结果：两个公开仓库均已创建并上传完成。每个仓库主动上传且仅上传 `README.md`、`adapter_config.json`、`adapter_model.safetensors`；Hugging Face 为大文件存储自动生成 `.gitattributes`。用户确认 Spec V2.1，允许该文件作为唯一的 Hub 管理文件。

| Epoch | 公开仓库 | HF commit |
| ---: | --- | --- |
| 1 | <https://huggingface.co/reimu996/OneReason-0.8B-ORPO-Epoch1> | `8e445e02f6bf6acb4da7f0a656f584a5ea694e68` |
| 2 | <https://huggingface.co/reimu996/OneReason-0.8B-ORPO-Epoch2> | `143637c7c4ac3120cdb135aa75503428b11a70f5` |

## 2. 固定输入与远端文件

| Epoch | 本地 checkpoint | adapter 权重 bytes | adapter 权重 SHA256 |
| ---: | --- | ---: | --- |
| 1 | `artifacts/orpo/runs/orpo_from_base_epoch2/checkpoint-4089` | 80,792,456 | `c367c92cd09f5b8d8ee896a8928977767cbf544a0e261f5d318aba7bc2cb5c68` |
| 2 | `artifacts/orpo/runs/orpo_from_base_epoch2/checkpoint-8178` | 80,792,456 | `593a117229c11f8bc488acb4e6402c6ec879be116b6d063dd1901974c6fe6a27` |

两份 `adapter_config.json` 均为 1,082 bytes，SHA256 均为 `b8847f943c2f2050181c4ae2db76b189d5c387f9fafb2c3e4aa7bdbbf44dead7`。配置从训练 checkpoint 逐字节复制，未修改 `base_model_name_or_path`。

每个远端仓库的最终文件集合有且仅有：

```text
.gitattributes                 # Hugging Face 自动生成
README.md                      # 主动上传
adapter_config.json            # 主动上传
adapter_model.safetensors      # 主动上传
```

未上传 tokenizer、基础模型、optimizer、scheduler、trainer state、训练数据或 probe 产物。

## 3. 可复用操作

固定配置：`configs/orpo/hf_upload.json`。

上传工具：`scripts/orpo/upload_hf_adapters.py`。

使用环境：`/home/lyc/miniconda3/envs/MiniOneRec_kmeans`；执行前需要该环境的 Hugging Face CLI 已登录到配置中的 namespace。

仅做本地 staging 和哈希门禁，不创建远端仓库：

```bash
/home/lyc/miniconda3/envs/MiniOneRec_kmeans/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --run-id <unique_run_id>
```

创建公开仓库、每仓库一次 commit 上传三个业务文件，并执行脚本内置的公开访问和字节检查：

```bash
/home/lyc/miniconda3/envs/MiniOneRec_kmeans/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --execute \
  --run-id <unique_run_id>
```

安全边界：工具在任何远端写入前校验两个源 checkpoint 的 size/SHA256；目标仓库已存在时直接退出，不覆盖、不删除。若要上传到新的仓库，先修改配置中的两个 `repo_id`，并保证目标仓库尚不存在。运行 staging 位于 `artifacts/orpo/hf_upload/<run_id>/`，manifest 位于 `operation_logs/orpo/<run_id>/manifest.json`，两者均不进入 Git。

## 4. 本次执行记录

实际命令：

```bash
/home/lyc/miniconda3/envs/MiniOneRec_kmeans/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --execute \
  --run-id hf_upload_20260716_orpo_epoch1_epoch2
```

原始 manifest：`operation_logs/orpo/hf_upload_20260716_orpo_epoch1_epoch2/manifest.json`。

V2 脚本先完成两个仓库的上传，再因为远端出现 HF 自动生成的 `.gitattributes` 而按“三文件远端白名单”停止，故原始 manifest 如实保留 `failed`。这不是上传失败：manifest 已记录两个远端 commit，两个仓库也均为公开。随后 Spec V2.1 将 `.gitattributes` 明确为唯一允许的 Hub 管理文件；没有删除或覆盖任何远端内容。

用户在 2026-07-16 明确要求不再继续验证，直接记录并提交。因此本记录不追加新的远端请求、模型加载、前向推理或比赛平台提交。

## 5. 范围边界

- 本次只发布两个 LoRA adapter，没有发布或合并全参数模型。
- 没有修改训练 checkpoint 中的权重或 `adapter_config.json`。
- 没有运行模型加载、生成、受约束 beam search 或比赛隐藏集评测。
- 仓库 README 不声明性能提升；官方比赛得分仍未知。
