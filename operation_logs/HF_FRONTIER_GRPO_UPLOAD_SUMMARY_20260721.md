# Frontier GRPO LoRA Hugging Face 上传摘要

## 结论

Frontier SFT Epoch 2 -> GRPO 的 Epoch 1 与 Epoch 2 LoRA adapter 均已在
`reimu996` 名下公开，并完成匿名下载 SHA256 校验。

- Epoch 1：<https://huggingface.co/reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch1>
- Epoch 2：<https://huggingface.co/reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch2>

## 执行过程

1. 由 `configs/grpo/hf_upload_frontier_sft_epoch2.json` 生成 stage-only 运行：
   `frontier_sft_epoch2_grpo_stage_20260721_041542`。
2. stage-only 状态为 `prepared_only`；每轮 staging 目录有且仅有
   `README.md`、`adapter_config.json`、`adapter_model.safetensors`。
3. 首次正式执行
   `frontier_sft_epoch2_grpo_upload_20260721_041607` 被安全检查阻止，因为
   Epoch 1 目标仓库已存在；脚本没有覆盖任何文件，也没有继续处理 Epoch 2。
4. 对 Epoch 1 做只读匿名核验：仓库公开，权重和 config 与本次本地产物
   逐字节相同，因此保留已有仓库，不覆盖其 README。
5. Epoch 2 当时仍为 404；随后仅创建 Epoch 2 公共仓库，上传 stage-only
   中已经过哈希校验的三项业务文件。
6. 最后以 `token=False` 匿名读取两个仓库，核对公开状态、远端文件白名单、
   revision、文件大小和 SHA256。

## 远端结果

| Epoch | 状态 | HF revision |
|---|---|---|
| 1 | 已存在；adapter 精确匹配，未覆盖 | `1489da6d7b8a6cf99e6d3f3cb2c9c1a33a278191` |
| 2 | 本次创建、上传并验证 | `3587ed2a464d3ac70323939b7578ba81391c22ac` |

两个仓库的远端文件集合均精确为：

```text
.gitattributes
README.md
adapter_config.json
adapter_model.safetensors
```

## 匿名下载校验

### Epoch 1

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `README.md` | 1,017 | `7c6a1182f0c5da9224ad31c4eeace9cfd7f45d35d569299d0a7579345dc60e60` |
| `adapter_config.json` | 1,081 | `28ab07197f42ec5b551622437a008721e4b7296bd12f79dd347e028c4d93de3e` |
| `adapter_model.safetensors` | 80,792,456 | `51787cf83fea5bfa11b1015acaf42236cbc3eec5bda22511e59f6e1ffb6f842c` |

### Epoch 2

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `README.md` | 898 | `345072a60d3c4034dce7b7b28e8bd0b3fe1cfffa2b261175ce8b37584e880e1e` |
| `adapter_config.json` | 1,081 | `28ab07197f42ec5b551622437a008721e4b7296bd12f79dd347e028c4d93de3e` |
| `adapter_model.safetensors` | 80,792,456 | `6541a936c982af332c969da83e867fcad556e9125ac2db6947dd18082acc1724` |

Epoch 1 的 README 是已有仓库中的准确说明，但仍写着本地 probe pending；本次
没有为更新说明而覆盖一个已经正确的 adapter 仓库。该差异不影响 adapter 权重、
PEFT 配置或平台加载。

## 本地绑定

- 上传前 Git HEAD：`e62b0cef581623941e0255686ee7a00c3122658c`
- 本地 Epoch 1/2 adapter SHA 与上表远端匿名下载 SHA 完全一致。
- 两个远端仓库均为 public；校验过程没有输出或记录 HF token。
