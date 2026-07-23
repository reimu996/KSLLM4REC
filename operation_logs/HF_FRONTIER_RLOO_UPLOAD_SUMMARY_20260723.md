# Frontier RLOO LoRA Hugging Face 上传摘要

## 目标与结论

上传要解决的问题是让比赛平台能够通过公开 URL 下载两轮 RLOO LoRA adapter，同时证明远端文件与正式训练产物逐字节一致。Epoch 1 和 Epoch 2 现已公开；两轮均固定 HF revision，以匿名身份重新下载 README、配置和权重并通过大小与 SHA256 校验。

| Epoch | 公开仓库 | 最终 HF revision |
|---:|---|---|
| 1 | <https://huggingface.co/reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-RLOO-Epoch1> | `805a41067a30396f5e4239a57e4837960b4aec87` |
| 2 | <https://huggingface.co/reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-RLOO-Epoch2> | `d3fce5489986f65080f944de51b0a029d5fc345d` |

HF 登录身份为 `reimu996`，两个仓库均为 `private=false`。仓库模型卡只陈述训练方法和本地 probe 状态，不声明官方比赛提分。

## 固定上传合同

合同文件：`configs/rloo/hf_upload_frontier_sft_epoch2.json`，SHA256 为 `b5a9e9a31612b869dec748ca1647751638bab95ead3db29454e3371911ef7512`。

stage-only manifest 记录的上传前 Git HEAD 为 `e3795825907cbd1175d1b6c77001c2a7ff896bee`。

每个仓库最终的业务文件集合仅包含：

```text
README.md
adapter_config.json
adapter_model.safetensors
```

Hugging Face 自动维护 `.gitattributes`。最终两个远端仓库的文件集合均有且仅有：

```text
.gitattributes
README.md
adapter_config.json
adapter_model.safetensors
```

没有上传基础模型、tokenizer、optimizer、scheduler、recovery、训练数据、逐组 audit、门禁报告或 probe 预测。

## 远端执行过程

1. 使用既有配置驱动上传器完成本地 stage-only 运行 `frontier_rloo_stage_20260723_01`。两轮 staging 均只有三项业务文件，源权重与配置先通过固定 size/SHA256 门禁。
2. 远端预检发现 Epoch 1 仓库已经存在，revision 为 `2e8466fdac216a1617630e66285b282c68200e79`，当时只有 `.gitattributes`、配置和权重；Epoch 2 仓库不存在。
3. 匿名下载 Epoch 1 的配置和 161,533,160 字节权重，与本次正式 Epoch 1 产物逐字节一致。因此保留已有参数文件，只新增经过 staging 的 `README.md`，没有覆盖配置或权重。
4. 首次远端运行 `frontier_rloo_upload_20260723_01` 在任何写入前失败：匿名查询不存在的 Epoch 2 时 HF 返回 401，而检查代码只接受 404。失败 manifest 被保留；Epoch 1 revision 未变、Epoch 2 仍不存在。
5. 新运行 `frontier_rloo_upload_20260723_02` 改用已登录身份做“不存在”检查，预检通过；随后为 Epoch 1 补 README，新建并上传 Epoch 2。
6. 最后固定两个新 revision，使用 `token=False` 匿名下载三项业务文件，逐项比较大小和 SHA256；两个仓库均通过。

## 匿名下载校验

| Epoch | 文件 | 字节数 | SHA256 |
|---:|---|---:|---|
| 1 | `README.md` | 984 | `01523f9f21649fe31c2efacb88e8f432db113d54a9ecaae9a893d76a29d2db0a` |
| 1 | `adapter_config.json` | 1,081 | `32e96266efa21abaebd076e9287ca108c3a5abab0c4be3aa1e13fe3427bb109e` |
| 1 | `adapter_model.safetensors` | 161,533,160 | `796a960c1759770f19cdde632bafeac56d848f6aac64436de413933cc3c66dc2` |
| 2 | `README.md` | 984 | `f17f58f5b345fd7b6fbd1fea5643712132e12fd29b7fbb086c4342e2808977d9` |
| 2 | `adapter_config.json` | 1,081 | `32e96266efa21abaebd076e9287ca108c3a5abab0c4be3aa1e13fe3427bb109e` |
| 2 | `adapter_model.safetensors` | 161,533,160 | `aeebde52b58efd3685a2925de9b66163d40a91e62467fac7e233b68e11c8fdb2` |

两个远端 adapter 权重 SHA 与正式训练目录和最终 verifier 中记录的 SHA 完全一致。`adapter_config.json` 也从正式 checkpoint 原样上传，没有修改 `base_model_name_or_path`。

## 可复用操作

固定环境：`/home/lyc/miniconda3/envs/onereason_lora_sft`。只做本地 staging 和文件门禁、不写远端：

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --config configs/rloo/hf_upload_frontier_sft_epoch2.json \
  --run-id <new-run-id>
```

对于两个全新的目标仓库，在配置中换成尚不存在的 repo ID 后增加 `--execute`，工具会在任何远端写入前验证身份、源文件哈希和两个目标均不存在；上传后再做公开状态、文件白名单、revision 和匿名下载校验。当前两个固定 repo ID 已存在，安全检查会拒绝覆盖，这是预期行为。

本次“Epoch 1 已存在且参数精确匹配、Epoch 2 不存在”的混合修复没有包装成可重复覆盖的 CLI：实际动作是为 Epoch 1 仅提交缺失 README，并为 Epoch 2 创建仓库后提交三项业务文件。该边界刻意保留了“禁止覆盖已有仓库”的安全规则；三个 manifest 记录前置状态、动作、revision 和最终匿名校验结果。

原始审计记录：

- stage-only manifest：`operation_logs/rloo/frontier_sft_epoch2_hf_upload/frontier_rloo_stage_20260723_01/manifest.json`，SHA256 `e2418bccf253758040952bd88667907680a420ec735533219df0b98767248405`
- 无远端写入的失败 manifest：`operation_logs/rloo/frontier_sft_epoch2_hf_upload/frontier_rloo_upload_20260723_01/manifest.json`，SHA256 `8272d39836f98c08b53f4a03678f9cc654b24308159adcdda230b778029a173b`
- 完成 manifest：`operation_logs/rloo/frontier_sft_epoch2_hf_upload/frontier_rloo_upload_20260723_02/manifest.json`，SHA256 `2b687db6e252acc287df3ae20971a016dfb78c194b91f01da3dad5cf957b14e1`

这些原始 manifest、staging 和匿名下载副本均位于 Git 忽略目录；Git 只提交固定上传合同和本摘要。

一句话结论：两个 RLOO adapter 已公开，远端权重和配置与本地正式产物逐字节一致。
