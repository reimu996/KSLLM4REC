# Frontier SFT Epoch 2 GRPO adapter upload

## Why

The Frontier-started GRPO run has different input data and different output
weights from the historical baseline-started GRPO run. It therefore uses a
separate upload contract and separate Hugging Face repositories. The historical
`configs/grpo/hf_upload.json` file is not modified.

## Local contract generation

Run this only after both GRPO epoch directories are complete:

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
python scripts/grpo/prepare_frontier_hf_upload.py \
  --run-dir artifacts/grpo/runs/frontier_sft_epoch2_grpo_g8_forcedgt_p050_zscore_2epoch
```

The command is local-only. It reads `adapter_model.safetensors` and
`adapter_config.json` from `epoch_001` and `epoch_002`, computes their exact
size/SHA256 values, rejects mismatched adapter configurations, and writes:

```text
configs/grpo/hf_upload_frontier_sft_epoch2.json
```

It prints `remote_upload=not_performed` and does not import or contact the
Hugging Face API.

`configs/grpo/frontier_sft_epoch2_grpo_README.md.tmpl` is the review template
for the repository description. The existing uploader renders the staged
README from the generated configuration's `training` block; that block carries
the same start-adapter, data, normalization, reward, LoRA, and evaluation
statements. The template path and hash are also recorded in the generated
configuration for audit.

## Staging and upload

The existing configuration-driven uploader is reused after the generated
configuration has been reviewed. A stage-only run performs local file and
allowlist checks:

```bash
/home/lyc/miniconda3/envs/MiniOneRec_kmeans/bin/python \
  scripts/orpo/upload_hf_adapters.py \
  --config configs/grpo/hf_upload_frontier_sft_epoch2.json \
  --run-id frontier_sft_epoch2_grpo_stage
```

Only an explicit `--execute` performs remote writes. The uploader refuses to
overwrite an existing repository, uploads `README.md`, `adapter_config.json`,
and `adapter_model.safetensors`, permits Hugging Face's `.gitattributes`, then
checks the public remote revision and anonymous downloads against local
hashes.

The target repositories are fixed:

```text
reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch1
reimu996/OneReason-0.8B-Frontier-SFT-Epoch2-GRPO-Epoch2
```

The clean base model, tokenizer, optimizer state, recovery checkpoints,
training data, and audit logs are not uploaded.
