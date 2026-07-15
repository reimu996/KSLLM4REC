#!/usr/bin/env bash

PROJECT_ROOT="/home/lyc/REC_PROJECTS/KSLLM4REC"
LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
ENV_PREFIX="/home/lyc/miniconda3/envs/onereason_lora_sft"
PYTHON="${ENV_PREFIX}/bin/python"
BASE_MODEL="/home/lyc/models/OneReason-0.8B-pretrain-competition"
SOURCE_DATA="/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl"
PID2SID_DIR="/home/lyc/Data/Explorer_LLM_Rec_Competition/data/OneReason_Pid2Sid"
CAPTION_DIR="/home/lyc/Data/KSLLM4REC_Explorer_SFT/upload_parts/dongwuliao_sid_to_text"
TEXT_PROBE_DIR="/home/lyc/Data/KSLLM4REC_Explorer_SFT/upload_parts/dongwuliao_text_to_sid"
RECOMMEND_PROBE_DIR="/home/lyc/Data/KSLLM4REC_Explorer_SFT/upload_parts/dongyonghu_timeline_to_sid"
DATA_DIR="${PROJECT_ROOT}/artifacts/orpo/data/all_tasks_32705"
LOG_ROOT="${PROJECT_ROOT}/operation_logs/orpo"
CONFIG="${PROJECT_ROOT}/configs/orpo/onereason_lora_orpo.yaml"
GENERATION_LOCK="${PROJECT_ROOT}/configs/orpo/generation.lock.json"
ARTIFACT_LOCK="${PROJECT_ROOT}/configs/orpo/artifacts.lock.json"
ENVIRONMENT_LOCK="${PROJECT_ROOT}/configs/sft/environment.lock.txt"
GATES_REPORT="${PROJECT_ROOT}/artifacts/orpo/gates_report.json"
FULL_OUTPUT="${PROJECT_ROOT}/artifacts/orpo/runs/orpo_from_base_epoch2"

export CONDA_PREFIX="${ENV_PREFIX}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/artifacts/orpo/cache/huggingface/datasets"
export HF_HOME="${PROJECT_ROOT}/artifacts/orpo/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="${PROJECT_ROOT}/artifacts/orpo/cache/triton"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
