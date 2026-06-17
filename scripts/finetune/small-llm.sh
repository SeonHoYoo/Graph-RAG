#!/usr/bin/env bash
#SBATCH --job-name=Q4b-qts
#SBATCH --output=/home/hyeseojeon/data/graph/logs/finetune/qwen4b/0519/question+think+search-%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/finetune/qwen4b/0519/question+think+search-%j.err
#SBATCH --gres=gpu:1
#SBATCH --exclude=master,n01,n02
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd /home/hyeseojeon/data/graph

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

INPUT_FIELD="${INPUT_FIELD:-question+think+search}"         # document | question | think | think+search | think+nosearch | question+think | question+think+search | question+think+nosearch

_data_field="${INPUT_FIELD/nosearch/search}"
DATA_PATH="${DATA_PATH:-/home/hyeseojeon/data/graph/results/train/${_data_field}/combined.jsonl}"

# Supported models (set MODEL_NAME to switch):
#   Qwen/Qwen2.5-0.5B-Instruct
#   meta-llama/Llama-3.2-1B-Instruct
#   microsoft/Phi-4-mini-instruct
#   meta-llama/Llama-3.2-3B-Instruct
#   mistralai/Ministral-3-3B-Instruct-2512
#   Qwen/Qwen3-4B-Instruct-2507
#   Qwen/Qwen2.5-7B-Instruct
#   meta-llama/Llama-3.1-8B-Instruct
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"  #1B에서도 잘 버팀.

# Auto-adjust batch/grad_accum for RTX Pro 6000 (24 GB) based on model size.
# max_length is kept uniform across all models for fair comparison.
# All tiers keep effective batch size = 32. Override any var before calling.
_model_lower="${MODEL_NAME,,}"
if [[ "${_model_lower}" =~ (7b|8b) ]]; then
  : "${TRAIN_BATCH_SIZE:=1}"
  : "${EVAL_BATCH_SIZE:=1}"
  : "${GRAD_ACCUM_STEPS:=32}"
  : "${USE_GRAD_CKPT:=1}"
elif [[ "${_model_lower}" =~ (3b|4b) ]]; then
  : "${TRAIN_BATCH_SIZE:=2}"
  : "${EVAL_BATCH_SIZE:=2}"
  : "${GRAD_ACCUM_STEPS:=16}"
  : "${USE_GRAD_CKPT:=1}"
else
  : "${TRAIN_BATCH_SIZE:=8}"
  : "${EVAL_BATCH_SIZE:=8}"
  : "${GRAD_ACCUM_STEPS:=4}"
  : "${USE_GRAD_CKPT:=0}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/finetune/$(basename ${MODEL_NAME})-${INPUT_FIELD}}"
RUN_NAME="${RUN_NAME:-graph}"

WANDB_FLAG=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  WANDB_FLAG=(--wandb --wandb_project "${WANDB_PROJECT:-graph}")
fi

PRECISION_FLAG=()
if [[ "${USE_FP16:-0}" == "1" ]]; then
  PRECISION_FLAG=(--fp16)
else
  PRECISION_FLAG=(--bf16)
fi

FLASH_FLAG=()
if [[ "${USE_FLASH_ATTENTION:-1}" == "1" ]]; then
  FLASH_FLAG=(--flash_attention)
fi

GRAD_CKPT_FLAG=()
if [[ "${USE_GRAD_CKPT:-0}" == "1" ]]; then
  GRAD_CKPT_FLAG=(--gradient_checkpointing)
fi

/data3/hyeseojeon/.conda/envs/sllm3/bin/python scripts/finetune/small-llm.py \
  --data_path "${DATA_PATH}" \
  --input_field "${INPUT_FIELD}" \
  --model_name_or_path "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --triple_delimiter "${TRIPLE_DELIMITER:-\\n}" \
  --max_length "${MAX_LENGTH:-1536}" \
  --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS:-1024}" \
  --generation_num_beams "${GENERATION_NUM_BEAMS:-1}" \
  --train_ratio "${TRAIN_RATIO:-0.95}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --learning_rate "${LEARNING_RATE:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-4}" \
  --logging_steps "${LOGGING_STEPS:-50}" \
  --f1_eval_samples "${F1_EVAL_SAMPLES:-100}" \
  --preview_samples "${PREVIEW_SAMPLES:-3}" \
  "${PRECISION_FLAG[@]}" \
  "${FLASH_FLAG[@]}" \
  "${GRAD_CKPT_FLAG[@]}" \
  "${WANDB_FLAG[@]}" \
  "$@"
