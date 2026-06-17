#!/usr/bin/env bash
#SBATCH --job-name=t5-question
#SBATCH --output=/home/hyeseojeon/data/graph/logs/finetune/flant5/0514/flan-t5-question-%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/finetune/flant5/0514/flan-t5-question-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

cd /home/hyeseojeon/data/graph

mkdir -p scripts/finetune/logs outputs/finetune

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME="${MODEL_NAME:-google/flan-t5-base}"

DATA_PATH="${DATA_PATH:-scripts/finetune/data/question/question.json}"
INPUT_FIELD="${INPUT_FIELD:-question}"                    # question | document | think
OUTPUT_DIR="${OUTPUT_DIR:-outputs/finetune/flan-t5-base-question-triples}"
RUN_NAME="${RUN_NAME:-flan-t5-base-question-triples}"

WANDB_FLAG=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  WANDB_FLAG=(--wandb --wandb_project "${WANDB_PROJECT:-graph}")
fi

PRECISION_FLAG=()
if [[ "${USE_BF16:-1}" == "1" ]]; then
  PRECISION_FLAG=(--bf16)
elif [[ "${USE_FP16:-0}" == "1" ]]; then
  PRECISION_FLAG=(--fp16)
fi

python scripts/finetune/t5-base.py \
  --data_path "${DATA_PATH}" \
  --input_field "${INPUT_FIELD}" \
  --model_name_or_path "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --triple_delimiter "${TRIPLE_DELIMITER:-\\n}" \
  --additional_special_tokens "${ADDITIONAL_SPECIAL_TOKENS:-[SEP],[PREP]}" \
  --max_source_length "${MAX_SOURCE_LENGTH:-512}" \
  --max_target_length "${MAX_TARGET_LENGTH:-512}" \
  --train_ratio "${TRAIN_RATIO:-0.95}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-4}" \
  --logging_steps "${LOGGING_STEPS:-25}" \
  --eval_steps "${EVAL_STEPS:-250}" \
  --save_steps "${SAVE_STEPS:-250}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --generation_max_length "${GENERATION_MAX_LENGTH:-512}" \
  --generation_num_beams "${GENERATION_NUM_BEAMS:-4}" \
  --preview_samples "${PREVIEW_SAMPLES:-3}" \
  "${PRECISION_FLAG[@]}" \
  "${WANDB_FLAG[@]}" \
  "$@"
