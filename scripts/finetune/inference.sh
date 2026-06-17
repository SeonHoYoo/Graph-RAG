#!/usr/bin/env bash
#SBATCH --job-name=latency
#SBATCH --output=/home/hyeseojeon/data/graph/logs/finetune/inference/0515/%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/finetune/inference/0515/%j.err
#SBATCH --gres=gpu:1
#SBATCH --exclude=master,n03,n04
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

cd /home/hyeseojeon/data/graph

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_DIR="${MODEL_DIR:-outputs/finetune/Llama-3.2-3B-Instruct-think-triples/checkpoint-1000}"
DATA_PATH="${DATA_PATH:-scripts/finetune/data/think/think.json}"
INPUT_FIELD="${INPUT_FIELD:-think}"         # document | question | think
OUTPUT_PATH="${OUTPUT_PATH:-results/inference/$(basename $(dirname ${MODEL_DIR}))-$(basename ${MODEL_DIR}).json}"

mkdir -p "$(dirname "${OUTPUT_PATH}")" logs/finetune/inference/0515

python scripts/finetune/inference.py \
  --model_dir "${MODEL_DIR}" \
  --data_path "${DATA_PATH}" \
  --input_field "${INPUT_FIELD}" \
  --output_path "${OUTPUT_PATH}" \
  --max_length "${MAX_LENGTH:-1536}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  --num_beams "${NUM_BEAMS:-1}" \
  --hf_home "${HF_HOME}" \
  "$@"
