#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hyeseojeon/data/graph/logs/graph/0526/vanilla_%x_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/graph/0526/vanilla_%x_%j.err

set -euo pipefail

cd /home/hyeseojeon/data/graph

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs/extract

# Available finetuned models (outputs/finetune/):
#   document      : Qwen2.5-0.5B-Instruct-document  |  Llama-3.2-1B-Instruct-document
#                   Llama-3.2-3B-Instruct-document   |  Qwen2.5-7B-Instruct-document
#                   Llama-3.1-8B-Instruct-document   |  Phi-4-mini-instruct-document
#                   Qwen3-4B-Instruct-2507-document
#
#   think+search  : Qwen2.5-0.5B-Instruct-question+think+search
#                   Llama-3.2-1B-Instruct-question+think+search
#                   Llama-3.2-3B-Instruct-question+think+search
#                   Qwen2.5-7B-Instruct-question+think+search
#                   Llama-3.1-8B-Instruct-question+think+search
#                   Phi-4-mini-instruct-question+think+search
#                   Qwen3-4B-Instruct-2507-question+think+search
#
# Example submission:
#   DATASET=2wikimultihopqa \
#   INPUT_FILE=results/vanilla/0407/2wiki_vanilla_searchr1_128615_500.json \
#   sbatch --job-name=2wiki scripts/extract/vanilla.sh

DATASET="${DATASET:?DATASET env var required (hotpotqa|2wikimultihopqa|musique)}"
INPUT_FILE="${INPUT_FILE:?INPUT_FILE env var required}"
OUTPUT_DIR="${OUTPUT_DIR:-results/graph/0526}"

_base="outputs/finetune"
DOC_MODEL="${DOC_MODEL:-${_base}/Qwen2.5-0.5B-Instruct-document}"
TS_MODEL="${TS_MODEL:-${_base}/Qwen2.5-0.5B-Instruct-question+think+search}"

# Derive short model tag from TS_MODEL basename
# e.g. Llama-3.1-8B-Instruct-question+think+search → llama3.1-8b
_model_tag() {
    local name
    name=$(basename "$1")
    name=$(echo "$name" | sed 's/-\(document\|question\|think\).*//')   # strip task suffix
    name=$(echo "$name" | sed 's/-[Ii]nstruct.*//')                     # strip -Instruct[-...]
    name=$(echo "$name" | tr '[:upper:]' '[:lower:]')                   # lowercase
    name=$(echo "$name" | sed 's/\([a-z]\)-\([0-9]\)/\1\2/')           # llama-3 → llama3, phi-4 → phi4
    echo "$name"
}
MODEL_TAG="${MODEL_TAG:-$(_model_tag "${TS_MODEL}")}"

echo "========== vanilla.sh =========="
echo "DATASET    = ${DATASET}"
echo "INPUT_FILE = ${INPUT_FILE}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"
echo "DOC_MODEL  = ${DOC_MODEL}"
echo "TS_MODEL   = ${TS_MODEL}"
echo "MODEL_TAG  = ${MODEL_TAG}"
echo "================================="

EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
    EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${MERGE_LORA:-0}" == "1" ]; then
    EXTRA_ARGS+=(--merge_lora)
fi
if [ "${FLASH_ATTENTION:-0}" == "1" ]; then
    EXTRA_ARGS+=(--flash_attention)
fi

/data3/hyeseojeon/.conda/envs/sllm3/bin/python -u scripts/extract/vanilla.py \
    --dataset                   "${DATASET}" \
    --input_file                "${INPUT_FILE}" \
    --output_dir                "${OUTPUT_DIR}" \
    --document_model_path       "${DOC_MODEL}" \
    --think_search_model_path   "${TS_MODEL}" \
    --model_tag                 "${MODEL_TAG}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
