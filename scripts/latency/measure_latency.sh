#!/bin/bash
#SBATCH --job-name=measure-latency
#SBATCH --output=/home/hyeseojeon/data/graph/logs/latency/measure_latency_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/latency/measure_latency_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=02:00:00

set -euo pipefail

cd /data3/hyeseojeon/graph

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs/latency results/latency

USE_VLLM="${USE_VLLM:-0}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-results/latency}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-latency_benchmark.json}"

DOC_MODEL="${DOC_MODEL:-outputs/finetune/Llama-3.2-3B-Instruct-document}"
TS_MODEL="${TS_MODEL:-outputs/finetune/Llama-3.2-3B-Instruct-question+think+search}"

# Optional args
EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
    EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${DATASETS:-}" ]; then
    EXTRA_ARGS+=(--datasets ${DATASETS})
fi
if [ "${USE_VLLM}" == "1" ]; then
    EXTRA_ARGS+=(--use_vllm --vllm_base_url "${VLLM_BASE_URL}")
fi

echo "========== measure_latency.sh =========="
echo "USE_VLLM             = ${USE_VLLM}"
if [ "${USE_VLLM}" == "1" ]; then
    echo "VLLM_BASE_URL        = ${VLLM_BASE_URL}"
fi
echo "SAMPLES_PER_DATASET  = ${SAMPLES_PER_DATASET}"
echo "DOC_MODEL            = ${DOC_MODEL}"
echo "TS_MODEL             = ${TS_MODEL}"
echo "OUTPUT_DIR           = ${OUTPUT_DIR}"
echo "OUTPUT_FILENAME      = ${OUTPUT_FILENAME}"
echo "========================================="

/data3/hyeseojeon/.conda/envs/sllm3/bin/python -u scripts/latency/measure_latency.py \
    --samples_per_dataset       "${SAMPLES_PER_DATASET}" \
    --document_model_path       "${DOC_MODEL}" \
    --think_search_model_path   "${TS_MODEL}" \
    --output_dir                "${OUTPUT_DIR}" \
    --output_filename           "${OUTPUT_FILENAME}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
