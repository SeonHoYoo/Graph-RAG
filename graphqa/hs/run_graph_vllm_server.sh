#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=40000MB
#SBATCH --job-name=graph_vllm
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hyeseojeon/data/graph/graphqa/hs/logs/online_feedback/graph_vllm_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/graphqa/hs/logs/online_feedback/graph_vllm_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hyeseojeon/data/graph}"
PY="${PY:-python}"

cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${PROJECT_ROOT}/graphqa/hs/logs/online_feedback"

PORT="${PORT:-8006}"
HOST="${HOST:-127.0.0.1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
QUESTION_MODEL="${QUESTION_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-question}"
DOCUMENT_MODEL="${DOCUMENT_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-document}"
THINK_MODEL="${THINK_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-think}"
QUESTION_NAME="${QUESTION_NAME:-Qwen2.5-0.5B-Instruct-question}"
DOCUMENT_NAME="${DOCUMENT_NAME:-Qwen2.5-0.5B-Instruct-document}"
THINK_NAME="${THINK_NAME:-Qwen2.5-0.5B-Instruct-think}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "============================================================"
echo "[graph_vllm] host:port       = ${HOST}:${PORT}"
echo "[graph_vllm] base_model      = ${BASE_MODEL}"
echo "[graph_vllm] question        = ${QUESTION_NAME} -> ${QUESTION_MODEL}"
echo "[graph_vllm] document        = ${DOCUMENT_NAME} -> ${DOCUMENT_MODEL}"
echo "[graph_vllm] think           = ${THINK_NAME} -> ${THINK_MODEL}"
echo "[graph_vllm] max_lora_rank   = ${MAX_LORA_RANK}"
echo "============================================================"

"${PY}" -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --enable-lora \
    --max-loras 3 \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --lora-modules \
        "${QUESTION_NAME}=${QUESTION_MODEL}" \
        "${DOCUMENT_NAME}=${DOCUMENT_MODEL}" \
        "${THINK_NAME}=${THINK_MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size 1 \
    "$@"
