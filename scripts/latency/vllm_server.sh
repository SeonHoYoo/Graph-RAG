#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --output=/home/hyeseojeon/data/graph/logs/latency/vllm_server_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/latency/vllm_server_%j.err
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

mkdir -p logs/latency

MODEL_PATH="${MODEL_PATH:?MODEL_PATH required (e.g., outputs/finetune/Llama-3.2-3B-Instruct-document)}"
PORT="${PORT:-8001}"

echo "========== vLLM Server =========="
echo "MODEL_PATH = ${MODEL_PATH}"
echo "PORT       = ${PORT}"
echo "=================================="

/data3/hyeseojeon/.conda/envs/sllm3/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    "$@"
