#!/bin/bash
#SBATCH --job-name=vllm-latency
#SBATCH --output=/home/hyeseojeon/data/graph/logs/latency/vllm/graph_latency_vllm_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/latency/vllm/graph_latency_vllm_%j.err
#SBATCH --exclude=master,n03,n04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=08:00:00

set -euo pipefail

cd /data3/hyeseojeon/graph

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs/latency/vllm results/latency/vllm

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

PORT="${PORT:-8006}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
DOC_MODEL="${DOC_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-document}"
TS_MODEL="${TS_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-question+think+search}"
VANILLA_RESULTS="${VANILLA_RESULTS:-/home/hyeseojeon/data/graph/results/latency/latency_benchmark_159194_vanilla_only.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/latency/vllm}"

_doc_basename=$(basename "${DOC_MODEL}")
_model_short=$(echo "${_doc_basename}" | sed 's/-[Ii]nstruct.*//')
OUTPUT_FILENAME="${OUTPUT_FILENAME:-graph_latency_${_model_short}.json}"

VLLM_PID=""

cleanup() {
    if [ -n "${VLLM_PID}" ]; then
        echo "[cleanup] Terminating vLLM server (PID: ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[cleanup] Done."
}
trap cleanup EXIT

wait_for_vllm() {
    echo "[vLLM] Waiting for server to be ready (max 600s)..."
    for i in {1..600}; do
        if curl -s -o /dev/null -w "%{http_code}" "${VLLM_BASE_URL}/models" 2>/dev/null | grep -q "200"; then
            echo "[vLLM] Server is ready!"
            return 0
        fi
        if [ "${i}" -eq 600 ]; then
            echo "ERROR: vLLM server failed to start within 600s"
            exit 1
        fi
        sleep 1
    done
}

# Find adapter path: prefer final/, fall back to latest checkpoint
find_adapter() {
    local model_dir="$1"
    if [ -d "${model_dir}/final" ]; then
        echo "${model_dir}/final"
    else
        ls -d "${model_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -1
    fi
}

PYTHON=/data3/hyeseojeon/.conda/envs/sllm3/bin/python

_doc_adapter=$(find_adapter "${DOC_MODEL}")
_ts_adapter=$(find_adapter "${TS_MODEL}")

if [ -z "${_doc_adapter}" ] || [ -z "${_ts_adapter}" ]; then
    echo "ERROR: Could not find adapter path for DOC_MODEL or TS_MODEL"
    exit 1
fi

_base_model=$($PYTHON -c "import json; print(json.load(open('${_doc_adapter}/adapter_config.json'))['base_model_name_or_path'])")
_lora_rank=$($PYTHON -c "import json; print(json.load(open('${_doc_adapter}/adapter_config.json'))['r'])")
_doc_name=$(basename "${DOC_MODEL}")
_ts_name=$(basename "${TS_MODEL}")

echo "========== Graph Latency (vLLM + LoRA) =========="
echo "PORT              = ${PORT}"
echo "BASE_MODEL        = ${_base_model}"
echo "DOC_ADAPTER       = ${_doc_adapter}"
echo "TS_ADAPTER        = ${_ts_adapter}"
echo "VANILLA_RESULTS   = ${VANILLA_RESULTS}"
echo "OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "OUTPUT_FILENAME   = ${OUTPUT_FILENAME}"
echo "=================================================="

# Start vLLM with both LoRA adapters loaded simultaneously
echo "[1/3] Starting vLLM server with both LoRA adapters..."
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "${_base_model}" \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank "${_lora_rank}" \
    --lora-modules "${_doc_name}=${_doc_adapter}" "${_ts_name}=${_ts_adapter}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    >> "logs/latency/vllm/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

wait_for_vllm

echo "[2/3] Running latency measurement (document + think+search phases)..."
$PYTHON -u scripts/latency/measure_latency.py \
    --skip_vanilla \
    --vanilla_results_path "${VANILLA_RESULTS}" \
    --use_vllm \
    --vllm_base_url "${VLLM_BASE_URL}" \
    --document_model_path "${DOC_MODEL}" \
    --think_search_model_path "${TS_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_filename "${OUTPUT_FILENAME}" \
    "$@"

echo "[done] Graph latency measurement complete!"
