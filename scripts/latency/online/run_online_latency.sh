#!/bin/bash
#SBATCH --job-name=latency
#SBATCH --output=/home/hyeseojeon/data/graph/logs/latency/online/0528/online_latency_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/latency/online/0528/online_latency_%j.err
#SBATCH --exclude=master,n01,n03,n04
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --time=0-48:00:00

set -euo pipefail

cd /data3/hyeseojeon/graph

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PYTHON=/data3/hyeseojeon/.conda/envs/sllm3/bin/python

# ── input ────────────────────────────────────────────────────────────────────
DATASETS_ROOT="${DATASETS_ROOT:-/home/hyeseojeon/data/graph/datasets}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa musique}"                # hotpotqa 2wikimultihopqa musique
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-}"                        # empty = all
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8001/retrieve}"
SEARCHR1_MODEL="${SEARCHR1_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"

# ── graph models ──────────────────────────────────────────────────────────────
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
DOC_MODEL="${DOC_MODEL:-outputs/finetune/Qwen3-4B-Instruct-2507-document}"
TS_MODEL="${TS_MODEL:-outputs/finetune/Qwen3-4B-Instruct-2507-question+think+search}"

# ── output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-results/latency/online}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-online_latency_$(basename "${DOC_MODEL}" | sed 's/-[Ii]nstruct.*//').json}"
VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

mkdir -p logs/latency/online "${OUTPUT_DIR}"

# ── adapter paths ────────────────────────────────────────────────────────────

find_adapter() {
    local dir="$1"
    if   [ -d "${dir}/final" ];          then echo "${dir}/final"
    else ls -d "${dir}"/checkpoint-* 2>/dev/null | sort -V | tail -1
    fi
}

DOC_ADAPTER=$(find_adapter "${DOC_MODEL}")
TS_ADAPTER=$(find_adapter "${TS_MODEL}")

if [ -z "${DOC_ADAPTER}" ] || [ -z "${TS_ADAPTER}" ]; then
    echo "ERROR: adapter not found for DOC_MODEL or TS_MODEL"; exit 1
fi

BASE_MODEL=$($PYTHON -c "import json; print(json.load(open('${DOC_ADAPTER}/adapter_config.json'))['base_model_name_or_path'])")
LORA_RANK=$($PYTHON -c  "import json; print(json.load(open('${DOC_ADAPTER}/adapter_config.json'))['r'])")

echo "DATASETS    = ${DATASETS}  (n=${SAMPLES_PER_DATASET} each)"
echo "SEARCHR1    = ${SEARCHR1_MODEL}"
echo "BASE_MODEL  = ${BASE_MODEL}"
echo "DOC_ADAPTER = ${DOC_ADAPTER}"
echo "TS_ADAPTER  = ${TS_ADAPTER}"
echo "OUTPUT      = ${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# ── vLLM server (GPU 1) ──────────────────────────────────────────────────────

VLLM_PID=""
cleanup() {
    [ -n "${VLLM_PID}" ] && kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model              "${BASE_MODEL}"    \
    --enable-lora                           \
    --max-loras          2                  \
    --max-lora-rank      "${LORA_RANK}"     \
    --lora-modules       "$(basename "${DOC_MODEL}")=${DOC_ADAPTER}" \
                         "$(basename "${TS_MODEL}")=${TS_ADAPTER}"   \
    --host               127.0.0.1          \
    --port               "${PORT}"          \
    --gpu-memory-utilization 0.95           \
    --max-model-len          131072         \
    --tensor-parallel-size   1              \
    >> "logs/latency/online/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server..."
for i in {1..600}; do
    curl -s -o /dev/null -w "%{http_code}" "${VLLM_BASE_URL}/models" 2>/dev/null \
        | grep -q "200" && echo "vLLM ready." && break
    [ "${i}" -eq 600 ] && { echo "ERROR: vLLM server timed out"; exit 1; }
    sleep 1
done

# ── latency measurement (GPU 0) ──────────────────────────────────────────────

CUDA_VISIBLE_DEVICES=0 $PYTHON -u scripts/latency/online/measure_online_latency.py \
    --datasets_root           "${DATASETS_ROOT}"         \
    --datasets                ${DATASETS}                \
    ${SAMPLES_PER_DATASET:+--samples_per_dataset "${SAMPLES_PER_DATASET}"} \
    --searchr1_model_id       "${SEARCHR1_MODEL}"        \
    --retriever_url           "${RETRIEVER_URL}"         \
    --vllm_base_url           "${VLLM_BASE_URL}"         \
    --document_model_path     "${DOC_MODEL}"             \
    --think_search_model_path "${TS_MODEL}"              \
    --output_dir              "${OUTPUT_DIR}"            \
    --output_filename         "${OUTPUT_FILENAME}"       \
    "$@"
