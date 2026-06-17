#!/bin/bash
#SBATCH --job-name=measure-graph-latency
#SBATCH --output=/home/hyeseojeon/data/graph/logs/latency/graph_latency_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/latency/graph_latency_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00

set -euo pipefail

cd /data3/hyeseojeon/graph

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs/latency results/latency

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
VANILLA_RESULTS="${VANILLA_RESULTS:?VANILLA_RESULTS required (path to vanilla SearchR1 results JSON)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/latency}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-graph_latency_benchmark.json}"

DOC_MODEL="${DOC_MODEL:-outputs/finetune/Llama-3.2-3B-Instruct-document}"
TS_MODEL="${TS_MODEL:-outputs/finetune/Llama-3.2-3B-Instruct-question+think+search}"

echo "========== measure_graph_latency.sh =========="
echo "VANILLA_RESULTS    = ${VANILLA_RESULTS}"
echo "VLLM_BASE_URL      = ${VLLM_BASE_URL}"
echo "DOC_MODEL          = ${DOC_MODEL}"
echo "TS_MODEL           = ${TS_MODEL}"
echo "OUTPUT_DIR         = ${OUTPUT_DIR}"
echo "OUTPUT_FILENAME    = ${OUTPUT_FILENAME}"
echo "=============================================="

/data3/hyeseojeon/.conda/envs/sllm3/bin/python -u scripts/latency/measure_latency.py \
    --skip_vanilla \
    --vanilla_results_path "${VANILLA_RESULTS}" \
    --use_vllm \
    --vllm_base_url "${VLLM_BASE_URL}" \
    --document_model_path "${DOC_MODEL}" \
    --think_search_model_path "${TS_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_filename "${OUTPUT_FILENAME}" \
    "$@"
