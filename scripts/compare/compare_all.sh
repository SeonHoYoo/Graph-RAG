#!/bin/bash
#SBATCH --job-name=r1_compare
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeseojeon/data/graph/logs/compare/0507/%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/compare/0507/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hyeseojeon/data/graph}"
cd "${REPO_ROOT}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export HF_HOME=/home/hyeseojeon/data/.cache/huggingface
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" logs/compare

INPUT_FILE="${INPUT_FILE:-results/vanilla/0407(open-book)/musique_vanilla_searchr1_128617_1000.json}"
QUESTION_GRAPH_FILE="${QUESTION_GRAPH_FILE:-results/graph/0409/musique_question_graph.json}"
OUTPUT_FILE="${OUTPUT_FILE:-results/compare/0507/4.1-mini/musique_compare_all.json}"

MODEL="${MODEL:-openai/gpt-4.1-mini-2025-04-14}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TRACE_SOURCE="${TRACE_SOURCE:-file}"   # file | searchr1
RESUME="${RESUME:-false}"
RERUN_EMPTY_SEARCHR1="${RERUN_EMPTY_SEARCHR1:-false}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

cmd=(
    python -u scripts/compare/compare_all.py
    --output "${OUTPUT_FILE}"
    --question-graph-input "${QUESTION_GRAPH_FILE}"
    --model "${MODEL}"
    --trace-source "${TRACE_SOURCE}"
    --temperature "${TEMPERATURE:-0}"
    --start "${START:-0}"
    --sleep "${SLEEP:-0}"
    --max-doc-chars "${MAX_DOC_CHARS:-12000}"
    --max-graph-chars "${MAX_GRAPH_CHARS:-6000}"
    --max-think-chars "${MAX_THINK_CHARS:-4000}"
    --max-query-chars "${MAX_QUERY_CHARS:-1000}"
    --searchr1-retriever-url "http://127.0.0.1:8000/retrieve"
    --searchr1-max-turns 5
    --searchr1-top-k 5
)

if [ "${RESUME}" = true ]; then
    cmd+=(--resume)
fi

if [ "${RERUN_EMPTY_SEARCHR1}" = true ]; then
    cmd+=(--rerun-empty-searchr1)
fi

if [ "${TRACE_SOURCE}" = file ]; then
    cmd+=(--input "${INPUT_FILE}")
fi

if [ -n "${MAX_SAMPLES}" ]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
fi

if [ "${KEEP_SOURCE:-false}" = true ]; then
    cmd+=(--keep-source)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
