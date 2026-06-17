#!/bin/bash
#SBATCH --job-name=r1_compare_one
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=master
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeseojeon/data/graph/logs/compare/0514/%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/compare/0514/%j.err

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
mkdir -p "${HUGGINGFACE_HUB_CACHE}" logs/compare/0514

INPUT_FILE="${INPUT_FILE:-results/graph/combined/0514/musique_combined_0514.json}"
QUESTION_GRAPH_FILE="${QUESTION_GRAPH_FILE:-results/graph/0409/musique_question_graph.json}"
OUTPUT_FILE="${OUTPUT_FILE:-results/compare/0514/musique_compare_one.json}"

MODEL="${MODEL:-openai/gpt-4.1-mini-2025-04-14}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-false}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

cmd=(
    python -u scripts/compare/compare_one.py
    --input "${INPUT_FILE}"
    --output "${OUTPUT_FILE}"
    --question-graph-input "${QUESTION_GRAPH_FILE}"
    --model "${MODEL}"
    --temperature "${TEMPERATURE:-0}"
    --start "${START:-0}"
    --sleep "${SLEEP:-0}"
    --max-doc-chars "${MAX_DOC_CHARS:-30000}"
    --max-think-chars "${MAX_THINK_CHARS:-6000}"
    --max-triple-chars "${MAX_TRIPLE_CHARS:-1000}"
)

if [ "${RESUME}" = true ]; then
    cmd+=(--resume)
fi

if [ -n "${MAX_SAMPLES}" ]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
fi

if [ "${KEEP_SOURCE:-false}" = true ]; then
    cmd+=(--keep-source)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
