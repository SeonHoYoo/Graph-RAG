#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-48:00:00
#SBATCH --exclude=master
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=think_graph
#SBATCH --output=/home/hyeseojeon/data/graph/logs/extract/think_%x_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/extract/think_%x_%j.err

set -euo pipefail

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"

# Usage (one call per dataset):
#   DATASET=hotpotqa        sbatch --job-name=t-hotpot   think.sh
#   DATASET=2wikimultihopqa sbatch --job-name=t-2wiki    think.sh
#   DATASET=musique         sbatch --job-name=t-musique  think.sh
# SKIML_API_KEY is read from .env automatically.

DATASET="${DATASET:?DATASET env var required}"
model_name="${MODEL:-openai/gpt-4.1-mini-2025-04-14}"
output_dir="${OUTPUT_DIR:-results/graph/0516}"
max_samples="${MAX_SAMPLES:-}"
input_file="${INPUT_FILE:-}"

echo "========== think.sh =========="
echo "DATASET      = ${DATASET}"
echo "model_name   = ${model_name}"
echo "output_dir   = ${output_dir}"
echo "=============================="

mkdir -p /home/hyeseojeon/data/graph/logs/extract

cmd=(
    /data3/hyeseojeon/.conda/envs/sllm3/bin/python -u scripts/extract/think.py
    --dataset "${DATASET}"
    --construct_model_name "${model_name}"
    --output_dir "${output_dir}"
)

if [[ -n "${input_file}" ]]; then
    cmd+=(--input_file "${input_file}")
fi

if [[ -n "${max_samples}" ]]; then
    cmd+=(--max_samples "${max_samples}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
