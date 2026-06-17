#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --exclude=master,n01,n02
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hyeseojeon/data/graph/logs/extract/document_%x_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/extract/document_%x_%j.err

set -euo pipefail

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/extract

# DATASET must be set when submitting:
#   DATASET=hotpotqa        sbatch --job-name=hotpotqa   scripts/extract/document.sh
#   DATASET=2wikimultihopqa sbatch --job-name=2wiki       scripts/extract/document.sh
#   DATASET=musique         sbatch --job-name=musique     scripts/extract/document.sh
DATASET="${DATASET:?DATASET env var required (hotpotqa|2wikimultihopqa|musique)}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/graph/0515}"

echo "========== document.sh =========="
echo "DATASET    = ${DATASET}"
echo "MODEL      = ${MODEL}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"
echo "================================="

/data3/hyeseojeon/miniconda3/bin/python -u scripts/extract/document.py \
    --dataset        "${DATASET}" \
    --construct_model_name "${MODEL}" \
    --output_dir     "${OUTPUT_DIR}"
