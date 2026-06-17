#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=docgraph
#SBATCH --exclude=master,n01,n02
#SBATCH --output=/home/hyeseojeon/data/graph/logs/graph/0514/fill_document_graphs_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/graph/0514/fill_document_graphs_%j.err

set -euo pipefail

cd /home/hyeseojeon/data/graph
mkdir -p logs/graph

COMBINED_DIR="${COMBINED_DIR:-/home/hyeseojeon/data/graph/results/graph/combined/0514}"
GRAPHCHECK_ROOT="${GRAPHCHECK_ROOT:-/home/hyeseojeon/data/graph/__graphcheck-qa-2}"
CONSTRUCT_MODEL="${CONSTRUCT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
INPUT_FILES="${INPUT_FILES:-${COMBINED_DIR}/hotpotqa_combined_0514.json}"

#  ${COMBINED_DIR}/hotpotqa_combined_0514.json ${COMBINED_DIR}/musique_combined_0514.json

python -u scripts/fill_document_graphs.py \
  --input-file ${INPUT_FILES} \
  --graphcheck-root "${GRAPHCHECK_ROOT}" \
  --construct-model-name "${CONSTRUCT_MODEL}"
