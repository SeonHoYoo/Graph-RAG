#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --exclude=n01,n02
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=G-think
#SBATCH --output=/home/hyeseojeon/data/graph/logs/train/0528/think_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/logs/train/0528/think_%j.err

set -euo pipefail

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

REASON_DIR="/home/hyeseojeon/data/graph/datasets/train/reason"
OUTPUT_DIR="/home/hyeseojeon/data/graph/datasets/train/think"
MODEL_DIR="/home/hyeseojeon/data/graph/outputs/finetune/Llama-3.2-3B-Instruct-question+think+search"
HF_HOME="/home/hyeseojeon/data"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}" /home/hyeseojeon/data/graph/logs/train/0528

for entry in \
    "hotpotqa        ${REASON_DIR}/hotpot_500_open-book_searchr1_168383.jsonl" \
    "2wikimultihopqa ${REASON_DIR}/2wiki_500_open-book_searchr1_168383.jsonl" \
    "musique         ${REASON_DIR}/musique_500_open-book_searchr1_168383.jsonl"
do
    dataset=$(echo $entry | awk '{print $1}')
    input_file=$(echo $entry | awk '{print $2}')

    echo "===== ${dataset} ====="
    /data3/hyeseojeon/.conda/envs/sllm3/bin/python -u scripts/extract/think.py \
        --dataset    "${dataset}" \
        --input_file "${input_file}" \
        --model_dir  "${MODEL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --hf_home    "${HF_HOME}"
done
