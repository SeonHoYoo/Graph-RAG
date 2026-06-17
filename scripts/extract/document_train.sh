#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --exclude=n01,n02
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=2
#SBATCH --job-name=G-doc
#SBATCH --output=/home/hyeseojeon/data/graph/logs/train/0528/document-2wiki_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/logs/train/0528/document-2wiki_%j.err

set -euo pipefail

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME="/home/hyeseojeon/data"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

REASON_DIR="/home/hyeseojeon/data/graph/datasets/train/reason"
OUTPUT_DIR="/home/hyeseojeon/data/graph/datasets/train/graph/document"
MODEL_DIR="/home/hyeseojeon/data/graph/outputs/finetune/Llama-3.2-3B-Instruct-document"

mkdir -p "${OUTPUT_DIR}" /home/hyeseojeon/data/graph/logs/train/0528

TARGET="${1:-all}"  # 사용법: bash document_train.sh hotpotqa  (기본값: all)

declare -A INPUT_FILES
INPUT_FILES["hotpotqa"]="${REASON_DIR}/hotpot_500_open-book_searchr1_168383.jsonl"
INPUT_FILES["2wikimultihopqa"]="${REASON_DIR}/2wiki_500_open-book_searchr1_168383.jsonl"
INPUT_FILES["musique"]="${REASON_DIR}/musique_500_open-book_searchr1_168383.jsonl"

if [[ "${TARGET}" == "all" ]]; then
    datasets=("hotpotqa" "2wikimultihopqa" "musique")
else
    datasets=("${TARGET}")
fi

for dataset in "${datasets[@]}"; do
    input_file="${INPUT_FILES[${dataset}]}"
    if [[ -z "${input_file}" ]]; then
        echo "Unknown dataset: ${dataset}. Choose from: hotpotqa, 2wikimultihopqa, musique"
        exit 1
    fi

    echo "===== ${dataset} ====="
    /data3/hyeseojeon/.conda/envs/sllm3/bin/python -u scripts/extract/document_train.py \
        --dataset    "${dataset}" \
        --input_file "${input_file}" \
        --model_dir  "${MODEL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --hf_home    "${HF_HOME}"
done
