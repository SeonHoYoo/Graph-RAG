#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=master
#SBATCH --time=0-24:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=2
#SBATCH --output=../../logs/embedding_%j.log
#SBATCH --error=../../logs/embedding_%j.err

source ~/data/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate graph

# 작업 디렉토리로 이동
cd /home/hyeseojeon/data/Graph-RAG

# Hugging Face 캐시 경로 고정(노드가 달라도 동일 경로 사용)
export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN="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"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# -----------------------
# 임베딩 검증 설정
# -----------------------
DATA_FILE="/home/hyeseojeon/data/Graph-RAG/results/2wikimultihopqa/graph_infill/train_sampled_multihop_graphcheck_triplets_115374_500_Qwen2.5-7B-Instruct_triplet_only_triplet_only_115726.json"
OUTPUT_DIR="/home/hyeseojeon/data/Graph-RAG/results/2wikimultihopqa/verification/embedding"

MODEL_NAME="BAAI/bge-large-en-v1.5"
TOP_K=10
BATCH_SIZE=8
MAX_LENGTH=1024
# MAX_SAMPLES=100

mkdir -p "${OUTPUT_DIR}"
INPUT_STEM="$(basename "${DATA_FILE}" .json)"
JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_STEM}_embedding_top${TOP_K}_${JOB_ID}.json"

echo "========== embedding.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "data_file=${DATA_FILE}"
echo "output_file=${OUTPUT_FILE}"
echo "model_name=${MODEL_NAME}"
echo "top_k=${TOP_K}"
echo "batch_size=${BATCH_SIZE}"
echo "max_length=${MAX_LENGTH}"
echo "========================================="

cmd=(
    python -u scripts/verification/embedding.py
    --data_file "${DATA_FILE}"
    --output_file "${OUTPUT_FILE}"
    --model_name "${MODEL_NAME}"
    --top_k "${TOP_K}"
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
