#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=n03,n04
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=../logs/infill/0316/infill_graphs_%j.out
#SBATCH --error=../logs/infill/0316/infill_graphs_%j.err

source ~/data/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate graph

cd /home/hyeseojeon/data/Graph-RAG

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN="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"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# -----------------------
# 엔티티 infill config
# -----------------------
# GPT 사용 시 OPENAI_API_KEY 필요
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-14B-Instruct | gpt-4o-mini

DATA_FILE="/home/hyeseojeon/data/Graph-RAG/results/graph/0316/musique_triplets_train_sampled_new.json"
OUTPUT_DIR="/home/hyeseojeon/data/Graph-RAG/results/infill/0316/musique"

QUESTION_STRATEGY="triplet_only" # triplet_only | combined(권장하지 않음)
INFILL_STRATEGY="combined" # doc_only | triplet_only | combined

USE_GOLD_ONLY=1 # Gold 문서 정보만 활용
MAX_TRIALS=3


echo "========== infill_graphs.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "model_name=${MODEL_NAME}"
echo "data_file=${DATA_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "infill_strategy=${INFILL_STRATEGY}"
echo "question_strategy=${QUESTION_STRATEGY}"
echo "use_gold_only=${USE_GOLD_ONLY}"
echo "max_trials=${MAX_TRIALS}"
echo "============================================="

cmd=(
    python -u scripts/infill_graphs.py
    --model_name "${MODEL_NAME}"
    --data_file "${DATA_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --infill_strategy "${INFILL_STRATEGY}"
    --question_strategy "${QUESTION_STRATEGY}"
    --use_gold_only "${USE_GOLD_ONLY}"
    --max_trials "${MAX_TRIALS}"
)

echo "Command: ${cmd[*]}"

if [[ "${MODEL_NAME}" == gpt* ]] && [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set for GPT model."
    exit 1
fi

"${cmd[@]}"
