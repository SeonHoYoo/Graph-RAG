#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=answer
#SBATCH --gres=gpu:1
#SBATCH --exclude=n03,n04,master
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=../logs/answer/0316/answer_%j.out
#SBATCH --error=../logs/answer/0316/answer_%j.err

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
# 답변 생성 config
# -----------------------
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-14B-Instruct | gpt-4o-mini
DATA_FILE="/home/hyeseojeon/data/Graph-RAG/results/infill/0316/musique/infill_musique_triplet_only_triplet_only_gold.json"
OUTPUT_DIR="/home/hyeseojeon/data/Graph-RAG/results/answer/0316/musique"

ENT_EXIST_FLAG="all"     # all | false((ENT)가 모두 infill된 데이터만)
MAX_TRIALS=3

echo "========== answer.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "model_name=${MODEL_NAME}"
echo "data_file=${DATA_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "ent_exist_flag=${ENT_EXIST_FLAG}"
echo "max_trials=${MAX_TRIALS}"
echo "====================================="

cmd=(
    python -u scripts/answer.py
    --model_name "${MODEL_NAME}"
    --data_file "${DATA_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --ent_exist_flag "${ENT_EXIST_FLAG}"
    --max_trials "${MAX_TRIALS}"
)

echo "Command: ${cmd[*]}"

if [[ "${MODEL_NAME}" == gpt* ]] && [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set for GPT model."
    exit 1
fi

"${cmd[@]}"
