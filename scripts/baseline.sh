#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=base
#SBATCH --gres=gpu:1
#SBATCH --exclude=n03,n04,master
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=../logs/answer/0320/baseline_%j.out
#SBATCH --error=../logs/answer/0320/baseline_%j.err

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
# baseline config
# -----------------------
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-14B-Instruct | gpt-4o-mini
DATA_FILE="/home/hyeseojeon/data/Graph-RAG/results/graph/0320/gpt4.1/musique_triplets_train_sampled.json"
OUTPUT_DIR="/home/hyeseojeon/data/Graph-RAG/results/baseline/0320/musique"

QUESTION_STRATEGY="question_only"  # question_only | triplet_only | combined
CONTEXT_STRATEGY="combined"     # doc_only | triplet_only | combined
USE_GOLD_DOC=1             # 0 | 1 (Gold document만)
USE_GOLD_GRAPH=1            # 0 | 1 (Gold document graph만) 

MAX_TRIALS=3

echo "========== baseline.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "model_name=${MODEL_NAME}"
echo "data_file=${DATA_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "question_strategy=${QUESTION_STRATEGY}"
echo "context_strategy=${CONTEXT_STRATEGY}"
echo "use_gold_graph=${USE_GOLD_GRAPH}"
echo "use_gold_doc=${USE_GOLD_DOC}"
echo "max_trials=${MAX_TRIALS}"
echo "======================================="

cmd=(
    python -u scripts/baseline.py
    --model_name "${MODEL_NAME}"
    --data_file "${DATA_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --question_strategy "${QUESTION_STRATEGY}"
    --context_strategy "${CONTEXT_STRATEGY}"
    --use_gold_graph "${USE_GOLD_GRAPH}"
    --use_gold_doc "${USE_GOLD_DOC}"
    --max_trials "${MAX_TRIALS}"
)

echo "Command: ${cmd[*]}"

if [[ "${MODEL_NAME}" == gpt* ]] && [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set for GPT model."
    exit 1
fi

"${cmd[@]}"
