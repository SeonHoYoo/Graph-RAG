#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=baseline
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=2
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/baseline_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/baseline_%j.err

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd /data3/seonhoyoo/graphcheck-qa/infilling/scripts

export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# -----------------------
# baseline config
# -----------------------
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-14B-Instruct | gpt-4o-mini
DATA_FILE="/data3/seonhoyoo/graphcheck-qa/infilling/output/infill_2wikimultihopqa_triplet_only_triplet_only_gold.json"
OUTPUT_DIR="/data3/seonhoyoo/graphcheck-qa/infilling/output"

CONTEXT_STRATEGY="doc"     # doc | docgraph | combined
USE_GOLD_DOC=1              # 0 | 1 (Gold document만)
USE_GOLD_GRAPH=1            # 0 | 1 (Gold document graph만) 

MAX_TRIALS=5

echo "========== baseline.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "model_name=${MODEL_NAME}"
echo "data_file=${DATA_FILE}"
echo "output_dir=${OUTPUT_DIR}"
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
