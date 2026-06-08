#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/infill_graphs_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/infill_graphs_%j.err

# Conda 환경 (경로 수정 필요시 변경)
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd /data3/seonhoyoo/graphcheck-qa/infilling/scripts

# HuggingFace 캐시 (Qwen 로컬 모델 사용 시)
export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
# HF_TOKEN은 필요시 .env 또는 export로 설정

# -----------------------
# 엔티티 infill config
# -----------------------
# GPT 사용 시 OPENAI_API_KEY 필요
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-14B-Instruct | gpt-4o-mini

DATA_FILE="/data3/seonhoyoo/graphcheck-qa/results/2wikimultihopqa/triplets/Qwen2.5-7B-Instruct/triplets_train_sampled.json"
OUTPUT_DIR="/data3/seonhoyoo/graphcheck-qa/infilling/output"

QUESTION_STRATEGY="triplet_only" # triplet_only | combined(권장하지 않음)
INFILL_STRATEGY="triplet_only" # triplet_only | doc_only | combined

USE_GOLD_ONLY=1 # Gold 문서 정보만 활용
MAX_TRIALS=3

# 실험 옵션: 2(전체 문서 O/X) × 5(triplet all/top1/top3/top5/top10)
USE_FULL_DOC=0      # 0: triplet만, 1: 원문 문서 포함
TRIPLET_SELECTION="all"  # all | top1 | top3 | top5 | top10

echo "========== infill_graphs.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "model_name=${MODEL_NAME}"
echo "data_file=${DATA_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "infill_strategy=${INFILL_STRATEGY}"
echo "question_strategy=${QUESTION_STRATEGY}"
echo "use_gold_only=${USE_GOLD_ONLY}"
echo "use_full_doc=${USE_FULL_DOC}"
echo "triplet_selection=${TRIPLET_SELECTION}"
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
    --use_full_doc "${USE_FULL_DOC}"
    --triplet_selection "${TRIPLET_SELECTION}"
)

echo "Command: ${cmd[*]}"

if [[ "${MODEL_NAME}" == gpt* ]] && [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set for GPT model."
    exit 1
fi

"${cmd[@]}"
