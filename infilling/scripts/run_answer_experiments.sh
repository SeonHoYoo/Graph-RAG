#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_answer_experiments_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_answer_experiments_%j.err

# 절대 경로 사용
BASE_DIR="/data3/seonhoyoo/graphcheck-qa"
INFILL_SCRIPTS="${BASE_DIR}/infilling/scripts"
ANSWER_PY="${INFILL_SCRIPTS}/scripts/answer.py"
OUTPUT_BASE="${BASE_DIR}/infilling/output"

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd "${INFILL_SCRIPTS}"

export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# 사용법: sbatch run_answer_experiments.sh [DATASET] [MODEL_NAME]
#   DATASET: 2wikimultihopqa | hotpotqa | musique | all
#   MODEL_NAME: gpt-4o-mini 등 (기본: gpt-4o-mini)

DATASET_ARG="${1:-all}"
MODEL_NAME="${2:-gpt-4o-mini}"
ENT_EXIST_FLAG="${3:-all}"

DATASETS=("2wikimultihopqa" "hotpotqa" "musique")
if [[ "${DATASET_ARG}" == "all" ]]; then
    TO_RUN=("${DATASETS[@]}")
else
    TO_RUN=("${DATASET_ARG}")
fi

for ds in "${TO_RUN[@]}"; do
    infill_dir="${OUTPUT_BASE}/infill/${ds}"
    answer_dir="${OUTPUT_BASE}/answer/${ds}"
    if [[ ! -d "${infill_dir}" ]]; then
        echo "SKIP: ${infill_dir} not found"
        continue
    fi
    mkdir -p "${answer_dir}"
    for f in "${infill_dir}"/infill_*.json; do
        [[ -f "$f" ]] || continue
        if [[ -f "${f%.json}.answer_done" ]]; then
            echo "SKIP (already done): ${f}"
            continue
        fi
        echo ""
        echo ">>> Answer: ${f}"
        python -u "${ANSWER_PY}" \
            --model_name "${MODEL_NAME}" \
            --data_file "${f}" \
            --output_dir "${answer_dir}" \
            --ent_exist_flag "${ENT_EXIST_FLAG}" \
            --max_trials 3
        touch "${f%.json}.answer_done" 2>/dev/null || true
    done
done

echo ""
echo "========== Answer experiments completed =========="
