#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_infill_experiments_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_infill_experiments_%j.err

# SLURM 실행 시 작업 디렉토리가 달라지므로 절대 경로 사용
BASE_DIR="/data3/seonhoyoo/graphcheck-qa"
INFILL_SCRIPTS="${BASE_DIR}/infilling/scripts"
INFILL_PY="${INFILL_SCRIPTS}/scripts/infill_graphs.py"

# Conda 환경
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

# 작업 디렉토리 고정 (SLURM spool 디렉토리 대신)
cd "${INFILL_SCRIPTS}"

export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# 10개 실험: 2(전체 문서 O/X) × 5(triplet all/top1/top3/top5/top10)
# 사용법: sbatch run_infill_experiments.sh [DATASET] [MODEL_NAME]
#   DATASET: 2wikimultihopqa | hotpotqa | musique | all (기본: all)
#   MODEL_NAME: Qwen/Qwen2.5-7B-Instruct 등 (기본: Qwen/Qwen2.5-7B-Instruct)

DATASET_ARG="${1:-all}"
MODEL_NAME="${2:-Qwen/Qwen2.5-7B-Instruct}"

# 3개 데이터셋
DATASETS=("2wikimultihopqa" "hotpotqa" "musique")
RESULTS_BASE="${BASE_DIR}/results"
OUTPUT_BASE="${BASE_DIR}/infilling/output"
TRIPLET_MODEL="${MODEL_NAME##*/}"
MODEL_SHORT="${MODEL_NAME##*/}"

# 실행할 데이터셋 목록
if [[ "${DATASET_ARG}" == "all" ]]; then
    TO_RUN=("${DATASETS[@]}")
else
    if [[ " ${DATASETS[*]} " =~ " ${DATASET_ARG} " ]]; then
        TO_RUN=("${DATASET_ARG}")
    else
        echo "ERROR: Unknown dataset '${DATASET_ARG}'. Use: 2wikimultihopqa, hotpotqa, musique, or all"
        exit 1
    fi
fi

for ds in "${TO_RUN[@]}"; do
    DATA_FILE="${RESULTS_BASE}/${ds}/triplets/${TRIPLET_MODEL}/triplets_train_sampled.json"
    OUTPUT_DIR="${OUTPUT_BASE}/infill/${MODEL_SHORT}/${ds}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "========== Dataset: ${ds} =========="
    echo "DATA_FILE=${DATA_FILE}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "MODEL_NAME=${MODEL_NAME}"
    python -u "${INFILL_PY}" \
        --model_name "${MODEL_NAME}" \
        --data_file "${DATA_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --question_strategy "triplet_only" \
        --infill_strategy "triplet_only" \
        --use_gold_only 1 \
        --max_trials 3 \
        --run_all_settings 1
done

echo ""
echo "========== All experiments completed =========="
echo "Infill results: ${OUTPUT_BASE}/infill/"
