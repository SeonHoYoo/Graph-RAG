#!/bin/bash
# 단일 프로세스/단일 모델 로드로 Triplet → Infill → Answer 전체 파이프라인
#
# 사용법:
#   sbatch run_full_pipeline.sh
#   sbatch run_full_pipeline.sh 2wikimultihopqa
#   sbatch run_full_pipeline.sh all Qwen/Qwen2.5-14B-Instruct
#
# 전제: retriever 서버(기본 localhost:8000)가 실행 중이어야 함

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_full_pipeline_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_full_pipeline_%j.err

set -e
BASE_DIR="/data3/seonhoyoo/graphcheck-qa"
INFILL_SCRIPTS="${BASE_DIR}/infilling/scripts"
PIPELINE_PY="${INFILL_SCRIPTS}/scripts/single_model_pipeline.py"
AGGREGATE_PY="${INFILL_SCRIPTS}/aggregate_answer_metrics_by_hop.py"
OUTPUT_BASE="${BASE_DIR}/infilling/output"

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck
cd "${INFILL_SCRIPTS}"

export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

DATASET_ARG="${1:-musique}"
MODEL_NAME="${2:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SHORT="${MODEL_NAME##*/}"
TRIPLET_TOPK_LIST="${3:-5,10}"
BASELINE_TOPK_LIST="${4:-5,10}"
JOB_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_TAG="openbook_nogold_tk${TRIPLET_TOPK_LIST//,/}_job${JOB_TAG}"
MODEL_OUTPUT_DIR="${MODEL_SHORT}__${OUTPUT_TAG}"

echo "=============================================="
echo "Full Pipeline: Single-Process Triplet → Infill → Answer"
echo "Dataset: ${DATASET_ARG}, Model: ${MODEL_NAME}"
echo "Setting: open-book (no gold), triplet top-k=${TRIPLET_TOPK_LIST}, baseline top-k=${BASELINE_TOPK_LIST}"
echo "Output tag: ${OUTPUT_TAG}"
echo "=============================================="

python -u "${PIPELINE_PY}" \
  --dataset "${DATASET_ARG}" \
  --model_name "${MODEL_NAME}" \
  --setting "open-book" \
  --triplet_top_k_list "${TRIPLET_TOPK_LIST}" \
  --baseline_top_k_list "${BASELINE_TOPK_LIST}" \
  --output_tag "${OUTPUT_TAG}" \
  --ent_exist_flag "all" \
  --max_trials 3

echo ""
echo "========== Aggregate Answer EM/F1 by strategy/hop =========="
python -u "${AGGREGATE_PY}" \
  --answer_root "${OUTPUT_BASE}/answer/${MODEL_OUTPUT_DIR}" \
  --output_dir "${OUTPUT_BASE}" \
  --out_csv "${OUTPUT_BASE}/answer_em_f1_by_hop_${MODEL_OUTPUT_DIR}.csv" \
  --out_md "${OUTPUT_BASE}/answer_em_f1_by_hop_${MODEL_OUTPUT_DIR}.md"

echo ""
echo "========== Pipeline Complete =========="
echo "Results: ${OUTPUT_BASE}/infill/${MODEL_OUTPUT_DIR}/ , ${OUTPUT_BASE}/answer/${MODEL_OUTPUT_DIR}/"
echo "Answer Summary CSV: ${OUTPUT_BASE}/answer_em_f1_by_hop_${MODEL_OUTPUT_DIR}.csv"
