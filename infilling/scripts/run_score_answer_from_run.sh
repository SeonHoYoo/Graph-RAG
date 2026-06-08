#!/bin/bash
# 특정 answer 결과 디렉터리에서 EM/F1(by hop) 집계
#
# 사용법:
#   bash run_score_answer_from_run.sh
#   bash run_score_answer_from_run.sh Qwen2.5-7B-Instruct__openbook_nogold_tk510_job121147
#   bash run_score_answer_from_run.sh <MODEL_RUN_DIR> <OUT_PREFIX>
#
# 예시:
#   bash run_score_answer_from_run.sh Qwen2.5-7B-Instruct__openbook_nogold_tk510_job121147
#
# 출력:
#   /data3/seonhoyoo/graphcheck-qa/infilling/output/answer_em_f1_by_hop_<OUT_PREFIX>.csv
#   /data3/seonhoyoo/graphcheck-qa/infilling/output/answer_em_f1_by_hop_<OUT_PREFIX>.md

set -euo pipefail

BASE_DIR="/data3/seonhoyoo/graphcheck-qa"
INFILL_SCRIPTS="${BASE_DIR}/infilling/scripts"
AGGREGATE_PY="${INFILL_SCRIPTS}/aggregate_answer_metrics_by_hop.py"
OUTPUT_BASE="${BASE_DIR}/infilling/output"

# 기본값: 요청하신 run 디렉터리
MODEL_RUN_DIR="${1:-Qwen2.5-7B-Instruct__openbook_nogold_tk510_job121147}"
OUT_PREFIX="${2:-${MODEL_RUN_DIR}}"
ANSWER_ROOT="${OUTPUT_BASE}/answer/${MODEL_RUN_DIR}"

if [[ ! -d "${ANSWER_ROOT}" ]]; then
  echo "[ERROR] answer root not found: ${ANSWER_ROOT}"
  exit 1
fi

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck
cd "${INFILL_SCRIPTS}"

echo "=============================================="
echo "Aggregate answer EM/F1 by hop"
echo "Answer root: ${ANSWER_ROOT}"
echo "Output prefix: ${OUT_PREFIX}"
echo "=============================================="

python -u "${AGGREGATE_PY}" \
  --answer_root "${ANSWER_ROOT}" \
  --output_dir "${OUTPUT_BASE}" \
  --out_csv "${OUTPUT_BASE}/answer_em_f1_by_hop_${OUT_PREFIX}.csv" \
  --out_md "${OUTPUT_BASE}/answer_em_f1_by_hop_${OUT_PREFIX}.md"

echo ""
echo "Done."
echo "CSV: ${OUTPUT_BASE}/answer_em_f1_by_hop_${OUT_PREFIX}.csv"
echo "MD : ${OUTPUT_BASE}/answer_em_f1_by_hop_${OUT_PREFIX}.md"
