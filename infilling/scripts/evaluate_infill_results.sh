#!/bin/bash
# 평가만 수행 (모델 호출 없음). predicted_answer가 있는 JSON 필요.
# answer.py 먼저 실행 후 사용: sbatch run_answer_and_aggregate.sh
#
# 또는 answer_*.json이 이미 있으면:
#   sbatch evaluate_infill_results.sh

BASE_DIR="/data3/seonhoyoo/graphcheck-qa"
INFILL_SCRIPTS="${BASE_DIR}/infilling/scripts"
OUTPUT_BASE="${BASE_DIR}/infilling/output"

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd "${INFILL_SCRIPTS}"

python -u "${INFILL_SCRIPTS}/evaluate_infill_results.py" \
    --output_dir "${OUTPUT_BASE}"
