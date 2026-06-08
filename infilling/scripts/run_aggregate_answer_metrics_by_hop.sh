#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=0-04:00:00
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=2
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_aggregate_answer_metrics_by_hop_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_aggregate_answer_metrics_by_hop_%j.err

# 기존 answer 결과를 기반으로 dataset x strategy x hop EM/F1 집계
set -euo pipefail

BASE_DIR="/data3/seonhoyoo/graphcheck-qa"
INFILL_SCRIPTS="${BASE_DIR}/infilling/scripts"
PY_SCRIPT="${INFILL_SCRIPTS}/aggregate_answer_metrics_by_hop.py"

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd "${INFILL_SCRIPTS}"

# 인자:
#   $1: answer_root (기본: Qwen2.5-7B-Instruct 결과 경로)
#   $2: output_dir  (기본: /data3/seonhoyoo/graphcheck-qa/infilling/output)
ANSWER_ROOT="${1:-/data3/seonhoyoo/graphcheck-qa/infilling/output/answer/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${2:-/data3/seonhoyoo/graphcheck-qa/infilling/output}"

python -u "${PY_SCRIPT}" \
  --answer_root "${ANSWER_ROOT}" \
  --output_dir "${OUTPUT_DIR}"
