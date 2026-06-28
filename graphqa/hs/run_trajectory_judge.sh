#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=0-02:00:00
#SBATCH --nodelist=n03
#SBATCH --mem=8000MB
#SBATCH --job-name=on-judge
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeseojeon/data/graph/graphqa/hs/logs/online_feedback/trajectory_judge_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/graphqa/hs/logs/online_feedback/trajectory_judge_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hyeseojeon/data/graph}"
PY="${PY:-python}"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CASES="${CASES:-}"
if [[ -z "${CASES}" ]]; then
    echo "Set CASES=/path/to/online_feedback_*_cases_*.json" >&2
    exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/graphqa/hs/outputs/online_feedback}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}_trajectory_judge}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4.1-mini-2025-04-14}"
SKIML_API_BASE="${SKIML_API_BASE:-${JUDGE_BASE_URL:-}}"
MAX_CASES="${MAX_CASES:-0}"

ARGS=()
if [[ -n "${SKIML_API_BASE}" ]]; then
    export SKIML_API_BASE
    ARGS+=(--base-url "${SKIML_API_BASE}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi
if [[ "${NO_PROGRESS:-0}" == "1" ]]; then
    ARGS+=(--no-progress)
fi
if [[ -n "${CASE_INDICES:-}" ]]; then
    read -r -a CASE_INDEX_ARR <<< "${CASE_INDICES}"
    ARGS+=(--case-indices "${CASE_INDEX_ARR[@]}")
fi

echo "============================================================"
echo "[trajectory_judge] cases         = ${CASES}"
echo "[trajectory_judge] output_dir    = ${OUTPUT_DIR}"
echo "[trajectory_judge] output_prefix = ${OUTPUT_PREFIX}"
echo "[trajectory_judge] judge_model   = ${JUDGE_MODEL}"
echo "[trajectory_judge] skiml_base    = ${SKIML_API_BASE:-default from model_library/openai_client.py}"
echo "[trajectory_judge] max_cases     = ${MAX_CASES}"
echo "============================================================"

"${PY}" "${PROJECT_ROOT}/graphqa/hs/scripts/judge_case_trajectory.py" \
    --cases "${CASES}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --model "${JUDGE_MODEL}" \
    --max-cases "${MAX_CASES}" \
    "${ARGS[@]}"
