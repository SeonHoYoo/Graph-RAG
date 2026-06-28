#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=4000MB
#SBATCH --job-name=vg_detect
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeseojeon/data/graph/graphqa/hs/logs/judge/vg_detection_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/graphqa/hs/logs/judge/vg_detection_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hyeseojeon/data/graph}"
PY="${PY:-python}"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CASES="${CASES:-}"
CASE_JUDGE="${CASE_JUDGE:-}"
OUTPUT="${OUTPUT:-}"

if [[ -z "${CASES}" ]]; then
    echo "Set CASES=/path/to/full_vghint_cases.json" >&2
    exit 2
fi
if [[ -z "${CASE_JUDGE}" ]]; then
    echo "Set CASE_JUDGE=/path/to/full_vghint_raw_case_judge.json" >&2
    exit 2
fi
if [[ -z "${OUTPUT}" ]]; then
    echo "Set OUTPUT=/path/to/vg_detection_output.json" >&2
    exit 2
fi

echo "============================================================"
echo "[vg_detection] cases      = ${CASES}"
echo "[vg_detection] case_judge = ${CASE_JUDGE}"
echo "[vg_detection] output     = ${OUTPUT}"
echo "============================================================"

"${PY}" "${PROJECT_ROOT}/graphqa/hs/scripts/analyze_vg_detection.py" \
    --cases "${CASES}" \
    --case-judge "${CASE_JUDGE}" \
    --output "${OUTPUT}"
