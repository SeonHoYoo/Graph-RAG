#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=hop
#SBATCH --exclude=n03,n04
#SBATCH --time=0-04:00:00
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/analysis/0311/perhop_%j.out
#SBATCH --error=../../logs/analysis/0311/perhop_%j.err

set -euo pipefail

REPO_ROOT="/home/hyeseojeon/data/Graph-RAG"
PERHOP_PY="${REPO_ROOT}/scripts/analysis/perhop.py"

source ~/data/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate graph

cd "${REPO_ROOT}"

mkdir -p /home/hyeseojeon/data/Graph-RAG/logs/analysis/0311

# -----------------------
# perhop config
# -----------------------
ROOT="/home/hyeseojeon/data/Graph-RAG/results/answer/0311"
OUT="/home/hyeseojeon/data/Graph-RAG/results/answer/0311/perhop_summary.txt"
PATTERN="*.json"

echo "========== perhop.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "pwd=$(pwd)"
echo "root=${ROOT}"
echo "out=${OUT}"
echo "pattern=${PATTERN}"
echo "perhop_py=${PERHOP_PY}"
echo "======================================"

cmd=(
    python -u "${PERHOP_PY}"
    --root "${ROOT}"
    --out "${OUT}"
    --pattern "${PATTERN}"
)

echo "Command: ${cmd[*]}"
"${cmd[@]}"
