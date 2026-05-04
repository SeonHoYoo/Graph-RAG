#!/bin/bash
#SBATCH --job-name=bleu
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/bleu_%j.log
#SBATCH --error=../../logs/bleu_%j.err

source ~/data/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate graph

# 작업 디렉토리로 이동
cd /home/hyeseojeon/data/Graph-RAG

# -----------------------
# BLEU 검증 설정
# -----------------------
DATA_FILE="/home/hyeseojeon/data/Graph-RAG/results/2wikimultihopqa/graph_infill/train_sampled_multihop_graphcheck_triplets_115374_500_Qwen2.5-7B-Instruct_triplet_only_triplet_only_115726.json"
OUTPUT_DIR="/home/hyeseojeon/data/Graph-RAG/results/2wikimultihopqa/verification/bleu"

BLEU_N=1
TAU=0.5
# MAX_SAMPLES=100

mkdir -p "${OUTPUT_DIR}"
INPUT_STEM="$(basename "${DATA_FILE}" .json)"
JOB_ID="${SLURM_JOB_ID:-local}"
TAU_TAG="${TAU/./p}"
OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_STEM}_bleu${BLEU_N}_tau${TAU_TAG}_${JOB_ID}.json"

echo "========== bleu.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "data_file=${DATA_FILE}"
echo "output_file=${OUTPUT_FILE}"
echo "bleu_n=${BLEU_N}"
echo "tau=${TAU}"
echo "========================================"

cmd=(
    python -u scripts/verification/bleu.py
    --data_file "${DATA_FILE}"
    --output_file "${OUTPUT_FILE}"
    --bleu_n "${BLEU_N}"
    --tau "${TAU}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
