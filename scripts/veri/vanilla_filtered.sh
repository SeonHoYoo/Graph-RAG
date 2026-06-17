#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --exclude=master,n01,n02,n03
#SBATCH --mem=40000MB
#SBATCH --job-name=vanilla_filtered
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeseojeon/data/graph/logs/vanilla/0516/vanilla_filtered_%x_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/vanilla/0516/vanilla_filtered_%x_%j.err

set -euo pipefail

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p logs/searchr1/0516

# Usage:
#   DATASET=hotpotqa        INPUT_FILE=datasets/hotpotqa/filtered/train.json        sbatch --job-name=hotpotqa   vanilla_filtered.sh
#   DATASET=2wikimultihopqa INPUT_FILE=datasets/2wikimultihopqa/filtered/train.json sbatch --job-name=2wiki       vanilla_filtered.sh
#   DATASET=musique         INPUT_FILE=datasets/musique/filtered/train.json         sbatch --job-name=musique     vanilla_filtered.sh

DATASET="${DATASET:?DATASET env var required}"
INPUT_FILE="${INPUT_FILE:?INPUT_FILE env var required}"
base_model="${MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"
evidence_setting="${EVIDENCE_SETTING:-open-book}"
searchr1_max_turns="${MAX_TURNS:-5}"
searchr1_top_k="${TOP_K:-3}"
max_samples="${MAX_SAMPLES:-}"
output_dir="${OUTPUT_DIR:-results/vanilla/0516}"
output_filename="${OUTPUT_FILENAME:-${DATASET}_vanilla_searchr1_filtered.json}"

echo "========== vanilla_filtered.sh =========="
echo "DATASET        = ${DATASET}"
echo "INPUT_FILE     = ${INPUT_FILE}"
echo "base_model     = ${base_model}"
echo "evidence_setting = ${evidence_setting}"
echo "output_dir     = ${output_dir}"
echo "output_filename = ${output_filename}"
echo "========================================="

cmd=(
    /data3/hyeseojeon/miniconda3/bin/python -u scripts/veri/vanilla.py
    --dataset "${DATASET}"
    --input_file_path "${INPUT_FILE}"
    --base_model_name "${base_model}"
    --evidence_setting "${evidence_setting}"
    --use_searchr1
    --searchr1_top_k "${searchr1_top_k}"
    --searchr1_max_turns "${searchr1_max_turns}"
    --seed "${SEED:-42}"
    --output_dir "${output_dir}"
    --output_filename "${output_filename}"
)

if [[ -n "${max_samples}" ]]; then
    cmd+=(--max_samples "${max_samples}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
