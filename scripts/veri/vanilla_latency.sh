#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --exclude=n03,n04,master
#SBATCH --mem=40000MB
#SBATCH --job-name=latency
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeseojeon/data/graph/logs/searchr1/0515/vanilla_latency_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/searchr1/0515/vanilla_latency_%j.err

set -euo pipefail

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" logs/searchr1/0515

input_file_path="${INPUT_FILE:-/home/hyeseojeon/data/graph/datasets/musique/claims/train_sampled.json}"
base_model="${MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"
evidence_setting="${EVIDENCE_SETTING:-open-book}"
searchr1_max_turns="${MAX_TURNS:-5}"
searchr1_top_k="${TOP_K:-3}"
max_samples="${MAX_SAMPLES:-100}"     # 소수만 돌려서 latency 측정
output_dir="${OUTPUT_DIR:-results/vanilla/0515}"
output_filename="${OUTPUT_FILENAME:-musique_vanilla_searchr1_latency.json}"

echo "========== vanilla_latency.sh Config =========="
echo "input_file_path=${input_file_path}"
echo "base_model=${base_model}"
echo "evidence_setting=${evidence_setting}"
echo "searchr1_max_turns=${searchr1_max_turns}"
echo "max_samples=${max_samples}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename}"
echo "==============================================="

cmd=(
    python -u scripts/veri/vanilla.py
    --input_file_path "${input_file_path}"
    --base_model_name "${base_model}"
    --evidence_setting "${evidence_setting}"
    --use_searchr1
    --searchr1_top_k "${searchr1_top_k}"
    --searchr1_max_turns "${searchr1_max_turns}"
    --max_samples "${max_samples}"
    --shuffle
    --seed "${SEED:-42}"
    --output_dir "${output_dir}"
    --output_filename "${output_filename}"
)

echo "Command: ${cmd[*]}"
"${cmd[@]}"
