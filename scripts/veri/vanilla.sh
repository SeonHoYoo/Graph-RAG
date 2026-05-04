#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=40000MB
#SBATCH --job-name=graph_search
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/searchr1/0407/vanilla_%j.log
#SBATCH --error=../../logs/searchr1/0407/vanilla_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

SCRIPT_DIR="/data3/hyeseojeon/graph/scripts/veri"
REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

input_file_path="/home/hyeseojeon/data/graph/datasets/musique/claims/train_sampled.json"

base_model="Qwen/Qwen2.5-7B-Instruct"

evidence_setting="open-book+gold" # open-book | open-book+gold | gold
use_searchr1=false
nudge_searchr1=false
searchr1_max_turns=5
use_total_search_results=true
bm25_top_k=5
max_samples=1000

output_dir="/home/hyeseojeon/data/graph/results/vanilla/0410"
output_filename="musique_vanilla_bm25.json"

echo "========== vanilla.sh Config =========="
echo "input_file_path=${input_file_path}"
echo "base_model=${base_model}"
echo "evidence_setting=${evidence_setting}"
echo "use_searchr1=${use_searchr1}"
echo "nudge_searchr1=${nudge_searchr1}"
echo "searchr1_max_turns=${searchr1_max_turns}"
echo "use_total_search_results=${use_total_search_results}"
echo "bm25_top_k=${bm25_top_k}"
echo "max_samples=${max_samples}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename}"
echo "======================================"

cmd=(
    python -u scripts/veri/vanilla.py
    --input_file_path "${input_file_path}"
    --base_model_name "${base_model}"
    --evidence_setting "${evidence_setting}"
    --bm25_top_k "${bm25_top_k}"
    --searchr1_max_turns "${searchr1_max_turns}"
    --max_samples "${max_samples}"
    --output_dir "${output_dir}"
    --output_filename "${output_filename}"
)

if [ "${use_searchr1}" = true ]; then
    cmd+=(--use_searchr1)
fi

if [ "${nudge_searchr1}" = true ]; then
    cmd+=(--nudge_searchr1)
fi

if [ "${use_total_search_results}" = true ]; then
    cmd+=(--use_total_search_results)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
