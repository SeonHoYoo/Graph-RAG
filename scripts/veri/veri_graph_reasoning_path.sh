#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=40000MB
#SBATCH --job-name=graph_path
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/veri/0409/reasoning_path_%j.log
#SBATCH --error=../../logs/veri/0409/reasoning_path_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN="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"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

SCRIPT_DIR="/data3/hyeseojeon/graph/scripts/veri"
REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

input_file_path="/home/hyeseojeon/data/graph/results/graph/0409/musique_question_graph.json"
dataset="2wikimultihopqa"  # musique | hotpotqa | 2wikimultihopqa
base_model_name="gpt-4.1-mini" # Qwen/Qwen2.5-7B-Instruct | meta-llama/Llama-3.1-8B-Instruct | gpt-4.1-mini
bm25_top_k=5
use_verification=true
use_verifier=false
use_binding_extraction=true
searchr1_top_k=3
searchr1_max_turns=5
evidence_setting="open-book+gold" # open-book | open-book+gold | gold
graph_prompt_mode="stepwise" # stepwise | full_graph
verification_top_k=-1
max_samples=1000
use_total_search_results=true

output_dir="/home/hyeseojeon/data/graph/results/veri/0410"
output_filename="2wiki_graph_reasoning_path.json"

echo "========== veri_graph_reasoning_path.sh Config =========="
echo "input_file_path=${input_file_path}"
echo "dataset=${dataset}"
echo "base_model_name=${base_model_name}"
echo "bm25_top_k=${bm25_top_k}"
echo "use_verification=${use_verification}"
echo "use_verifier=${use_verifier}"
echo "use_binding_extraction=${use_binding_extraction}"
echo "searchr1_top_k=${searchr1_top_k}"
echo "searchr1_max_turns=${searchr1_max_turns}"
echo "evidence_setting=${evidence_setting}"
echo "graph_prompt_mode=${graph_prompt_mode}"
echo "verification_top_k=${verification_top_k}"
echo "max_samples=${max_samples}"
echo "use_total_search_results=${use_total_search_results}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename}"
echo "========================================================="

cmd=(
    python -u scripts/veri/veri_graph_reasoning_path.py
    --input_file_path "${input_file_path}"
    --dataset "${dataset}"
    --base_model_name "${base_model_name}"
    --bm25_top_k "${bm25_top_k}"
    --searchr1_top_k "${searchr1_top_k}"
    --searchr1_max_turns "${searchr1_max_turns}"
    --evidence_setting "${evidence_setting}"
    --graph_prompt_mode "${graph_prompt_mode}"
    --verification_top_k "${verification_top_k}"
    --max_samples "${max_samples}"
    --output_dir "${output_dir}"
    --output_filename "${output_filename}"
    --use_searchr1
    --nudge_searchr1
)

if [ "${use_verification}" = true ]; then
    cmd+=(--use_verification)
fi

if [ "${use_verifier}" = true ]; then
    cmd+=(--use_verifier)
fi

if [ "${use_binding_extraction}" = true ]; then
    cmd+=(--use_binding_extraction)
fi

if [ "${use_total_search_results}" = true ]; then
    cmd+=(--use_total_search_results)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
