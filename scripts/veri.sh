#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=24000MB
#SBATCH --job-name=veri
#SBATCH --cpus-per-task=1
#SBATCH --output=../logs/veri/0411/veri_%j.log
#SBATCH --error=../logs/veri/0411/veri_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

run_mode="evidence"             # evidence | pipeline

evidence_source="retrieved"     # retrieved | gold
verification_target="triplet"   # triplet | raw_question
model_name="gpt-4.1-mini"
retrieval_input_file="/data3/hyeseojeon/graph/results/vanilla/0410/hotpotqa_vanilla_bm25_open-book+gold_retriever_129225_500.json"
gold_input_file_path="/data3/hyeseojeon/graph/datasets/hotpotqa/claims/train_sampled.json"
graph_input_file="/data3/hyeseojeon/graph/__graphcheck-qa-2/results/hotpotqa/triplets/gpt-4.1/hotpotqa_triplets_train_sampled.json"
max_trials=3
max_documents_chars=12000

input_file_path="/data3/hyeseojeon/graph/datasets/hotpotqa/claims/train_sampled.json"
construct_model="Qwen/Qwen2.5-7B-Instruct"
base_model="Qwen/Qwen2.5-7B-Instruct"
reasoning_mode="searchr1_graph"   # standard | searchr1_graph
evidence_setting="open-book"      # open-book | open-book+gold | gold
verification_source="triplet"     # triplet | doc
use_searchr1=true
nudge_searchr1=true
use_total_search_results=true
bm25_top_k=5
verification_top_k=-1

max_samples=10
output_dir="/home/hyeseojeon/data/graph/results/veri/0411"
output_filename="hotpotqa_veri.json"

echo "=============== veri.sh Config ==============="
echo "run_mode=${run_mode}"
echo "evidence_source=${evidence_source}"
echo "verification_target=${verification_target}"
echo "model_name=${model_name}"
echo "retrieval_input_file=${retrieval_input_file}"
echo "gold_input_file_path=${gold_input_file_path}"
echo "graph_input_file=${graph_input_file}"
echo "input_file_path=${input_file_path}"
echo "construct_model=${construct_model}"
echo "base_model=${base_model}"
echo "reasoning_mode=${reasoning_mode}"
echo "evidence_setting=${evidence_setting}"
echo "verification_source=${verification_source}"
echo "use_searchr1=${use_searchr1}"
echo "nudge_searchr1=${nudge_searchr1}"
echo "use_total_search_results=${use_total_search_results}"
echo "bm25_top_k=${bm25_top_k}"
echo "verification_top_k=${verification_top_k}"
echo "max_samples=${max_samples}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename}"
echo "=============================================="

if [ "${run_mode}" = "evidence" ]; then
    cmd=(
        python -u scripts/veri/veri_evidence.py
        --evidence_source "${evidence_source}"
        --output_dir "${output_dir}"
        --output_filename "${output_filename}"
        --model_name "${model_name}"
        --verification_target "${verification_target}"
        --max_samples "${max_samples}"
        --max_trials "${max_trials}"
        --max_documents_chars "${max_documents_chars}"
    )

    if [ "${evidence_source}" = "gold" ]; then
        cmd+=(--gold_input_file_path "${gold_input_file_path}")
    else
        cmd+=(--retrieval_input_file "${retrieval_input_file}")
        cmd+=(--graph_input_file "${graph_input_file}")
    fi
else
    cmd=(
        python -u scripts/veri.py
        --input_file_path "${input_file_path}"
        --construct_model_name "${construct_model}"
        --base_model_name "${base_model}"
        --reasoning_mode "${reasoning_mode}"
        --evidence_setting "${evidence_setting}"
        --verification_source "${verification_source}"
        --bm25_top_k "${bm25_top_k}"
        --verification_top_k "${verification_top_k}"
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
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
