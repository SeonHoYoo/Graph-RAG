#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=24000MB
#SBATCH --job-name=veri-evidence
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/veri/0409/veri_evidence_%j.log
#SBATCH --error=../../logs/veri/0409/veri_evidence_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

evidence_source="retrieved"    # retrieved | gold
verification_target="triplet"  # triplet | raw_question
model_name="gpt-4.1-mini"

retrieval_input_file="/home/hyeseojeon/data/graph/results/vanilla/0407(open-book)/musique_vanilla_searchr1_128617_1000.json"
graph_input_file="/home/hyeseojeon/data/graph/results/graph/0409/musique_question_graph.json"


gold_input_file_path="/home/hyeseojeon/data/graph/results/graph/0409/musique_question_graph.json"


output_dir="/home/hyeseojeon/data/graph/results/veri_evidence/0411"
output_filename="musique_veri_evidence.json"
max_samples=1000
max_trials=3
max_documents_chars=1200000

echo "========== veri_evidence.sh Config =========="
echo "evidence_source=${evidence_source}"
echo "verification_target=${verification_target}"
echo "model_name=${model_name}"
echo "retrieval_input_file=${retrieval_input_file}"
echo "gold_input_file_path=${gold_input_file_path}"
echo "graph_input_file=${graph_input_file}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename}"
echo "max_samples=${max_samples}"
echo "max_trials=${max_trials}"
echo "max_documents_chars=${max_documents_chars}"
echo "============================================"

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

echo "Command: ${cmd[*]}"
"${cmd[@]}"
