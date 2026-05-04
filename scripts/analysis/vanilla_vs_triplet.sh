#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=master
#SBATCH --mem=40000MB
#SBATCH --job-name=veri_triplet
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/veri/0413/vanilla_vs_triplet_%j.log
#SBATCH --error=../../logs/veri/0413/vanilla_vs_triplet_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN="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"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

SCRIPT_DIR="/home/hyeseojeon/data/graph/scripts/analysis"

input_file_1="/home/hyeseojeon/data/graph/results/vanilla/0407(open-book)/musique_vanilla_searchr1_128617_1000.json"
input_file_2="/home/hyeseojeon/data/graph/results/graph/0409/musique_question_graph.json"
base_model_name="gpt-4.1-mini" # Qwen/Qwen2.5-7B-Instruct | meta-llama/Llama-3.1-8B-Instruct | gpt-4.1-mini
reasoning_source="think+subquery" # think | subquery | think+subquery
max_samples=1000
output_dir="/home/hyeseojeon/data/graph/results/analysis/0413"
output_filename="musique_vanilla_vs_triplet_think+subquery_openbook.json"

cmd=(
  python -u "${SCRIPT_DIR}/vanilla_vs_triplet.py"
  --input_file_1 "${input_file_1}"
  --input_file_2 "${input_file_2}"
  --base_model_name "${base_model_name}"
  --reasoning_source "${reasoning_source}"
  --output_dir "${output_dir}"
  --output_filename "${output_filename}"
)

if [[ "${max_samples}" -gt 0 ]]; then
  cmd+=(--max_samples "${max_samples}")
fi

"${cmd[@]}"
