#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=master
#SBATCH --mem=40000MB
#SBATCH --job-name=graph_extract
#SBATCH --cpus-per-task=1
#SBATCH --output=../logs/graph/0409/extract_qgraph_%j.log
#SBATCH --error=../logs/graph/0409/extract_qgraph_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN="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"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

REPO_ROOT="/home/hyeseojeon/data/graph"
cd "${REPO_ROOT}"

input_file_path="/home/hyeseojeon/data/graph/datasets/musique/claims/train_sampled.json"
model_name="gpt-4.1-mini" # Qwen/Qwen2.5-7B-Instruct | meta-llama/Llama-3.1-8B-Instruct | gpt-4.1-mini | Qwen/Qwen2.5-14B-Instruct
max_samples=1000

output_dir="/home/hyeseojeon/data/graph/results/graph/0409"
output_filename="musique_question_graph.json"

echo "========== extract_question_graph.sh Config =========="
echo "input_file_path=${input_file_path}"
echo "model_name=${model_name}"
echo "max_samples=${max_samples}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename}"
echo "====================================================="

cmd=(
    python -u /home/hyeseojeon/data/graph/scripts/extract_question_graph.py
    --input_file_path "${input_file_path}"
    --model_name "${model_name}"
    --output_dir "${output_dir}"
    --output_filename "${output_filename}"
    --max_samples "${max_samples}"
)

echo "Command: ${cmd[*]}"
"${cmd[@]}"
