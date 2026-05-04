#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=master
#SBATCH --mem=40000MB
#SBATCH --job-name=veri_gold
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/veri/0409/veri_gold_%j.log
#SBATCH --error=../../logs/veri/0409/veri_gold_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

SCRIPT_DIR="/data3/hyeseojeon/graph/scripts/veri"
REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

input_file_path="/home/hyeseojeon/data/graph/datasets/musique/claims/train_sampled.json"
model_name="gpt-4.1-mini" # Qwen/Qwen2.5-7B-Instruct | gpt-4.1-mini
max_samples=1000
max_trials=3

output_dir="/home/hyeseojeon/data/graph/results/veri/0409"
output_filename="musique_veri_gold_doc.json"

echo "========== veri_gold_evidence.sh Config =========="
echo "input_file_path=${input_file_path}"
echo "model_name=${model_name}"
echo "max_samples=${max_samples}"
echo "max_trials=${max_trials}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename}"
echo "=================================================="

cmd=(
    python -u scripts/veri/veri_gold_evidence.py
    --model_name "${model_name}"
    --input_file_path "${input_file_path}"
    --output_dir "${output_dir}"
    --output_filename "${output_filename}"
    --max_samples "${max_samples}"
    --max_trials "${max_trials}"
)

echo "Command: ${cmd[*]}"
"${cmd[@]}"
