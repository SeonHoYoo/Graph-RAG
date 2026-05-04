#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=40000MB
#SBATCH --job-name=r1_neural_step
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeseojeon/data/graph/logs/latent/0427/searchr1_neural_step_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/logs/latent/0427/searchr1_neural_step_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck
export PYTHONNOUSERSITE=1

SCRIPT_DIR="/home/hyeseojeon/data/graph/scripts/latent"
REPO_ROOT="/home/hyeseojeon/data/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
mkdir -p /home/hyeseojeon/data/graph/logs/latent/0427

input_file_path="/home/hyeseojeon/data/graph/datasets/hotpotqa/claims/train_sampled.json"
model_name="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
retriever_url="http://127.0.0.1:8000/retrieve"
output_dir="/home/hyeseojeon/data/graph/results/latent/0427/hotpotqa_searchr1_generate_latent"

max_samples=1000
max_turns=5
max_new_tokens=500
temperature=1.0
topk=3
dtype="float16"
verbose=false

layers=(4 16 20 24 28)
boundaries=("<think>" "</think>" "<search>" "</search>" "<information>" "</information>" "<answer>" "</answer>")
think_token_offsets=(1 5 10 20)
dense_think_stride=5

echo "========== searchr1_neuro_step_latents.sh Config =========="
echo "input_file_path=${input_file_path}"
echo "model_name=${model_name}"
echo "retriever_url=${retriever_url}"
echo "output_dir=${output_dir}"
echo "max_samples=${max_samples}"
echo "max_turns=${max_turns}"
echo "max_new_tokens=${max_new_tokens}"
echo "temperature=${temperature}"
echo "topk=${topk}"
echo "dtype=${dtype}"
echo "verbose=${verbose}"
echo "layers=${layers[*]}"
echo "boundaries=${boundaries[*]}"
echo "think_token_offsets=${think_token_offsets[*]}"
echo "dense_think_stride=${dense_think_stride}"
echo "====================================================================="

cmd=(
    python -u scripts/latent/searchr1_neuro_step_latents.py
    --input_file_path "${input_file_path}"
    --model_name "${model_name}"
    --retriever_url "${retriever_url}"
    --output_dir "${output_dir}"
    --max_samples "${max_samples}"
    --max_turns "${max_turns}"
    --max_new_tokens "${max_new_tokens}"
    --temperature "${temperature}"
    --topk "${topk}"
    --dtype "${dtype}"
)

for layer in "${layers[@]}"; do
    cmd+=(--layer "${layer}")
done

for boundary in "${boundaries[@]}"; do
    cmd+=(--boundary "${boundary}")
done

for offset in "${think_token_offsets[@]}"; do
    cmd+=(--think_token_offset "${offset}")
done

if [ -n "${dense_think_stride}" ]; then
    cmd+=(--dense_think_stride "${dense_think_stride}")
fi

if [ "${verbose}" = true ]; then
    cmd+=(--verbose)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
