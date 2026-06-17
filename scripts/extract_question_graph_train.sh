#!/bin/bash
#SBATCH --job-name=qgraph_train
#SBATCH --output=/home/hyeseojeon/data/graph/logs/train/0528/question-%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/logs/train/0528/question-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=n01,n02
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=2

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

export HF_HOME="/home/hyeseojeon/data"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="/home/hyeseojeon/data/graph"
MODEL_DIR="${REPO_ROOT}/outputs/finetune/Llama-3.2-3B-Instruct-question+think+search"
OUTPUT_DIR="${REPO_ROOT}/datasets/train/graph"

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs/train/0528"

python -u "${REPO_ROOT}/scripts/extract_question_graph_train.py" \
    --input_file      "${REPO_ROOT}/datasets/train/raw/hotpot_500.json" \
    --dataset         hotpotqa \
    --model_dir       "${MODEL_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --output_filename hotpot_question_graph.json \
    --hf_home         /home/hyeseojeon/data

python -u "${REPO_ROOT}/scripts/extract_question_graph_train.py" \
    --input_file      "${REPO_ROOT}/datasets/train/raw/2wiki_500.json" \
    --dataset         2wikimultihopqa \
    --model_dir       "${MODEL_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --output_filename 2wiki_question_graph.json \
    --hf_home         /home/hyeseojeon/data

python -u "${REPO_ROOT}/scripts/extract_question_graph_train.py" \
    --input_file      "${REPO_ROOT}/datasets/train/raw/musique_500.json" \
    --dataset         musique \
    --model_dir       "${MODEL_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --output_filename musique_question_graph.json \
    --hf_home         /home/hyeseojeon/data
