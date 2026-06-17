#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=40000MB
#SBATCH --job-name=reason_train
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hyeseojeon/data/graph/logs/train/0528/reason_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/logs/train/0528/reason_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

RAW_DIR="/home/hyeseojeon/data/graph/datasets/train/raw"
OUTPUT_DIR="/home/hyeseojeon/data/graph/datasets/train/reason"

mkdir -p "${OUTPUT_DIR}" /home/hyeseojeon/data/graph/logs/train/0528

evidence_setting="open-book"
searchr1_max_turns=5
bm25_top_k=5

for entry in \
    "hotpotqa    ${RAW_DIR}/hotpot_500.json    hotpot_500" \
    "2wikimultihopqa ${RAW_DIR}/2wiki_500.json 2wiki_500" \
    "musique     ${RAW_DIR}/musique_500.json   musique_500"
do
    dataset=$(echo $entry | awk '{print $1}')
    input_file=$(echo $entry | awk '{print $2}')
    out_name=$(echo $entry | awk '{print $3}')

    echo "===== ${dataset} ====="
    python -u scripts/veri/vanilla.py \
        --input_file_path   "${input_file}" \
        --dataset           "${dataset}" \
        --evidence_setting  "${evidence_setting}" \
        --bm25_top_k        "${bm25_top_k}" \
        --searchr1_max_turns "${searchr1_max_turns}" \
        --output_dir        "${OUTPUT_DIR}" \
        --output_filename   "${out_name}.jsonl" \
        --use_searchr1 \
        --use_total_search_results
done
