#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/extract_triplets_%j.log
#SBATCH --error=./logs/extract_triplets_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd /data3/seonhoyoo/graphcheck-qa

export HF_HOME=/data3/seonhoyoo/.cache/huggingface
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" ./logs

pip install --upgrade transformers>=4.37.0 --quiet

# 사용법:
#   sbatch extract_triplets.sh
#   sbatch extract_triplets.sh hotpotqa
#   sbatch extract_triplets.sh all Qwen/Qwen2.5-14B-Instruct
#   sbatch extract_triplets.sh hotpotqa Qwen/Qwen2.5-7B-Instruct train_sampled 10 open-book triplets_train_sampled_open-book_top10.json
DATASET_ARG="${1:-all}"
construct_model="${2:-Qwen/Qwen2.5-7B-Instruct}"
input_fname="${3:-train_sampled}"
bm25_top_k="${4:-5}"
setting="${5:-open-book+gold}"
output_fname="${6:-}"

ALL_DATASETS=("musique" "hotpotqa" "2wikimultihopqa")
if [[ "${DATASET_ARG}" == "all" ]]; then
    datasets=("${ALL_DATASETS[@]}")
else
    if [[ " ${ALL_DATASETS[*]} " =~ " ${DATASET_ARG} " ]]; then
        datasets=("${DATASET_ARG}")
    else
        echo "ERROR: Unknown dataset '${DATASET_ARG}'. Use: 2wikimultihopqa, hotpotqa, musique, or all"
        exit 1
    fi
fi

for dataset in "${datasets[@]}"; do
    echo "============================================"
    echo "  Dataset: ${dataset}"
    echo "  Model:   ${construct_model}"
    echo "  Setting: ${setting}"
    if [[ -n "${output_fname}" ]]; then
        echo "  Output:  ${output_fname}"
    fi
    echo "============================================"

    output_args=()
    if [[ -n "${output_fname}" ]]; then
        output_args=(--output_filename "${output_fname}")
    fi

    python -u extract_triplets.py \
        --dataset "${dataset}" \
        --input_filename "${input_fname}.json" \
        --construct_model_name "${construct_model}" \
        --bm25_top_k "${bm25_top_k}" \
        --setting "${setting}" \
        "${output_args[@]}" \
        --checkpoint_every 10

    echo ""
    echo ">>> ${dataset} done."
    echo ""
done

echo "All datasets finished."
