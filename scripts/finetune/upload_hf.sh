#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --output=/home/hyeseojeon/data/graph/logs/finetune/upload_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/finetune/upload_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-02:00:00
#SBATCH --exclude=master

set -euo pipefail

REPO_ROOT="/data3/hyeseojeon/graph"
cd "${REPO_ROOT}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

HF_USER="${HF_USER:-doupari}"
FINETUNE_DIR="${FINETUNE_DIR:-${REPO_ROOT}/outputs/finetune}"

mkdir -p /home/hyeseojeon/data/graph/logs/finetune

# Upload all final models (skip dirs without adapter_config.json)
for model_dir in "${FINETUNE_DIR}"/*/; do
    model_name="$(basename "${model_dir}")"

    if [ ! -f "${model_dir}/adapter_config.json" ]; then
        echo "Skipping ${model_name}: no adapter_config.json"
        continue
    fi

    repo_id="${HF_USER}/${model_name//+/-}"
    echo "=========================================="
    echo "Uploading : ${model_name}"
    echo "  dir     : ${model_dir}"
    echo "  repo    : ${repo_id}"
    echo "=========================================="

    /data3/hyeseojeon/.conda/envs/sllm3/bin/python scripts/finetune/upload_hf.py \
        --model_dir "${model_dir}" \
        --repo_id "${repo_id}" \
        ${PRIVATE:+--private}

    echo "Done: https://huggingface.co/${repo_id}"
    echo ""
done

echo "All uploads complete."
