#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=answer
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/answer_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/answer_%j.err

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd /data3/seonhoyoo/graphcheck-qa/infilling/scripts

export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# -----------------------
# 답변 생성 config
# -----------------------
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-14B-Instruct | gpt-4o-mini
OUTPUT_BASE="/data3/seonhoyoo/graphcheck-qa/infilling/output"
INFILL_DIR="${OUTPUT_BASE}/infill"
ANSWER_DIR="${OUTPUT_BASE}/answer"
DATASETS=("2wikimultihopqa" "hotpotqa" "musique")

ENT_EXIST_FLAG="all"     # all | false((ENT)가 모두 infill된 데이터만)
MAX_TRIALS=3

echo "========== answer.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "model_name=${MODEL_NAME}"
echo "infill_dir=${INFILL_DIR}"
echo "answer_dir=${ANSWER_DIR}"
echo "datasets=${DATASETS[*]}"
echo "ent_exist_flag=${ENT_EXIST_FLAG}"
echo "max_trials=${MAX_TRIALS}"
echo "====================================="

if [[ "${MODEL_NAME}" == gpt* ]] && [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set for GPT model."
    exit 1
fi

for ds in "${DATASETS[@]}"; do
    infill_path="${INFILL_DIR}/${ds}"
    answer_path="${ANSWER_DIR}/${ds}"
    [[ -d "${infill_path}" ]] || continue
    mkdir -p "${answer_path}"
    for f in "${infill_path}"/infill_*.json; do
        [[ -f "$f" ]] || continue
        echo ""
        echo ">>> $f"
        python -u scripts/answer.py \
            --model_name "${MODEL_NAME}" \
            --data_file "${f}" \
            --output_dir "${answer_path}" \
            --ent_exist_flag "${ENT_EXIST_FLAG}" \
            --max_trials "${MAX_TRIALS}"
    done
done

echo ""
echo "========== Done =========="
