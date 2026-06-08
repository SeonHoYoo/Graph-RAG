#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_answer_aggregate_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_answer_aggregate_%j.err

# 1. 모든 infill 결과에 answer.py 실행
# 2. EM/F1 집계하여 CSV + 표 출력

BASE_DIR="/data3/seonhoyoo/graphcheck-qa"
INFILL_SCRIPTS="${BASE_DIR}/infilling/scripts"
ANSWER_PY="${INFILL_SCRIPTS}/scripts/answer.py"
OUTPUT_BASE="${BASE_DIR}/infilling/output"

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd "${INFILL_SCRIPTS}"

export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# sbatch run_answer_and_aggregate.sh 만 실행해도 동작 (인자 없음)
MODEL_NAME="${1:-Qwen/Qwen2.5-7B-Instruct}"
ENT_EXIST_FLAG="${2:-all}"
MODEL_SHORT="${MODEL_NAME##*/}"

echo "========== Step 1: Run answer.py on all infill results =========="
echo "MODEL=${MODEL_NAME}, ent_exist_flag=${ENT_EXIST_FLAG}"

for ds in 2wikimultihopqa hotpotqa musique; do
    infill_dir="${OUTPUT_BASE}/infill/${MODEL_SHORT}/${ds}"
    answer_dir="${OUTPUT_BASE}/answer/${MODEL_SHORT}/${ds}"
    [[ -d "${infill_dir}" ]] || continue
    mkdir -p "${answer_dir}"
    echo ""
    echo ">>> infill_dir=${infill_dir}"
    python -u "${ANSWER_PY}" \
        --model_name "${MODEL_NAME}" \
        --data_dir "${infill_dir}" \
        --output_dir "${answer_dir}" \
        --ent_exist_flag "${ENT_EXIST_FLAG}" \
        --max_trials 3
done

echo ""
echo "========== Step 2: Aggregate EM/F1 =========="
python -u "${INFILL_SCRIPTS}/aggregate_infill_metrics.py" \
    --output_dir "${OUTPUT_BASE}" \
    --model_name "${MODEL_NAME}" \
    --ent_exist_flag "${ENT_EXIST_FLAG}"

echo ""
echo "========== Done =========="
