#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-00:10:00
#SBATCH --mem=4000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_evaluate_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_evaluate_%j.err

# answer_*.json 있으면 평가만 (모델 호출 없음)
SCRIPT_DIR="/data3/seonhoyoo/graphcheck-qa/infilling/scripts"
OUTPUT_BASE="/data3/seonhoyoo/graphcheck-qa/infilling/output"

source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

cd "${SCRIPT_DIR}"
python -u "${SCRIPT_DIR}/evaluate_infill_results.py" --output_dir "${OUTPUT_BASE}" -q
