#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=40000MB
#SBATCH --job-name=iter_slot
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/logs/iterative_slot_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/logs/iterative_slot_%j.err

# 실험 A: --qa-mode iterative (중간 abstain·스텝 align 없음)
# 실험 B: 동일 + --iter-abstain (스텝마다 align 신호 + 중간 abstain)
# 산출물: 각 OUT 아래 <dataset>/tasi_eval_<dataset>.csv 및 *_summary.json
#
#   sbatch graphqa/run_iterative_slot_experiments.sh
#   bash    graphqa/run_iterative_slot_experiments.sh
set -euo pipefail

PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"
PY="/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3"
OUT_A="${PROJECT_ROOT}/graphqa/outputs/iterative_slot_base"
OUT_B="${PROJECT_ROOT}/graphqa/outputs/iterative_slot_align_abstain"

mkdir -p "${OUT_A}" "${OUT_B}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

DATASETS=(2wikimultihopqa hotpotqa musique)
LLM_FLAGS=(
    --pc-mode log_mean
    --use-llm qwen-local
    --llm-model Qwen/Qwen2.5-7B-Instruct
    --llm-dtype bfloat16
    --qa-mode iterative
)

echo "[iterative-slot] A → ${OUT_A}"
"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_eval.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${OUT_A}" \
    "${LLM_FLAGS[@]}"

echo "[iterative-slot] B → ${OUT_B}"
"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_eval.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${OUT_B}" \
    "${LLM_FLAGS[@]}" \
    --iter-abstain

echo "DONE  A=${OUT_A}  B=${OUT_B}  (각 데이터셋 폴더에 tasi_eval_*.csv / *_summary.json)"
