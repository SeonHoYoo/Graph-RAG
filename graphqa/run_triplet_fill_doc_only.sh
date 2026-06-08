#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=40000MB
#SBATCH --job-name=trip_fill_doc
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/triplet_fill_doc_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/triplet_fill_doc_%j.err

# Triplet-fill doc-only experiment.
#
# Query triple의 UNKNOWN을 채울 때:
#   1. 현재 query triple 중 UNKNOWN entity 수가 가장 적은 triple을 선택
#   2. combined/0514 step evidence를 combined_full / combined_strict /
#      combined_prefix 세 모드로 돌림
#   3. document는 whole-triple sentence cosine top-K 중 field score 최고 후보 선택
#      think 후보 triple은 분석용 top-1 선택
#   4. head / relation / tail field-level cosine을 보되 UNKNOWN field는 제외
#   5. document field score가 threshold 이상이면 document triple에서 슬롯 채움
#      --triplet-fill-think-rescue 사용 시 doc fail + think ok + UNKNOWN 1개일 때
#      think triple의 concrete value로 슬롯 채움
#   6. 모든 query slot이 채워지면 filled query chain만 보고 LLM이 최종 답
#
# 실행:
#   sbatch graphqa/run_triplet_fill_doc_only.sh
#   bash    graphqa/run_triplet_fill_doc_only.sh
set -euo pipefail

PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"
PY="/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3"
LOG_DIR="${PROJECT_ROOT}/graphqa/logs"

THRESHOLD="${TRIPLET_FILL_THRESHOLD:-0.50}"
MAX_STEPS="${TRIPLET_FILL_MAX_STEPS:-16}"
DOC_TOP_K="${TRIPLET_FILL_DOC_TOP_K:-5}"
ANSWER_ON_FAIL="${TRIPLET_FILL_ANSWER_ON_FAIL:-1}"
THINK_RESCUE="${TRIPLET_FILL_THINK_RESCUE:-0}"
RUN_BOTH_RESCUE="${TRIPLET_FILL_RUN_BOTH_RESCUE:-1}"
COMBINED_DIR="${TRIPLET_FILL_COMBINED_DIR:-${PROJECT_ROOT}/graph_data/combined/0514}"
EVIDENCE_SCOPES_STR="${TRIPLET_FILL_EVIDENCE_SCOPES:-combined_full combined_strict combined_prefix}"
OUT_BASE="${TRIPLET_FILL_OUT_BASE:-${PROJECT_ROOT}/graphqa/outputs}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

DATASETS=(2wikimultihopqa hotpotqa musique)
BASE_LLM_FLAGS=(
    --pc-mode log_mean
    --use-llm qwen-local
    --llm-model Qwen/Qwen2.5-7B-Instruct
    --llm-dtype bfloat16
    --qa-mode triplet_fill
    --triplet-fill-threshold "${THRESHOLD}"
    --triplet-fill-max-steps "${MAX_STEPS}"
    --triplet-fill-doc-top-k "${DOC_TOP_K}"
)

_enabled() {
    local v="${1,,}"
    [[ "${v}" != "0" && "${v}" != "false" && "${v}" != "no" ]]
}

run_one() {
    local label="$1"
    local out_dir="$2"
    local rescue="$3"
    local evidence_scope="$4"
    local llm_flags=("${BASE_LLM_FLAGS[@]}")

    if _enabled "${ANSWER_ON_FAIL}"; then
        llm_flags+=(--triplet-fill-answer-on-fail)
    fi
    if _enabled "${rescue}"; then
        llm_flags+=(--triplet-fill-think-rescue)
    fi

    mkdir -p "${out_dir}"
    echo "============================================================"
    echo "[triplet-fill-doc] label       = ${label}"
    echo "[triplet-fill-doc] output      = ${out_dir}"
    echo "[triplet-fill-doc] threshold   = ${THRESHOLD}"
    echo "[triplet-fill-doc] max_steps   = ${MAX_STEPS}"
    echo "[triplet-fill-doc] doc_top_k   = ${DOC_TOP_K}"
    echo "[triplet-fill-doc] answer_fail = ${ANSWER_ON_FAIL}"
    echo "[triplet-fill-doc] think_rescue= ${rescue}"
    echo "[triplet-fill-doc] evidence   = ${evidence_scope}"
    echo "[triplet-fill-doc] combined   = ${COMBINED_DIR}"
    echo "[triplet-fill-doc] SLURM_JOB_ID= ${SLURM_JOB_ID:-(none)}"
    echo "============================================================"

    "${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_eval.py" \
        --datasets "${DATASETS[@]}" \
        --output-dir "${out_dir}" \
        --combined-dir "${COMBINED_DIR}" \
        --triplet-fill-evidence-scope "${evidence_scope}" \
        "${llm_flags[@]}" \
        --save-plots

    echo
    echo "[triplet-fill-doc] doc/think OK pair analysis (${label})"
    echo "------------------------------------------------------------"
    "${PY}" -m graphqa.scripts.triplet_ok_confusion \
        --input-dir "${out_dir}" \
        --save-plots

    echo
    echo "[triplet-fill-doc] route alignment analysis (${label})"
    echo "------------------------------------------------------------"
    "${PY}" -m graphqa.scripts.triplet_route_alignment \
        --input-dir "${out_dir}"

    echo
    echo "[triplet-fill-doc] finished ${label}"
    echo "  results: ${out_dir}"
    echo "  aggregate: ${out_dir}/triplet_ok_aggregate.json"
    echo "  debug json: ${out_dir}/<dataset>/tasi_eval_<dataset>_triplet_debug.json"
    echo "  death steps: ${out_dir}/<dataset>/tasi_eval_<dataset>_triplet_death_steps.csv"
    echo "  route: ${out_dir}/<dataset>/tasi_eval_<dataset>_triplet_route_alignment.json"
    echo "  per dataset: ${out_dir}/<dataset>/tasi_eval_<dataset>_triplet_ok_summary.json"
}

IFS=' ' read -r -a EVIDENCE_SCOPES <<< "${EVIDENCE_SCOPES_STR}"
for evidence_scope in "${EVIDENCE_SCOPES[@]}"; do
    if _enabled "${RUN_BOTH_RESCUE}"; then
        run_one \
            "${evidence_scope}_topk_no_think_rescue" \
            "${OUT_BASE}/triplet_fill_${evidence_scope}_topk_answer_on_fail" \
            0 \
            "${evidence_scope}"
        echo
        run_one \
            "${evidence_scope}_topk_with_think_rescue" \
            "${OUT_BASE}/triplet_fill_${evidence_scope}_topk_think_rescue_answer_on_fail" \
            1 \
            "${evidence_scope}"
    else
        if _enabled "${THINK_RESCUE}"; then
            out_dir="${OUT_BASE}/triplet_fill_${evidence_scope}_topk_think_rescue_answer_on_fail"
        else
            out_dir="${OUT_BASE}/triplet_fill_${evidence_scope}_topk_answer_on_fail"
        fi
        run_one \
            "${evidence_scope}_topk_single" \
            "${out_dir}" \
            "${THINK_RESCUE}" \
            "${evidence_scope}"
    fi
    echo
done

echo
echo "DONE"
