#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=40000MB
#SBATCH --job-name=online_feedback
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hyeseojeon/data/graph/graphqa/hs/logs/online_feedback/online_feedback_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/graphqa/hs/logs/online_feedback/online_feedback_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hyeseojeon/data/graph}"
PY="${PY:-python}"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/hyeseojeon/data/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${PROJECT_ROOT}/graphqa/hs/logs/online_feedback"

DATASETS_STR="${DATASETS:-2wikimultihopqa}"  # 2wikimultihopqa | hotpotqa | musique
read -r -a DATASETS <<< "${DATASETS_STR}"

LIMIT="${LIMIT:-20}"
DATASET_LIMITS_STR="${DATASET_LIMITS:-}"
read -r -a DATASET_LIMITS <<< "${DATASET_LIMITS_STR}"

START="${START:-0}"
UID_FILE="${UID_FILE:-}"
INPUT_FILENAME="${INPUT_FILENAME:-train_sampled.json}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/graphqa/hs/outputs}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"

RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8003/retrieve}"
SEARCHR1_MODEL="${SEARCHR1_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"
SEARCHR1_TOP_K="${SEARCHR1_TOP_K:-3}"
SEARCHR1_MAX_TURNS="${SEARCHR1_MAX_TURNS:-10}"
SEARCHR1_MAX_NEW_TOKENS="${SEARCHR1_MAX_NEW_TOKENS:-500}"
SEARCHR1_TEMPERATURE="${SEARCHR1_TEMPERATURE:-1.0}"

ONLINE_FEEDBACK_MAX_TURNS="${ONLINE_FEEDBACK_MAX_TURNS:-8}"
ONLINE_FEEDBACK_THRESHOLD="${ONLINE_FEEDBACK_THRESHOLD:-0.50}"
FEEDBACK_ENGINE="${FEEDBACK_ENGINE:-legacy}"  # legacy | labels
FEEDBACK_SOURCE="${FEEDBACK_SOURCE:-full}"  # full | question_only
FEEDBACK_STYLE="${FEEDBACK_STYLE:-natural_mismatch}"  # natural_mismatch | natural_mismatch_query | natural_mismatch_query_norid | natural_mismatch_query_short | natural_revision | revision_brief | correction_brief | repair_brief | explicit_requirements | explicit_requirements_short | explicit_requirements_soft | requirement_brief | natural | natural_brief | natural_check | natural_guard | soft | score | strict | checklist | next_query
FEEDBACK_POSITION="${FEEDBACK_POSITION:-after_info}"  # after_info | before_info | inside_info | next_think_prefix
FEEDBACK_TRIGGER="${FEEDBACK_TRIGGER:-all}"  # all | non_support | actionable | strict | strict_query | unsupported_only | conflict_only | claim_only | missing_query | off
ABSTAIN_ON_UNRESOLVED="${ABSTAIN_ON_UNRESOLVED:-0}"  # 0 | 1
ABSTAIN_MIN_TURN="${ABSTAIN_MIN_TURN:-3}"
WAIT_FOR_SERVERS="${WAIT_FOR_SERVERS:-0}"  # 0 | 1
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"

DOCUMENT_MODEL="${DOCUMENT_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-document}"
QUESTION_MODEL="${QUESTION_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-question}"
THINK_MODEL="${THINK_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-think}"
QUESTION_TASK="${QUESTION_TASK:-question}"
THINK_TASK="${THINK_TASK:-think}"
GRAPH_BACKEND="${GRAPH_BACKEND:-local}"  # local | vllm
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8006/v1}"
VLLM_QUESTION_BASE_URL="${VLLM_QUESTION_BASE_URL:-}"
VLLM_DOCUMENT_BASE_URL="${VLLM_DOCUMENT_BASE_URL:-}"
VLLM_THINK_BASE_URL="${VLLM_THINK_BASE_URL:-}"
VLLM_QUESTION_MODEL_NAME="${VLLM_QUESTION_MODEL_NAME:-}"
VLLM_DOCUMENT_MODEL_NAME="${VLLM_DOCUMENT_MODEL_NAME:-}"
VLLM_THINK_MODEL_NAME="${VLLM_THINK_MODEL_NAME:-}"
VLLM_BATCH_CONCURRENCY="${VLLM_BATCH_CONCURRENCY:-8}"
GRAPH_OVERLAP_WORKERS="${GRAPH_OVERLAP_WORKERS:-4}"
GRAPH_DTYPE="${GRAPH_DTYPE:-bfloat16}"  # bfloat16 | float16 | float32
GRAPH_DEVICE_MAP="${GRAPH_DEVICE_MAP:-auto}"
GRAPH_BASE_MODEL="${GRAPH_BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
GRAPH_MAX_NEW_TOKENS="${GRAPH_MAX_NEW_TOKENS:-512}"
GRAPH_TEMPERATURE="${GRAPH_TEMPERATURE:-0.0}"

DOC_MAX_WORDS="${DOC_MAX_WORDS:-500}"
MAX_DOCS_PER_SAMPLE="${MAX_DOCS_PER_SAMPLE:-10}"
COSINE_DOC_TOP_K="${COSINE_DOC_TOP_K:-3}"
ENCODER="${ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
DEVICE="${DEVICE:-cuda}"  # cuda | cpu | cuda:<idx>

mkdir -p "${OUT_BASE}/online_feedback"

LIMIT_FLAGS=()
LIMIT_LABEL=""
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
    LIMIT_FLAGS=(--limit "${LIMIT}")
    LIMIT_LABEL="${LIMIT}"
else
    if [[ ${#DATASET_LIMITS[@]} -gt 0 ]]; then
        LIMIT_FLAGS=(--limit 0 --dataset-limits "${DATASET_LIMITS[@]}")
        LIMIT_LABEL="per-dataset:${DATASET_LIMITS[*]}"
    else
        LIMIT_FLAGS=(--limit 0)
        LIMIT_LABEL="all"
    fi
fi

echo "============================================================"
echo "[online_feedback] datasets          = ${DATASETS[*]}"
echo "[online_feedback] input/start/limit = ${INPUT_FILENAME}/${START}/${LIMIT_LABEL}"
echo "[online_feedback] uid_file          = ${UID_FILE:-<none>}"
echo "[online_feedback] output            = ${OUT_BASE}/online_feedback"
echo "[online_feedback] output_suffix     = ${OUTPUT_SUFFIX}"
echo "[online_feedback] retriever_url     = ${RETRIEVER_URL}"
echo "[online_feedback] max_turns         = ${ONLINE_FEEDBACK_MAX_TURNS}"
echo "[online_feedback] threshold         = ${ONLINE_FEEDBACK_THRESHOLD}"
echo "[online_feedback] feedback_engine   = ${FEEDBACK_ENGINE}"
echo "[online_feedback] feedback_source   = ${FEEDBACK_SOURCE}"
echo "[online_feedback] feedback_style    = ${FEEDBACK_STYLE}"
echo "[online_feedback] feedback_position = ${FEEDBACK_POSITION}"
echo "[online_feedback] feedback_trigger  = ${FEEDBACK_TRIGGER}"
echo "[online_feedback] graph_backend     = ${GRAPH_BACKEND}"
echo "[online_feedback] vllm_base_url     = ${VLLM_BASE_URL}"
echo "[online_feedback] vllm_q_url        = ${VLLM_QUESTION_BASE_URL:-${VLLM_BASE_URL}}"
echo "[online_feedback] vllm_d_url        = ${VLLM_DOCUMENT_BASE_URL:-${VLLM_BASE_URL}}"
echo "[online_feedback] vllm_t_url        = ${VLLM_THINK_BASE_URL:-${VLLM_BASE_URL}}"
echo "[online_feedback] vllm_concurrency  = ${VLLM_BATCH_CONCURRENCY}"
echo "[online_feedback] overlap_workers   = ${GRAPH_OVERLAP_WORKERS}"
echo "[online_feedback] abstain_unresolved= ${ABSTAIN_ON_UNRESOLVED}"
echo "[online_feedback] wait_for_servers  = ${WAIT_FOR_SERVERS}"
echo "============================================================"

wait_http() {
    local name="$1"
    local url="$2"
    local data="${3:-}"
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if [[ -n "${data}" ]]; then
            if curl -fsS --max-time 5 -H 'Content-Type: application/json' -d "${data}" "${url}" >/dev/null 2>&1; then
                echo "[online_feedback] ${name} ready: ${url}"
                return 0
            fi
        else
            if curl -fsS --max-time 5 "${url}" >/dev/null 2>&1; then
                echo "[online_feedback] ${name} ready: ${url}"
                return 0
            fi
        fi
        if (( $(date +%s) - start_ts >= WAIT_TIMEOUT_SEC )); then
            echo "[online_feedback] timeout waiting for ${name}: ${url}" >&2
            return 1
        fi
        sleep 10
    done
}

if [[ "${WAIT_FOR_SERVERS}" == "1" ]]; then
    wait_http "retriever" "${RETRIEVER_URL}" '{"queries":["server readiness probe"],"topk":1,"return_scores":true}'
    if [[ "${GRAPH_BACKEND}" == "vllm" ]]; then
        wait_http "graph_vllm" "${VLLM_BASE_URL%/v1}/v1/models"
    fi
fi

ABSTAIN_FLAGS=()
if [[ "${ABSTAIN_ON_UNRESOLVED}" == "1" ]]; then
    ABSTAIN_FLAGS=(--abstain-on-unresolved --abstain-min-turn "${ABSTAIN_MIN_TURN}")
fi

"${PY}" "${PROJECT_ROOT}/graphqa/hs/scripts/run_online_feedback.py" \
    --datasets "${DATASETS[@]}" \
    --input-filename "${INPUT_FILENAME}" \
    "${LIMIT_FLAGS[@]}" \
    --start "${START}" \
    --uid-file "${UID_FILE}" \
    --output-dir "${OUT_BASE}/online_feedback" \
    --output-suffix "${OUTPUT_SUFFIX}" \
    --retriever-url "${RETRIEVER_URL}" \
    --searchr1-model "${SEARCHR1_MODEL}" \
    --searchr1-top-k "${SEARCHR1_TOP_K}" \
    --searchr1-max-turns "${SEARCHR1_MAX_TURNS}" \
    --searchr1-max-new-tokens "${SEARCHR1_MAX_NEW_TOKENS}" \
    --searchr1-temperature "${SEARCHR1_TEMPERATURE}" \
    --online-feedback-max-turns "${ONLINE_FEEDBACK_MAX_TURNS}" \
    --online-feedback-threshold "${ONLINE_FEEDBACK_THRESHOLD}" \
    --feedback-engine "${FEEDBACK_ENGINE}" \
    --feedback-source "${FEEDBACK_SOURCE}" \
    --feedback-style "${FEEDBACK_STYLE}" \
    --feedback-position "${FEEDBACK_POSITION}" \
    --feedback-trigger "${FEEDBACK_TRIGGER}" \
    "${ABSTAIN_FLAGS[@]}" \
    --max-docs-per-sample "${MAX_DOCS_PER_SAMPLE}" \
    --doc-max-words "${DOC_MAX_WORDS}" \
    --question-model "${QUESTION_MODEL}" \
    --document-model "${DOCUMENT_MODEL}" \
    --think-model "${THINK_MODEL}" \
    --question-task "${QUESTION_TASK}" \
    --think-task "${THINK_TASK}" \
    --graph-backend "${GRAPH_BACKEND}" \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --vllm-question-base-url "${VLLM_QUESTION_BASE_URL}" \
    --vllm-document-base-url "${VLLM_DOCUMENT_BASE_URL}" \
    --vllm-think-base-url "${VLLM_THINK_BASE_URL}" \
    --vllm-question-model-name "${VLLM_QUESTION_MODEL_NAME}" \
    --vllm-document-model-name "${VLLM_DOCUMENT_MODEL_NAME}" \
    --vllm-think-model-name "${VLLM_THINK_MODEL_NAME}" \
    --vllm-batch-concurrency "${VLLM_BATCH_CONCURRENCY}" \
    --graph-overlap-workers "${GRAPH_OVERLAP_WORKERS}" \
    --graph-dtype "${GRAPH_DTYPE}" \
    --graph-device-map "${GRAPH_DEVICE_MAP}" \
    --graph-base-model "${GRAPH_BASE_MODEL}" \
    --graph-max-new-tokens "${GRAPH_MAX_NEW_TOKENS}" \
    --graph-temperature "${GRAPH_TEMPERATURE}" \
    --cosine-doc-top-k "${COSINE_DOC_TOP_K}" \
    --encoder "${ENCODER}" \
    --device "${DEVICE}"

echo
echo "[online_feedback] DONE"
echo "  output:  ${OUT_BASE}/online_feedback"
echo "  summary: ${OUT_BASE}/online_feedback/online_feedback_all_summary_${OUTPUT_SUFFIX}.json"
