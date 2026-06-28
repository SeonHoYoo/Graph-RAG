#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=40000MB
#SBATCH --job-name=baseline
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hyeseojeon/data/graph/graphqa/hs/logs/baseline/verigraph_baseline_%j.log
#SBATCH --error=/home/hyeseojeon/data/graph/graphqa/hs/logs/baseline/verigraph_baseline_%j.err
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

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${PROJECT_ROOT}/graphqa/hs/logs/baseline"

DATASETS_STR="${DATASETS:-2wikimultihopqa}"  # 2wikimultihopqa | hotpotqa | musique
read -r -a DATASETS <<< "${DATASETS_STR}"

LIMIT="${LIMIT:-20}"
DATASET_LIMITS_STR="${DATASET_LIMITS:-}"
read -r -a DATASET_LIMITS <<< "${DATASET_LIMITS_STR}"

START="${START:-0}"
INPUT_FILENAME="${INPUT_FILENAME:-train_sampled.json}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/graphqa/hs/outputs}"
BASELINE_MODES_STR="${BASELINE_MODES:-analyze_verigraph fallback_verigraph}"  # analyze_verigraph | fallback_verigraph
read -r -a BASELINE_MODES <<< "${BASELINE_MODES_STR}"

RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8003/retrieve}"
SEARCHR1_MODEL="${SEARCHR1_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"
SEARCHR1_TOP_K="${SEARCHR1_TOP_K:-3}"
SEARCHR1_MAX_TURNS="${SEARCHR1_MAX_TURNS:-5}"
SEARCHR1_MAX_NEW_TOKENS="${SEARCHR1_MAX_NEW_TOKENS:-500}"
SEARCHR1_TEMPERATURE="${SEARCHR1_TEMPERATURE:-1.0}"

DOCUMENT_MODEL="${DOCUMENT_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-document}"
QUESTION_MODEL="${QUESTION_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-question}"
THINK_MODEL="${THINK_MODEL:-outputs/finetune/Qwen2.5-0.5B-Instruct-think}"
QUESTION_TASK="${QUESTION_TASK:-question}"
THINK_TASK="${THINK_TASK:-think}"
GRAPH_DTYPE="${GRAPH_DTYPE:-bfloat16}"  # bfloat16 | float16 | float32
GRAPH_DEVICE_MAP="${GRAPH_DEVICE_MAP:-auto}"
GRAPH_BASE_MODEL="${GRAPH_BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
GRAPH_MAX_NEW_TOKENS="${GRAPH_MAX_NEW_TOKENS:-512}"
GRAPH_TEMPERATURE="${GRAPH_TEMPERATURE:-0.0}"

DOC_MAX_WORDS="${DOC_MAX_WORDS:-500}"
MAX_DOCS_PER_SAMPLE="${MAX_DOCS_PER_SAMPLE:-10}"
MAX_THINK_STEPS="${MAX_THINK_STEPS:-8}"

COSINE_THRESHOLD="${COSINE_THRESHOLD:-0.60}"
COSINE_DOC_TOP_K="${COSINE_DOC_TOP_K:-5}"
COSINE_MAX_STEPS="${COSINE_MAX_STEPS:-16}"
COSINE_ON_FAIL="${COSINE_ON_FAIL:-continue}"  # continue | abstain
COSINE_GATE_ON="${COSINE_GATE_ON:-doc}"  # doc | think | both | either
COSINE_FILL_SOURCE="${COSINE_FILL_SOURCE:-doc}"  # doc | think | doc_then_think
COSINE_STEP_ORDER="${COSINE_STEP_ORDER:-question}"  # question | think
DOC_RESCUE_ROUNDS="${DOC_RESCUE_ROUNDS:-1}"
DOC_RESCUE_MAX_QUERIES="${DOC_RESCUE_MAX_QUERIES:-4}"
DOC_RESCUE_MAX_DOCS="${DOC_RESCUE_MAX_DOCS:-8}"
CORRECTOR_COSINE_DOC_TOP_K="${CORRECTOR_COSINE_DOC_TOP_K:-3}"
ENCODER="${ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
DEVICE="${DEVICE:-cuda}"  # cuda | cpu | cuda:<idx>

VANILLA_TRIGGER_THRESHOLD="${VANILLA_TRIGGER_THRESHOLD:-4}"
CORRECTOR_MAX_TURNS="${CORRECTOR_MAX_TURNS:-8}"
CONTROL_MAX_TURNS="${CONTROL_MAX_TURNS:-8}"
CORRECTOR_V2_MAX_TURNS="${CORRECTOR_V2_MAX_TURNS:-8}"
CORRECTOR_V2_THRESHOLD="${CORRECTOR_V2_THRESHOLD:-0.50}"

mkdir -p "${OUT_BASE}"

_has_mode() {
    local want="$1"
    local mode
    for mode in "${BASELINE_MODES[@]}"; do
        [[ "${mode}" == "${want}" ]] && return 0
    done
    return 1
}

for mode in "${BASELINE_MODES[@]}"; do
    case "${mode}" in
        analyze_verigraph|fallback_verigraph)
            ;;
        *)
            echo "ERROR: unknown BASELINE_MODES entry: ${mode}" >&2
            echo "Allowed: analyze_verigraph fallback_verigraph" >&2
            exit 2
            ;;
    esac
done

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

GRAPH_SAVE_FLAGS=()
if [[ "${SAVE_GRAPHS:-1}" == "0" ]]; then
    GRAPH_SAVE_FLAGS=(--no-save-graphs)
fi

echo "============================================================"
echo "[baseline] project_root       = ${PROJECT_ROOT}"
echo "[baseline] python             = ${PY}"
echo "[baseline] datasets           = ${DATASETS[*]}"
echo "[baseline] modes              = ${BASELINE_MODES[*]}"
echo "[baseline] input/start/limit  = ${INPUT_FILENAME}/${START}/${LIMIT_LABEL}"
echo "[baseline] output             = ${OUT_BASE}"
echo "[baseline] retriever_url      = ${RETRIEVER_URL}"
echo "[baseline] searchr1_max_turns = ${SEARCHR1_MAX_TURNS}"
echo "[baseline] cosine_threshold   = ${COSINE_THRESHOLD}"
echo "============================================================"

if _has_mode analyze_verigraph; then
    echo
    echo "[analyze_verigraph] searchr1_first post-run VeriGraph analysis"
    echo "------------------------------------------------------------"
    "${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_online_eval.py" \
        --datasets "${DATASETS[@]}" \
        --input-filename "${INPUT_FILENAME}" \
        "${LIMIT_FLAGS[@]}" \
        --start "${START}" \
        --output-dir "${OUT_BASE}/analyze_verigraph" \
        --retrieval-mode searchr1 \
        --trajectory-mode searchr1_first \
        --retriever-url "${RETRIEVER_URL}" \
        --searchr1-model "${SEARCHR1_MODEL}" \
        --searchr1-top-k "${SEARCHR1_TOP_K}" \
        --searchr1-max-turns "${SEARCHR1_MAX_TURNS}" \
        --searchr1-max-new-tokens "${SEARCHR1_MAX_NEW_TOKENS}" \
        --searchr1-temperature "${SEARCHR1_TEMPERATURE}" \
        --answer-mode both \
        --document-model "${DOCUMENT_MODEL}" \
        --question-model "${QUESTION_MODEL}" \
        --think-model "${THINK_MODEL}" \
        --question-task "${QUESTION_TASK}" \
        --think-task "${THINK_TASK}" \
        --graph-dtype "${GRAPH_DTYPE}" \
        --graph-device-map "${GRAPH_DEVICE_MAP}" \
        --graph-base-model "${GRAPH_BASE_MODEL}" \
        --graph-max-new-tokens "${GRAPH_MAX_NEW_TOKENS}" \
        --graph-temperature "${GRAPH_TEMPERATURE}" \
        --doc-max-words "${DOC_MAX_WORDS}" \
        --max-docs-per-sample "${MAX_DOCS_PER_SAMPLE}" \
        --max-think-steps "${MAX_THINK_STEPS}" \
        --doc-rescue-rounds "${DOC_RESCUE_ROUNDS}" \
        --doc-rescue-max-queries "${DOC_RESCUE_MAX_QUERIES}" \
        --doc-rescue-max-docs "${DOC_RESCUE_MAX_DOCS}" \
        --cosine-threshold "${COSINE_THRESHOLD}" \
        --cosine-max-steps "${COSINE_MAX_STEPS}" \
        --cosine-doc-top-k "${COSINE_DOC_TOP_K}" \
        --cosine-on-fail "${COSINE_ON_FAIL}" \
        --cosine-gate-on "${COSINE_GATE_ON}" \
        --cosine-fill-source "${COSINE_FILL_SOURCE}" \
        --cosine-step-order "${COSINE_STEP_ORDER}" \
        --encoder "${ENCODER}" \
        --device "${DEVICE}" \
        "${GRAPH_SAVE_FLAGS[@]}"
else
    echo "[analyze_verigraph] skipped"
fi

if _has_mode fallback_verigraph; then
    echo
    echo "[fallback_verigraph] current fallback/rerun VeriGraph corrector"
    echo "------------------------------------------------------------"
    "${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_online_corrector.py" \
        --datasets "${DATASETS[@]}" \
        --input-filename "${INPUT_FILENAME}" \
        "${LIMIT_FLAGS[@]}" \
        --start "${START}" \
        --output-dir "${OUT_BASE}/fallback_verigraph" \
        --retriever-url "${RETRIEVER_URL}" \
        --searchr1-model "${SEARCHR1_MODEL}" \
        --searchr1-top-k "${SEARCHR1_TOP_K}" \
        --searchr1-max-turns "${SEARCHR1_MAX_TURNS}" \
        --searchr1-max-new-tokens "${SEARCHR1_MAX_NEW_TOKENS}" \
        --searchr1-temperature "${SEARCHR1_TEMPERATURE}" \
        --vanilla-trigger-threshold "${VANILLA_TRIGGER_THRESHOLD}" \
        --corrector-max-turns "${CORRECTOR_MAX_TURNS}" \
        --control-max-turns "${CONTROL_MAX_TURNS}" \
        --corrector-v2-max-turns "${CORRECTOR_V2_MAX_TURNS}" \
        --corrector-v2-threshold "${CORRECTOR_V2_THRESHOLD}" \
        --max-docs-per-sample "${MAX_DOCS_PER_SAMPLE}" \
        --doc-max-words "${DOC_MAX_WORDS}" \
        --question-model "${QUESTION_MODEL}" \
        --document-model "${DOCUMENT_MODEL}" \
        --think-model "${THINK_MODEL}" \
        --question-task "${QUESTION_TASK}" \
        --think-task "${THINK_TASK}" \
        --graph-dtype "${GRAPH_DTYPE}" \
        --graph-device-map "${GRAPH_DEVICE_MAP}" \
        --graph-base-model "${GRAPH_BASE_MODEL}" \
        --graph-max-new-tokens "${GRAPH_MAX_NEW_TOKENS}" \
        --graph-temperature "${GRAPH_TEMPERATURE}" \
        --cosine-threshold "${COSINE_THRESHOLD}" \
        --cosine-doc-top-k "${CORRECTOR_COSINE_DOC_TOP_K}" \
        --encoder "${ENCODER}" \
        --device "${DEVICE}"
else
    echo "[fallback_verigraph] skipped"
fi

echo
if _has_mode analyze_verigraph && _has_mode fallback_verigraph; then
    echo "[summary] collecting baseline metrics"
    echo "------------------------------------------------------------"
    "${PY}" "${PROJECT_ROOT}/graphqa/hs/scripts/sum_baseline.py" \
        --run-dir "${OUT_BASE}" \
        --datasets "${DATASETS[@]}" \
        --output "${OUT_BASE}/baseline_summary.json"
else
    echo "[summary] skipped (requires both analyze_verigraph and fallback_verigraph)"
fi

echo
echo "[baseline] DONE"
echo "  vanilla_searchr1:   ${OUT_BASE}/fallback_verigraph/<dataset>/online_corrector_<dataset>.csv (vanilla_* columns)"
echo "  analyze_verigraph:  ${OUT_BASE}/analyze_verigraph"
echo "  fallback_verigraph: ${OUT_BASE}/fallback_verigraph"
echo "  summary:            ${OUT_BASE}/baseline_summary.json"
