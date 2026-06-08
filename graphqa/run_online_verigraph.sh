#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=60000MB
#SBATCH --job-name=online_vgraph
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_verigraph_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_verigraph_%j.err

# Online Veri-Graph GraphQA.
#
# 흐름:
#   raw question
#     -> online retrieval/SearchR1
#     -> doupari graph extractor로 Q / D / T / Sr 구성
#     -> stepwise cosine-only doc/think check + slot filling
#     -> 0/0, 1/0, 0/1, 1/1 별 QA accuracy 집계
#
# 기본 시작점:
#   - document graph: doupari/Llama-3.2-1B-Instruct-document
#   - question graph: doupari/Llama-3.2-1B-Instruct-question-think-search + question prompt
#   - think graph:    doupari/Llama-3.2-1B-Instruct-question-think-search + think+search prompt
#
# 실행:
#   sbatch graphqa/run_online_verigraph.sh
#   bash graphqa/run_online_verigraph.sh
#
# 자주 바꾸는 옵션:
#   ONLINE_DATASETS="2wikimultihopqa hotpotqa musique" ONLINE_LIMIT=0 bash graphqa/run_online_verigraph.sh
#   ONLINE_DATASET_LIMITS="500 500 1000" bash graphqa/run_online_verigraph.sh
#   ONLINE_RETRIEVAL_MODE=bm25 ONLINE_LIMIT=20 bash graphqa/run_online_verigraph.sh
#   ONLINE_QUESTION_MODEL=doupari/Llama-3.2-1B-Instruct-question-think bash graphqa/run_online_verigraph.sh
#   ONLINE_THINK_MODEL=doupari/Llama-3.2-1B-Instruct-think-search bash graphqa/run_online_verigraph.sh
#   ONLINE_GRAPH_BASE_MODEL=/path/to/Llama-3.2-1B-Instruct bash graphqa/run_online_verigraph.sh
#   ONLINE_COSINE_ON_FAIL=abstain ONLINE_COSINE_GATE_ON=doc bash graphqa/run_online_verigraph.sh
#   ONLINE_COSINE_STEP_ORDER=think bash graphqa/run_online_verigraph.sh
#   ONLINE_ANSWER_MODE=final_q_hint bash graphqa/run_online_verigraph.sh
#   ONLINE_TRAJECTORY_MODE=q_guided ONLINE_OUT_DIR=graphqa/outputs/online_verigraph_q_guided bash graphqa/run_online_verigraph.sh
#   ONLINE_TRAJECTORY_MODE=observer ONLINE_OUT_DIR=graphqa/outputs/online_verigraph_observer bash graphqa/run_online_verigraph.sh
set -euo pipefail

PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"
PY="/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3"
LOG_DIR="${PROJECT_ROOT}/graphqa/logs"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

set +u
source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

DATASETS_STR="${ONLINE_DATASETS:-2wikimultihopqa}" # space-separated list, e.g. "2wikimultihopqa hotpotqa musique"
read -r -a DATASETS <<< "${DATASETS_STR}"

LIMIT="${ONLINE_LIMIT:-}"                      # set하면 모든 dataset에 동일 적용, 0이면 전체
DATASET_LIMITS_STR="${ONLINE_DATASET_LIMITS:-500 500 1000}"
read -r -a DATASET_LIMITS <<< "${DATASET_LIMITS_STR}"
START="${ONLINE_START:-0}"
INPUT_FILENAME="${ONLINE_INPUT_FILENAME:-train_sampled.json}"
OUT_DIR="${ONLINE_OUT_DIR:-${PROJECT_ROOT}/graphqa/outputs/online_verigraph}"

RETRIEVAL_MODE="${ONLINE_RETRIEVAL_MODE:-searchr1}"  # searchr1 | bm25
TRAJECTORY_MODE="${ONLINE_TRAJECTORY_MODE:-searchr1_first}"  # searchr1_first | q_guided | observer
RETRIEVER_URL="${ONLINE_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
BM25_TOP_K="${ONLINE_BM25_TOP_K:-10}"
SEARCHR1_MODEL="${ONLINE_SEARCHR1_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"
SEARCHR1_TOP_K="${ONLINE_SEARCHR1_TOP_K:-3}"
SEARCHR1_MAX_TURNS="${ONLINE_SEARCHR1_MAX_TURNS:-4}"
SEARCHR1_MAX_NEW_TOKENS="${ONLINE_SEARCHR1_MAX_NEW_TOKENS:-500}"
SEARCHR1_TEMPERATURE="${ONLINE_SEARCHR1_TEMPERATURE:-1.0}"
ANSWER_MODE="${ONLINE_ANSWER_MODE:-both}"  # searchr1 | final_q_hint | both

DOCUMENT_MODEL="${ONLINE_DOCUMENT_MODEL:-doupari/Llama-3.2-1B-Instruct-document}"
QUESTION_MODEL="${ONLINE_QUESTION_MODEL:-doupari/Llama-3.2-1B-Instruct-question-think-search}"
THINK_MODEL="${ONLINE_THINK_MODEL:-doupari/Llama-3.2-1B-Instruct-question-think-search}"
QUESTION_TASK="${ONLINE_QUESTION_TASK:-question}"
THINK_TASK="${ONLINE_THINK_TASK:-think+search}"

GRAPH_DTYPE="${ONLINE_GRAPH_DTYPE:-bfloat16}"
# doupari adapters point to meta-llama/Llama-3.2-1B-Instruct.  The public
# unsloth mirror keeps the script runnable on machines without a Meta-gated
# HF token; override this if you want the exact gated base/local checkpoint.
GRAPH_BASE_MODEL="${ONLINE_GRAPH_BASE_MODEL:-unsloth/Llama-3.2-1B-Instruct}"
GRAPH_MAX_NEW_TOKENS="${ONLINE_GRAPH_MAX_NEW_TOKENS:-512}"
GRAPH_TEMPERATURE="${ONLINE_GRAPH_TEMPERATURE:-0.0}"
DOC_MAX_WORDS="${ONLINE_DOC_MAX_WORDS:-500}"
MAX_DOCS_PER_SAMPLE="${ONLINE_MAX_DOCS_PER_SAMPLE:-10}"
MAX_THINK_STEPS="${ONLINE_MAX_THINK_STEPS:-8}"
DOC_RESCUE_ROUNDS="${ONLINE_DOC_RESCUE_ROUNDS:-1}"
DOC_RESCUE_MAX_QUERIES="${ONLINE_DOC_RESCUE_MAX_QUERIES:-4}"
DOC_RESCUE_MAX_DOCS="${ONLINE_DOC_RESCUE_MAX_DOCS:-8}"

COSINE_THRESHOLD="${ONLINE_COSINE_THRESHOLD:-0.60}"
COSINE_MAX_STEPS="${ONLINE_COSINE_MAX_STEPS:-16}"
COSINE_DOC_TOP_K="${ONLINE_COSINE_DOC_TOP_K:-5}"
COSINE_ON_FAIL="${ONLINE_COSINE_ON_FAIL:-continue}"     # continue | abstain
COSINE_GATE_ON="${ONLINE_COSINE_GATE_ON:-doc}"          # doc | think | both | either
COSINE_FILL_SOURCE="${ONLINE_COSINE_FILL_SOURCE:-doc}"  # doc | think | doc_then_think
COSINE_STEP_ORDER="${ONLINE_COSINE_STEP_ORDER:-question}"  # question | think

ENCODER="${ONLINE_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
DEVICE="${ONLINE_DEVICE:-cuda}"

LIMIT_FLAGS=()
LIMIT_LABEL=""
if [[ -n "${LIMIT}" ]]; then
    LIMIT_FLAGS=(--limit "${LIMIT}")
    LIMIT_LABEL="${LIMIT}"
else
    LIMIT_FLAGS=(--limit 0 --dataset-limits "${DATASET_LIMITS[@]}")
    LIMIT_LABEL="per-dataset:${DATASET_LIMITS[*]}"
fi

echo "============================================================"
echo "[online-verigraph] PROJECT_ROOT      = ${PROJECT_ROOT}"
echo "[online-verigraph] datasets          = ${DATASETS[*]}"
echo "[online-verigraph] input             = ${INPUT_FILENAME}"
echo "[online-verigraph] limit/start       = ${LIMIT_LABEL}/${START}"
echo "[online-verigraph] output            = ${OUT_DIR}"
echo "[online-verigraph] retrieval         = ${RETRIEVAL_MODE}"
echo "[online-verigraph] trajectory        = ${TRAJECTORY_MODE}"
echo "[online-verigraph] retriever_url     = ${RETRIEVER_URL}"
echo "[online-verigraph] answer_mode       = ${ANSWER_MODE}"
echo "[online-verigraph] document_model    = ${DOCUMENT_MODEL}"
echo "[online-verigraph] question_model    = ${QUESTION_MODEL} (${QUESTION_TASK})"
echo "[online-verigraph] think_model       = ${THINK_MODEL} (${THINK_TASK})"
echo "[online-verigraph] graph_base_model  = ${GRAPH_BASE_MODEL:-(adapter default)}"
echo "[online-verigraph] cosine            = threshold ${COSINE_THRESHOLD}, top_k ${COSINE_DOC_TOP_K}, gate ${COSINE_GATE_ON}, on_fail ${COSINE_ON_FAIL}, order ${COSINE_STEP_ORDER}"
echo "[online-verigraph] doc_rescue        = rounds ${DOC_RESCUE_ROUNDS}, max_queries ${DOC_RESCUE_MAX_QUERIES}, max_docs ${DOC_RESCUE_MAX_DOCS}"
echo "[online-verigraph] SLURM_JOB_ID      = ${SLURM_JOB_ID:-(none)}"
echo "============================================================"

"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_online_eval.py" \
    --datasets "${DATASETS[@]}" \
    --input-filename "${INPUT_FILENAME}" \
    "${LIMIT_FLAGS[@]}" \
    --start "${START}" \
    --output-dir "${OUT_DIR}" \
    --retrieval-mode "${RETRIEVAL_MODE}" \
    --trajectory-mode "${TRAJECTORY_MODE}" \
    --retriever-url "${RETRIEVER_URL}" \
    --bm25-top-k "${BM25_TOP_K}" \
    --searchr1-model "${SEARCHR1_MODEL}" \
    --searchr1-top-k "${SEARCHR1_TOP_K}" \
    --searchr1-max-turns "${SEARCHR1_MAX_TURNS}" \
    --searchr1-max-new-tokens "${SEARCHR1_MAX_NEW_TOKENS}" \
    --searchr1-temperature "${SEARCHR1_TEMPERATURE}" \
    --answer-mode "${ANSWER_MODE}" \
    --document-model "${DOCUMENT_MODEL}" \
    --question-model "${QUESTION_MODEL}" \
    --think-model "${THINK_MODEL}" \
    --question-task "${QUESTION_TASK}" \
    --think-task "${THINK_TASK}" \
    --graph-dtype "${GRAPH_DTYPE}" \
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
    --device "${DEVICE}"

echo
echo "[online-verigraph] 0/0, 1/0, 0/1, 1/1 QA accuracy"
echo "------------------------------------------------------------"
"${PY}" -m graphqa.scripts.triplet_ok_confusion \
    --input-dir "${OUT_DIR}" \
    --csv-glob "**/online_eval_*.csv" \
    --save-plots

echo
echo "[online-verigraph] DONE"
echo "  CSV:     ${OUT_DIR}/<dataset>/online_eval_<dataset>.csv"
echo "  summary: ${OUT_DIR}/<dataset>/online_eval_<dataset>_summary.json"
echo "  graphs:  ${OUT_DIR}/<dataset>/online_eval_<dataset>_graphs.json"
