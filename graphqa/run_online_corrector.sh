#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=60000MB
#SBATCH --job-name=online_corrector
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_corrector_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_corrector_%j.err

# Vanilla SearchR1 + selective VeriGraph reasoning corrector.
#
# Per-sample flow:
#   1) run vanilla SearchR1 (no Veri-Graph)
#   2) if vanilla n_searches <= VANILLA_TRIGGER_THRESHOLD AND it produced an
#      <answer>, accept it (cheap regime).
#   3) otherwise, re-run SearchR1 with an enlarged thinking budget while a
#      callback injects head/relation/tail cosine alignment of the latest
#      think+docs vs. the question graph after every turn. The injected
#      block guides reasoning only; it never reveals an answer.
#   4) if the corrector also has no <answer>, mark the sample as abstained.
#
# Outputs (per dataset):
#   <OUT_DIR>/<dataset>/online_corrector_<dataset>.csv             per-sample rows
#   <OUT_DIR>/<dataset>/online_corrector_<dataset>_summary.json    aggregate metrics
#   <OUT_DIR>/<dataset>/online_corrector_<dataset>_cases.jsonl     full trajectories + VG feedback
#   <OUT_DIR>/online_corrector_all.csv                              combined CSV
#   <OUT_DIR>/online_corrector_all_summary.json                     combined summary
#
# Common overrides:
#   CORRECTOR_DATASETS="2wikimultihopqa hotpotqa musique"
#   CORRECTOR_DATASET_LIMITS="500 500 1000"
#   CORRECTOR_VANILLA_TRIGGER_THRESHOLD=3
#   CORRECTOR_CORRECTOR_MAX_TURNS=8
#   CORRECTOR_COSINE_THRESHOLD=0.60
#
# Prerequisite: BM25 retrieval server must be running at the URL below.
#   cd /data3/seonhoyoo/multihopqa/Search-R1 && sbatch retrieval_launch_bm25.sh
# (Same server as run_online_verigraph.sh; nothing has changed there.)
#
# Run:
#   sbatch graphqa/run_online_corrector.sh
#   bash graphqa/run_online_corrector.sh
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
# Cut CUDA fragmentation OOMs: corrector turns accumulate long prompts which
# cause large contiguous-block allocations to fail even with free memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

DATASETS_STR="${CORRECTOR_DATASETS:-2wikimultihopqa}" # hotpotqa musique
read -r -a DATASETS <<< "${DATASETS_STR}"

LIMIT="${CORRECTOR_LIMIT:-}"
DATASET_LIMITS_STR="${CORRECTOR_DATASET_LIMITS:-500}" # 500 1000
read -r -a DATASET_LIMITS <<< "${DATASET_LIMITS_STR}"
START="${CORRECTOR_START:-0}"
INPUT_FILENAME="${CORRECTOR_INPUT_FILENAME:-train_sampled.json}"
OUT_DIR="${CORRECTOR_OUT_DIR:-${PROJECT_ROOT}/graphqa/outputs/online_corrector}"

RETRIEVER_URL="${CORRECTOR_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

SEARCHR1_MODEL="${CORRECTOR_SEARCHR1_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"
SEARCHR1_TOP_K="${CORRECTOR_SEARCHR1_TOP_K:-3}"
SEARCHR1_MAX_TURNS="${CORRECTOR_SEARCHR1_MAX_TURNS:-5}"           # vanilla budget (matches vanilla.sh)
SEARCHR1_MAX_NEW_TOKENS="${CORRECTOR_SEARCHR1_MAX_NEW_TOKENS:-500}"
SEARCHR1_TEMPERATURE="${CORRECTOR_SEARCHR1_TEMPERATURE:-1.0}"

VANILLA_TRIGGER_THRESHOLD="${CORRECTOR_VANILLA_TRIGGER_THRESHOLD:-4}"  # trigger when vanilla hit its max (5)
CORRECTOR_MAX_TURNS="${CORRECTOR_CORRECTOR_MAX_TURNS:-8}"              # extended budget when verigraph active (data shows corrector avg 3.87 turns → 8 is plenty, lowers OOM risk vs 10)
CONTROL_MAX_TURNS="${CORRECTOR_CONTROL_MAX_TURNS:-8}"                  # System-C ablation matched to corrector budget
CORRECTOR_V2_MAX_TURNS="${CORRECTOR_CORRECTOR_V2_MAX_TURNS:-8}"        # System-D: hint-style verigraph (post-case-study redesign)
CORRECTOR_V2_THRESHOLD="${CORRECTOR_CORRECTOR_V2_THRESHOLD:-0.50}"     # System-D relaxed threshold (catches passive-voice paraphrases like 'directed by' ≡ 'is the director of')
MAX_DOCS_PER_SAMPLE="${CORRECTOR_MAX_DOCS_PER_SAMPLE:-10}"
DOC_MAX_WORDS="${CORRECTOR_DOC_MAX_WORDS:-500}"

DOCUMENT_MODEL="${CORRECTOR_DOCUMENT_MODEL:-doupari/Llama-3.2-1B-Instruct-document}"
QUESTION_MODEL="${CORRECTOR_QUESTION_MODEL:-doupari/Llama-3.2-1B-Instruct-question-think-search}"
THINK_MODEL="${CORRECTOR_THINK_MODEL:-doupari/Llama-3.2-1B-Instruct-question-think-search}"
QUESTION_TASK="${CORRECTOR_QUESTION_TASK:-question}"
THINK_TASK="${CORRECTOR_THINK_TASK:-think+search}"

GRAPH_DTYPE="${CORRECTOR_GRAPH_DTYPE:-bfloat16}"
GRAPH_BASE_MODEL="${CORRECTOR_GRAPH_BASE_MODEL:-unsloth/Llama-3.2-1B-Instruct}"
GRAPH_MAX_NEW_TOKENS="${CORRECTOR_GRAPH_MAX_NEW_TOKENS:-512}"
GRAPH_TEMPERATURE="${CORRECTOR_GRAPH_TEMPERATURE:-0.0}"
GRAPH_DEVICE_MAP="${CORRECTOR_GRAPH_DEVICE_MAP:-auto}"

COSINE_THRESHOLD="${CORRECTOR_COSINE_THRESHOLD:-0.50}"   # both V1 (B) and V2 (D) gate at 0.50: format-only ablation
COSINE_DOC_TOP_K="${CORRECTOR_COSINE_DOC_TOP_K:-3}"   # standard Veri-Graph: top-3 candidates per Q triple

ENCODER="${CORRECTOR_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
DEVICE="${CORRECTOR_DEVICE:-cuda}"

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
echo "[corrector] PROJECT_ROOT             = ${PROJECT_ROOT}"
echo "[corrector] datasets                 = ${DATASETS[*]}"
echo "[corrector] input                    = ${INPUT_FILENAME}"
echo "[corrector] limit/start              = ${LIMIT_LABEL}/${START}"
echo "[corrector] output                   = ${OUT_DIR}"
echo "[corrector] retriever_url            = ${RETRIEVER_URL}"
echo "[corrector] searchr1_model           = ${SEARCHR1_MODEL}"
echo "[corrector] searchr1_max_turns       = ${SEARCHR1_MAX_TURNS} (vanilla)"
echo "[corrector] vanilla_trigger_thresh   = ${VANILLA_TRIGGER_THRESHOLD}"
echo "[corrector] corrector_max_turns      = ${CORRECTOR_MAX_TURNS} (verigraph-active)"
echo "[corrector] control_max_turns        = ${CONTROL_MAX_TURNS} (System-C: vanilla long, no verigraph)"
echo "[corrector] corrector_v2_max_turns   = ${CORRECTOR_V2_MAX_TURNS} (System-D: hint-style verigraph)"
echo "[corrector] corrector_v2_threshold   = ${CORRECTOR_V2_THRESHOLD} (System-D relaxed threshold)"
echo "[corrector] cosine_threshold         = ${COSINE_THRESHOLD}"
echo "[corrector] cosine_doc_top_k         = ${COSINE_DOC_TOP_K}"
echo "[corrector] document_model           = ${DOCUMENT_MODEL}"
echo "[corrector] question_model           = ${QUESTION_MODEL} (${QUESTION_TASK})"
echo "[corrector] think_model              = ${THINK_MODEL} (${THINK_TASK})"
echo "[corrector] graph_base_model         = ${GRAPH_BASE_MODEL:-(adapter default)}"
echo "[corrector] SLURM_JOB_ID             = ${SLURM_JOB_ID:-(none)}"
echo "============================================================"

"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_online_corrector.py" \
    --datasets "${DATASETS[@]}" \
    --input-filename "${INPUT_FILENAME}" \
    "${LIMIT_FLAGS[@]}" \
    --start "${START}" \
    --output-dir "${OUT_DIR}" \
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
    --cosine-doc-top-k "${COSINE_DOC_TOP_K}" \
    --encoder "${ENCODER}" \
    --device "${DEVICE}"

echo
echo "[corrector] DONE"
echo "  CSV:     ${OUT_DIR}/<dataset>/online_corrector_<dataset>.csv"
echo "  summary: ${OUT_DIR}/<dataset>/online_corrector_<dataset>_summary.json"
echo "  cases:   ${OUT_DIR}/<dataset>/online_corrector_<dataset>_cases.jsonl"
