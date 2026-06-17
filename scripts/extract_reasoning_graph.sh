#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=60000MB
#SBATCH --job-name=reasoning_graph_extract
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/logs/extract_reasoning_graph_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/logs/extract_reasoning_graph_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

REPO_ROOT="/data3/seonhoyoo/graphcheck-qa"
cd "${REPO_ROOT}"

export HF_HOME=/data3/seonhoyoo/.cache/huggingface
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
# 보안: 키를 코드에 하드코딩하지 말고 실행 전 환경변수로 설정하세요.
#   export HF_TOKEN=...        # https://huggingface.co/settings/tokens
#   export OPENAI_API_KEY=...  # https://platform.openai.com/api-keys
export HF_TOKEN="${HF_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" ./logs/graph

# ---- Config ----
# sbatch extract_reasoning_graph.sh [DATASET] [MODEL] [MODE] [MAX_SAMPLES]
#   DATASET     : 2wiki | hotpotqa | musique | all      (default: all)
#   MODEL       : gpt-4.1-mini | Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-14B-Instruct | ...
#                 (default: gpt-4.1-mini)
#   MODE        : per_step | whole                      (default: per_step)
#   MAX_SAMPLES : integer                               (default: 전체)
DATASET_ARG="${1:-all}"
model_name="${2:-gpt-4.1-mini}"
mode="${3:-per_step}"
max_samples="${4:-}"

input_dir="${REPO_ROOT}/graph_data/searchr1/0407(open-book)"
output_dir="${REPO_ROOT}/graph_data/searchr1/0407(open-book)/reasoning_graph"

declare -A INPUT_FILES=(
    [2wiki]="2wiki_vanilla_searchr1_128615_500.json"
    [hotpotqa]="hotpotqa_vanilla_searchr1_128616_500.json"
    [musique]="musique_vanilla_searchr1_128617_1000.json"
)
ALL_DATASETS=("2wiki" "hotpotqa" "musique")

if [[ "${DATASET_ARG}" == "all" ]]; then
    datasets=("${ALL_DATASETS[@]}")
elif [[ -n "${INPUT_FILES[${DATASET_ARG}]:-}" ]]; then
    datasets=("${DATASET_ARG}")
else
    echo "ERROR: Unknown dataset '${DATASET_ARG}'. Use one of: 2wiki, hotpotqa, musique, all"
    exit 1
fi

mkdir -p "${output_dir}"

for ds in "${datasets[@]}"; do
    input_file="${input_dir}/${INPUT_FILES[${ds}]}"
    output_filename="reasoning_graph_${INPUT_FILES[${ds}]}"

    echo "=============================================================="
    echo "  Dataset : ${ds}"
    echo "  Input   : ${input_file}"
    echo "  Output  : ${output_dir}/${output_filename}"
    echo "  Model   : ${model_name}"
    echo "  Mode    : ${mode}"
    echo "  Max     : ${max_samples:-<all>}"
    echo "=============================================================="

    cmd=(
        python -u "${REPO_ROOT}/scripts/extract_reasoning_graph.py"
        --input_file_path "${input_file}"
        --model_name "${model_name}"
        --mode "${mode}"
        --output_dir "${output_dir}"
        --output_filename "${output_filename}"
        --checkpoint_every 10
    )
    # OpenAI/Claude 계열 API 모델일 때만 api_key 전달 (Qwen 등 로컬 모델은 불필요)
    lower_model="$(echo "${model_name}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lower_model}" == gpt* || "${lower_model}" == claude* ]]; then
        cmd+=(--api_key "${OPENAI_API_KEY}")
    fi
    if [[ -n "${max_samples}" ]]; then
        cmd+=(--max_samples "${max_samples}")
    fi

    echo "Command: ${cmd[*]}"
    "${cmd[@]}"

    echo ""
    echo ">>> ${ds} done."
    echo ""
done

echo "All datasets finished."
