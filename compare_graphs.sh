#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/veri/0402/compare_graphs_%j.log
#SBATCH --error=./logs/veri/0402/compare_graphs_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

# 작업 디렉토리로 이동
cd /home/hyeseojeon/data/graph

# Hugging Face 캐시 경로 고정(노드가 달라도 동일 경로 사용)
export HF_HOME=/home/hyeseojeon/data/hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN="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"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# Transformers 업그레이드 (Qwen2.5 지원) - graphcheck 환경에 설치됨
pip install --upgrade transformers>=4.37.0 --quiet

input_fname="train_sampled"
dataset="hotpotqa"   # musique | hotpotqa | 2wikimultihopqa

# Qwen2.5 모델 사용
construct_model="Qwen/Qwen2.5-7B-Instruct"
# construct_model="Qwen/Qwen2.5-32B-Instruct"  # 더 작은 모델 사용 시
# construct_model="Qwen/Qwen2.5-14B-Instruct"  # 더 작은 모델 사용 시

# 검색 전략 설정
# 옵션: question, cot_reasoning, triplets, combined, multihop_cot_triplets, question_triplets, multihop_graphcheck_triplets
# - question: 질문만으로 검색 (기본 BM25)
# - cot_reasoning: CoT reasoning을 thinking으로 사용 (SearchR1 필요)
# - triplets: CoT triplets를 검색 쿼리로 사용
# - combined: 질문 + triplets + CoT reasoning 모두 사용

# - question_triplets: GraphCheck 스타일 질문 triplets로 검색
# - multihop_cot_triplets: CoT triplets 각 항목별 멀티홉 검색 수행
# - multihop_graphcheck_triplets: GraphCheck triplets 각 항목별 멀티홉 검색
retrieval_strategy="question"

# SearchR1 사용 여부 (cot_reasoning, combined 전략 사용 시 권장)
# use_searchr1=false
use_searchr1=true

# Nudge 모델과 함께 SearchR1 사용 (use_searchr1=true일 때만)
# nudge_searchr1=false
nudge_searchr1=true

# 멀티홉 검색 시 각 triplet당 검색할 문서 수 (multihop_cot_triplets, multihop_graphcheck_triplets 전략 사용 시)
multihop_top_k_per_triplet=2

bm25_top_k=5
setting="open-book+gold"
max_samples=30
cot_retry=3
force_cot_regen=false
compare_question_graph=true
output_dir="/home/hyeseojeon/data/graph/results/veri/0402/"
output_filename="hotpotqa_veri_sample.json"

echo "========== compare_graphs.sh Config =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "dataset=${dataset}"
echo "input_filename=${input_fname}.json"
echo "construct_model_name=${construct_model}"
echo "retrieval_strategy=${retrieval_strategy}"
echo "bm25_top_k=${bm25_top_k}"
echo "setting=${setting}"
echo "max_samples=${max_samples}"
echo "force_cot_regen=${force_cot_regen}"
echo "cot_retry=${cot_retry}"
echo "compare_question_graph=${compare_question_graph}"
echo "use_searchr1=${use_searchr1}"
echo "nudge_searchr1=${nudge_searchr1}"
echo "multihop_top_k_per_triplet=${multihop_top_k_per_triplet}"
echo "output_dir=${output_dir}"
echo "output_filename=${output_filename:-auto}"
echo "=================================================="

cmd=(
    python -u compare_graphs.py
    --dataset "${dataset}"
    --input_filename "${input_fname}.json"
    --construct_model_name "${construct_model}"
    --bm25_top_k "${bm25_top_k}"
    --setting "${setting}"
    --max_samples "${max_samples}"
    --cot_retry "${cot_retry}"
    --retrieval_strategy "${retrieval_strategy}"
    --multihop_top_k_per_triplet "${multihop_top_k_per_triplet}"
    --output_dir "${output_dir}"
)

if [ -n "${output_filename}" ]; then
    cmd+=(--output_filename "${output_filename}")
fi

if [ "${force_cot_regen}" = true ]; then
    cmd+=(--force_cot_regen)
fi

if [ "${compare_question_graph}" = true ]; then
    cmd+=(--compare_question_graph)
fi

if [ "${use_searchr1}" = true ]; then
    cmd+=(--use_searchr1)
fi

if [ "${nudge_searchr1}" = true ]; then
    cmd+=(--nudge_searchr1)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
