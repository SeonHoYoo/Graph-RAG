#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/compare_graphs_%j.log
#SBATCH --error=./logs/compare_graphs_%j.err

source /data3/seonhoyoo/.bashrc
source /data3/seonhoyoo/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

# 작업 디렉토리로 이동
cd /data3/seonhoyoo/graphcheck-qa

# Hugging Face 캐시 경로 고정(노드가 달라도 동일 경로 사용)
export HF_HOME=/data3/seonhoyoo/.cache/huggingface
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# Transformers 업그레이드 (Qwen2.5 지원) - graphcheck 환경에 설치됨
pip install --upgrade transformers>=4.37.0 --quiet

input_fname="train_sampled"
dataset="musique"

# Qwen2.5 모델 사용
construct_model="Qwen/Qwen2.5-7B-Instruct"
# construct_model="Qwen/Qwen2.5-32B-Instruct"  # 더 작은 모델 사용 시
# construct_model="Qwen/Qwen2.5-14B-Instruct"  # 더 작은 모델 사용 시

# 검색 전략 설정
# 옵션: question, cot_reasoning, triplets, combined, multihop_triplets
# - question: 질문만으로 검색 (기본 BM25)
# - cot_reasoning: CoT reasoning을 thinking으로 사용 (SearchR1 필요)
# - triplets: CoT triplets를 검색 쿼리로 사용
# - combined: 질문 + triplets + CoT reasoning 모두 사용
# - multihop_triplets: 각 triplet별로 멀티홉 검색 수행
# - question_triplets: GraphCheck 스타일 질문 triplets로 검색
retrieval_strategy="question_triplets" # cot_reasoning, question_triplets

# SearchR1 사용 여부 (cot_reasoning, combined 전략 사용 시 권장)
use_searchr1=false
# use_searchr1=true

# Nudge 모델과 함께 SearchR1 사용 (use_searchr1=true일 때만)
nudge_searchr1=false
# nudge_searchr1=true

# 멀티홉 검색 시 각 triplet당 검색할 문서 수 (multihop_triplets 전략 사용 시)
multihop_top_k_per_triplet=2

python -u compare_graphs.py \
    --dataset ${dataset} \
    --input_filename ${input_fname}.json \
    --construct_model_name ${construct_model} \
    --bm25_top_k 5 \
    --setting open-book+gold \
    --max_samples 10 \
    --force_cot_regen \
    --cot_retry 3 \
    --retrieval_strategy ${retrieval_strategy} \
    --compare_question_graph \
    $(if [ "$use_searchr1" = true ]; then echo "--use_searchr1"; fi) \
    $(if [ "$nudge_searchr1" = true ]; then echo "--nudge_searchr1"; fi) \
    --multihop_top_k_per_triplet ${multihop_top_k_per_triplet}
