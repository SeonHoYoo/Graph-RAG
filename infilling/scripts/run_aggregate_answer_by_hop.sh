#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_aggregate_answer_by_hop_%j.out
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/infilling/sample/run_aggregate_answer_by_hop_%j.err
# Answer EM/F1을 num_hops 기준으로 집계 (모델/실험별 answer 루트 지정)
# 출력: output/answer_em_f1_by_hop_<model_or_tag>.csv, .md
set -euo pipefail

# 프로젝트 scripts 디렉터리 (SLURM job CWD가 spool일 때 절대경로 폴백)
INFILLING_SCRIPTS="/data3/seonhoyoo/graphcheck-qa/infilling/scripts"
OUTPUT_BASE="/data3/seonhoyoo/graphcheck-qa/infilling/output"
MODEL_OUTPUT_DIR="${1:-Qwen2.5-7B-Instruct}"
try_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [[ -n "$try_dir" && -f "${try_dir}/aggregate_answer_metrics_by_hop.py" ]]; then
  INFILLING_SCRIPTS="$try_dir"
fi

cd "$INFILLING_SCRIPTS"
exec python3 "${INFILLING_SCRIPTS}/aggregate_answer_metrics_by_hop.py" \
  --answer_root "${OUTPUT_BASE}/answer/${MODEL_OUTPUT_DIR}" \
  --output_dir "${OUTPUT_BASE}" \
  --out_csv "${OUTPUT_BASE}/answer_em_f1_by_hop_${MODEL_OUTPUT_DIR}.csv" \
  --out_md "${OUTPUT_BASE}/answer_em_f1_by_hop_${MODEL_OUTPUT_DIR}.md"
