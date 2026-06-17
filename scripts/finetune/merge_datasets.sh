#!/bin/bash
#SBATCH --job-name=merge_datasets
#SBATCH --output=/home/hyeseojeon/data/graph/logs/merge_datasets_%j.out
#SBATCH --error=/home/hyeseojeon/data/graph/logs/merge_datasets_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --exclude=master,n01,n02

set -euo pipefail

OUT="/data3/hyeseojeon/graph/results/graph/0515/all_documents.jsonl"

cat \
  /data3/hyeseojeon/graph/results/graph/0515/hotpotqa_documents.jsonl \
  /data3/hyeseojeon/graph/results/graph/0515/2wikimultihopqa_documents.jsonl \
  /data3/hyeseojeon/graph/results/graph/0515/musique_documents.jsonl \
  > "${OUT}"

echo "Total lines: $(wc -l < "${OUT}")"
echo "Saved to: ${OUT}"
