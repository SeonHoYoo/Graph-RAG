#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=80000MB
#SBATCH --job-name=online_qgraph
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_verigraph_q_guided_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_verigraph_q_guided_%j.err

# Q-graph-guided Online Veri-Graph.
#
# This wrapper keeps the original online script intact but switches the
# trajectory mode: Q is extracted first, each unresolved Q triple becomes the
# next SearchR1 subgoal, and retrieved evidence is immediately graph-verified.
set -euo pipefail

PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"

export ONLINE_TRAJECTORY_MODE="${ONLINE_TRAJECTORY_MODE:-q_guided}"
export ONLINE_RETRIEVAL_MODE="${ONLINE_RETRIEVAL_MODE:-searchr1}"
export ONLINE_ANSWER_MODE="${ONLINE_ANSWER_MODE:-final_q_hint}"
export ONLINE_OUT_DIR="${ONLINE_OUT_DIR:-${PROJECT_ROOT}/graphqa/outputs/online_verigraph_q_guided}"

exec "${PROJECT_ROOT}/graphqa/run_online_verigraph.sh"
