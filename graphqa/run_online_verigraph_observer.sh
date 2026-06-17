#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --nodelist=n01
#SBATCH --mem=80000MB
#SBATCH --job-name=online_obsv
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_verigraph_observer_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/graphqa/logs/online_verigraph_observer_%j.err

# Q-graph-observer Online Veri-Graph.
#
# SearchR1 sees only the original question.  The Q graph is extracted outside
# the reasoning model, then used as a hidden controller that observes each
# SearchR1 think/search/document turn, fills aligned UNKNOWN slots, and
# abstains when a verified fill fails the configured gate.
set -euo pipefail

PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"

export ONLINE_TRAJECTORY_MODE="${ONLINE_TRAJECTORY_MODE:-observer}"
export ONLINE_RETRIEVAL_MODE="${ONLINE_RETRIEVAL_MODE:-searchr1}"
export ONLINE_ANSWER_MODE="${ONLINE_ANSWER_MODE:-final_q_hint}"
export ONLINE_COSINE_ON_FAIL="${ONLINE_COSINE_ON_FAIL:-abstain}"
export ONLINE_COSINE_GATE_ON="${ONLINE_COSINE_GATE_ON:-doc}"
export ONLINE_COSINE_FILL_SOURCE="${ONLINE_COSINE_FILL_SOURCE:-doc}"
export ONLINE_DOC_RESCUE_ROUNDS="${ONLINE_DOC_RESCUE_ROUNDS:-0}"
export ONLINE_OUT_DIR="${ONLINE_OUT_DIR:-${PROJECT_ROOT}/graphqa/outputs/online_verigraph_observer}"

exec "${PROJECT_ROOT}/graphqa/run_online_verigraph.sh"
