#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=40000MB
#SBATCH --job-name=evidence_and
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/logs/evidence_align_and_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/logs/evidence_align_and_%j.err

# ============================================================================
# Evidence-only QA + AND-style embedding alignment (min-of-max)
#
#   Alignment (graphqa/alignment.py):
#     각 Q triple i 에 대해 m_i = max_j cos(Q_i, B_j)  (B = D / Sr / T)
#     샘플 점수 align_* = min_i m_i  →  **모든** Q 줄이 각각 어느 B 와 잘 맞아야
#     점수가 오른다 (AND).  참고용 mean-of-max 는 CSV 의 align_*_mean.
#
#   QA 파이프라인 (--qa-mode evidence):
#     LLM 만으로 chain + evidence(cosine top-K in D) infill 및 최종 답.
#     TASI candidate / gate 는 답 결정에 사용하지 않음.
#
#   실험:
#     STEP 3  signal OFF  (--inject-alignment-signal 없음)
#     STEP 4  signal ON   (Q–D / Q–Sr / Q–T 점수를 prompt 에 주입)
#     STEP 5  alignment × is_correct 2×2 (alignment_confusion.py, LLM 0회)
#     STEP 6  데이터셋별 OFF vs ON 비교 표
#
# 실행:
#   sbatch graphqa/run_evidence_align_and.sh
#   bash    graphqa/run_evidence_align_and.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"
PY="/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3"

# 이전 full_evidence_* / full_llm_gated 와 섞이지 않도록 전용 디렉터리
OUT_OFF="${PROJECT_ROOT}/graphqa/outputs/evidence_and_align_off"
OUT_ON="${PROJECT_ROOT}/graphqa/outputs/evidence_and_align_on"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${OUT_OFF}" "${OUT_ON}" "${LOG_DIR}"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data3/seonhoyoo/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

DATASETS=(2wikimultihopqa hotpotqa musique)
LLM_FLAGS=(
    --pc-mode log_mean
    --use-llm qwen-local
    --llm-model Qwen/Qwen2.5-7B-Instruct
    --llm-dtype bfloat16
    --qa-mode evidence
)

START_TS="$(date +%Y%m%d_%H%M%S)"
echo "============================================================"
echo "[evidence+AND-align] start: ${START_TS}"
echo "  PROJECT_ROOT = ${PROJECT_ROOT}"
echo "  PY           = ${PY}"
echo "  OUT_OFF      = ${OUT_OFF}"
echo "  OUT_ON       = ${OUT_ON}"
echo "  SLURM_JOB_ID = ${SLURM_JOB_ID:-(none)}"
echo "  alignment    = min-of-max over Q triples (AND); see align_*_mean for mean"
echo "============================================================"

echo
echo "[STEP 1] env diagnostics"
echo "------------------------------------------------------------"
"${PY}" --version
echo "GPU:"
(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true) | sed 's/^/  /'

echo
echo "[STEP 2] unit tests"
echo "------------------------------------------------------------"
"${PY}" "${PROJECT_ROOT}/graphqa/tests/test_tasi_core.py"

echo
echo "[STEP 3] Experiment A — evidence LLM, alignment signal OFF"
echo "  output_dir = ${OUT_OFF}"
echo "------------------------------------------------------------"
"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_eval.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${OUT_OFF}" \
    "${LLM_FLAGS[@]}" \
    --save-plots

echo
echo "[STEP 4] Experiment B — evidence LLM, alignment signal ON"
echo "  output_dir = ${OUT_ON}"
echo "------------------------------------------------------------"
"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_eval.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${OUT_ON}" \
    "${LLM_FLAGS[@]}" \
    --inject-alignment-signal \
    --save-plots

echo
echo "[STEP 5] 2×2 confusion (align min-of-max × QA correct) — signal OFF CSVs"
echo "------------------------------------------------------------"
"${PY}" -m graphqa.scripts.alignment_confusion \
    --input-dir "${OUT_OFF}" \
    --thresholds 0.30 0.40 0.50 0.60 0.70 \
    --save-plots

echo
echo "[STEP 5b] same confusion — signal ON CSVs"
echo "------------------------------------------------------------"
"${PY}" -m graphqa.scripts.alignment_confusion \
    --input-dir "${OUT_ON}" \
    --thresholds 0.30 0.40 0.50 0.60 0.70 \
    --save-plots

echo
echo "[STEP 6] Compare OFF vs ON (per-dataset)"
echo "------------------------------------------------------------"
"${PY}" - <<PYEOF
import json
from pathlib import Path

OFF = Path("${OUT_OFF}")
ON = Path("${OUT_ON}")

def load_summary(d):
    rows = []
    for f in sorted(d.glob("*/tasi_eval_*_summary.json")):
        ds = f.parent.name
        with open(f) as h:
            j = json.load(h)
        rows.append((ds, j.get("n_samples"),
                     j.get("accuracy_final"),
                     j.get("f1_mean"),
                     j.get("abstain_rate"),
                     j.get("accuracy_when_answered")))
    return rows

off_rows, on_rows = load_summary(OFF), load_summary(ON)
on_map = {r[0]: r for r in on_rows}
print(f"\n{'dataset':<22s}  {'n':>5s}  {'acc OFF':>9s}  {'acc ON':>8s}  {'Δ':>+7s}  "
      f"{'f1 OFF':>8s}  {'f1 ON':>7s}  {'abst OFF':>9s}  {'abst ON':>8s}  "
      f"{'acc@ans OFF':>12s}  {'acc@ans ON':>11s}")
print("-" * 120)
for ro in off_rows:
    ds, n, a_off, f1_off, abst_off, aans_off = ro
    if ds not in on_map:
        continue
    _, _n_on, a_on, f1_on, abst_on, aans_on = on_map[ds]
    print(f"{ds:<22s}  {n:>5d}  {a_off:>9.3f}  {a_on:>8.3f}  {(a_on or 0)-(a_off or 0):>+7.3f}  "
          f"{f1_off:>8.3f}  {f1_on:>7.3f}  "
          f"{abst_off or 0:>9.3f}  {abst_on or 0:>8.3f}  "
          f"{aans_off or 0:>12.3f}  {aans_on or 0:>11.3f}")
PYEOF

echo
echo "============================================================"
echo "DONE."
echo "  OFF: ${OUT_OFF}"
echo "  ON : ${OUT_ON}"
echo "  Confusion JSON/PNG: each <dataset>/tasi_eval_*_align_confusion.*"
echo "============================================================"
