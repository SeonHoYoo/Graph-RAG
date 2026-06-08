#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=n02
#SBATCH --mem=40000MB
#SBATCH --job-name=tasi_full
#SBATCH --cpus-per-task=4
#SBATCH --output=/data3/seonhoyoo/graphcheck-qa/logs/tasi_full_%j.log
#SBATCH --error=/data3/seonhoyoo/graphcheck-qa/logs/tasi_full_%j.err

# ============================================================================
# All-In-One sbatch — 두 실험 집중판  (TASI 답 결정에 미관여)
#
#   답 생성 파이프라인:
#     LLM 이 [question + reasoning chain (UNKNOWN slots) + slot type hints
#            + evidence triples (D 의 cosine top-K)] 만 보고
#     UNKNOWN slot 직접 infill + 최종 답 산출.
#     TASI 의 candidate / 5-score / pre·post-gate 는 미사용.
#
#   alignment 는 오직 sentence-embedding cosine 으로만 계산:
#     (Q,D) / (Q,Sr) / (Q,T) 평균 cosine, 임계 sweep 0.30..0.70.
#
#   STEP 1.  환경 진단
#   STEP 2.  Module 1 단위 테스트
#   STEP 3.  실험 2-A : evidence-only LLM, alignment signal *OFF*
#   STEP 4.  실험 2-B : evidence-only LLM, alignment signal *ON*
#   STEP 5.  실험 1   : alignment binary × QA-correct 의 2×2 confusion matrix
#                       (LLM 재호출 0회, STEP 3 의 CSV 만 사용)
#   STEP 6.  두 실험 결과 비교 표
#
# 실행:
#   sbatch graphqa/run_tasi_full.sh
#   bash    graphqa/run_tasi_full.sh
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0) 경로/환경 변수
# ---------------------------------------------------------------------------
PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"
PY="/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3"

GATED_OFF_DIR="${PROJECT_ROOT}/graphqa/outputs/full_evidence_off"
GATED_ON_DIR="${PROJECT_ROOT}/graphqa/outputs/full_evidence_on"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${GATED_OFF_DIR}" "${GATED_ON_DIR}" "${LOG_DIR}"

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
echo "[TASI] start: ${START_TS}"
echo "[TASI] PROJECT_ROOT     = ${PROJECT_ROOT}"
echo "[TASI] PY               = ${PY}"
echo "[TASI] GATED_OFF_DIR    = ${GATED_OFF_DIR}"
echo "[TASI] GATED_ON_DIR     = ${GATED_ON_DIR}"
echo "[TASI] PYTHONPATH       = ${PYTHONPATH}"
echo "[TASI] SLURM_JOB_ID     = ${SLURM_JOB_ID:-(none)}"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 1) 환경 진단
# ---------------------------------------------------------------------------
echo
echo "[STEP 1] env diagnostics"
echo "------------------------------------------------------------"
"${PY}" --version
echo "GPU info:"
(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true) | sed 's/^/  /'

"${PY}" - <<'PYEOF'
import sys
print(f"  python : {sys.version.split()[0]}")
mods = ["numpy", "scipy", "pandas", "networkx", "sklearn",
        "torch", "transformers", "sentence_transformers", "matplotlib", "tqdm"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  {m:<22s} {getattr(mod, '__version__', '(no __version__)')}")
    except Exception as e:
        print(f"  {m:<22s} MISSING ({e.__class__.__name__})")
import torch
print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  torch.cuda.device_name  = {torch.cuda.get_device_name(0)}")
PYEOF

# ---------------------------------------------------------------------------
# STEP 2) 단위 테스트
# ---------------------------------------------------------------------------
echo
echo "[STEP 2] Module 1 unit tests"
echo "------------------------------------------------------------"
"${PY}" "${PROJECT_ROOT}/graphqa/tests/test_tasi_core.py"

# ---------------------------------------------------------------------------
# STEP 3) 실험 2-A : TASI-gated LLM, alignment signal OFF
# ---------------------------------------------------------------------------
echo
echo "[STEP 3] Experiment 2-A — evidence-only LLM (alignment signal OFF)"
echo "  output_dir = ${GATED_OFF_DIR}"
echo "------------------------------------------------------------"
"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_eval.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${GATED_OFF_DIR}" \
    "${LLM_FLAGS[@]}" \
    --save-plots

# ---------------------------------------------------------------------------
# STEP 4) 실험 2-B : TASI-gated LLM, alignment signal ON
# ---------------------------------------------------------------------------
echo
echo "[STEP 4] Experiment 2-B — evidence-only LLM (alignment signal ON)"
echo "  output_dir = ${GATED_ON_DIR}"
echo "------------------------------------------------------------"
"${PY}" "${PROJECT_ROOT}/graphqa/scripts/run_eval.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${GATED_ON_DIR}" \
    "${LLM_FLAGS[@]}" \
    --inject-alignment-signal \
    --save-plots

# ---------------------------------------------------------------------------
# STEP 5) 실험 1 : alignment 0/1 × QA-correct 2×2 confusion matrix
#         (LLM 재호출 없음. STEP 3 의 CSV 를 사용)
# ---------------------------------------------------------------------------
echo
echo "[STEP 5] Experiment 1 — alignment binary × correctness 2x2 confusion"
echo "  using CSVs in: ${GATED_OFF_DIR}"
echo "------------------------------------------------------------"
"${PY}" -m graphqa.scripts.alignment_confusion \
    --input-dir "${GATED_OFF_DIR}" \
    --thresholds 0.30 0.40 0.50 0.60 0.70 \
    --save-plots

# (보조) signal-ON 결과에 대해서도 동일 분석 — 비교용
echo
echo "[STEP 5b] Experiment 1 — same analysis on signal-ON CSVs (sanity)"
echo "------------------------------------------------------------"
"${PY}" -m graphqa.scripts.alignment_confusion \
    --input-dir "${GATED_ON_DIR}" \
    --thresholds 0.30 0.40 0.50 0.60 0.70 \
    --save-plots

# ---------------------------------------------------------------------------
# STEP 6) 두 실험 비교 표
# ---------------------------------------------------------------------------
echo
echo "[STEP 6] Compare signal OFF vs ON  (per-dataset accuracy)"
echo "------------------------------------------------------------"
"${PY}" - <<PYEOF
import json, os
from pathlib import Path

OFF = Path("${GATED_OFF_DIR}")
ON  = Path("${GATED_ON_DIR}")

def load_summary(d):
    rows = []
    for f in sorted(d.glob("*/tasi_eval_*_summary.json")):
        ds = f.parent.name
        with open(f) as h:
            j = json.load(h)
        rows.append((ds, j.get("n_samples"),
                     j.get("accuracy_final"),
                     j.get("em_mean"), j.get("f1_mean"),
                     j.get("yesno_accuracy"), j.get("open_accuracy"),
                     j.get("abstain_rate"),
                     j.get("accuracy_when_answered"),
                     j.get("abstain_reason_distribution", {})))
    return rows

off_rows = load_summary(OFF)
on_rows  = load_summary(ON)

print(f"\n{'dataset':<22s}  {'n':>5s}  {'acc(OFF)':>9s}  {'acc(ON)':>8s}  "
      f"{'Δ':>+7s}  {'f1(OFF)':>8s}  {'f1(ON)':>7s}  "
      f"{'abst(OFF)':>9s}  {'abst(ON)':>8s}  "
      f"{'acc_ans(OFF)':>12s}  {'acc_ans(ON)':>11s}")
print("-" * 130)
on_map = {r[0]: r for r in on_rows}
for ro in off_rows:
    ds, n, a_off, em_off, f1_off, _, _, abst_off, aans_off, _ = ro
    if ds not in on_map:
        continue
    rn = on_map[ds]
    _, _, a_on, _, f1_on, _, _, abst_on, aans_on, _ = rn
    delta = (a_on or 0) - (a_off or 0)
    print(f"{ds:<22s}  {n:>5d}  {a_off:>9.3f}  {a_on:>8.3f}  {delta:>+7.3f}  "
          f"{f1_off:>8.3f}  {f1_on:>7.3f}  "
          f"{abst_off or 0:>9.3f}  {abst_on or 0:>8.3f}  "
          f"{aans_off or 0:>12.3f}  {aans_on or 0:>11.3f}")
PYEOF

# ---------------------------------------------------------------------------
# 결과 위치 안내
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "[TASI] DONE. Result locations:"
echo "  Experiment 2-A (signal OFF) : ${GATED_OFF_DIR}"
echo "  Experiment 2-B (signal ON)  : ${GATED_ON_DIR}"
echo "  Experiment 1 (confusion)    : ${GATED_OFF_DIR}/align_confusion_aggregate.json"
echo "  Comparison table            : (printed above in STEP 6)"
echo
echo "Inspect per-dataset summary:"
echo "  cat ${GATED_OFF_DIR}/<dataset>/tasi_eval_<dataset>_summary.json"
echo "  cat ${GATED_ON_DIR}/<dataset>/tasi_eval_<dataset>_summary.json"
echo
echo "Inspect confusion matrices:"
echo "  cat ${GATED_OFF_DIR}/<dataset>/tasi_eval_<dataset>_align_confusion.json"
echo "  open ${GATED_OFF_DIR}/<dataset>/tasi_eval_<dataset>_align_confusion.png"
echo "============================================================"
