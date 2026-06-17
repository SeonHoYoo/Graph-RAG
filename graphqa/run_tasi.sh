#!/usr/bin/env bash
# ============================================================================
# TASI: Triplet Alignment with Structural Importance
#
# Usage:
#   bash graphqa/run_tasi.sh                # 전체 데이터셋 평가
#   bash graphqa/run_tasi.sh quick          # 각 데이터셋 30개 샘플로 빠른 검증
#   bash graphqa/run_tasi.sh single         # 단일 샘플 디버깅 출력
#   bash graphqa/run_tasi.sh test           # Module 1 단위 테스트
# ============================================================================
set -euo pipefail

PROJECT_ROOT="/data3/seonhoyoo/graphcheck-qa"
PY="/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3"
MODE="${1:-full}"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
    test)
        echo "[+] Running Module 1 unit tests"
        "${PY}" graphqa/tests/test_tasi_core.py
        ;;
    single)
        echo "[+] Running single-sample debug output (2wiki x 3)"
        "${PY}" graphqa/scripts/show_single.py --dataset 2wikimultihopqa --n 3
        ;;
    quick)
        echo "[+] Quick evaluation (each dataset, limit=30)"
        "${PY}" graphqa/scripts/run_eval.py \
            --datasets 2wikimultihopqa hotpotqa musique \
            --limit 30 \
            --output-dir graphqa/outputs/quick \
            --save-plots
        ;;
    full)
        echo "[+] Full evaluation on all 3 datasets (2000 samples total)"
        "${PY}" graphqa/scripts/run_eval.py \
            --datasets 2wikimultihopqa hotpotqa musique \
            --output-dir graphqa/outputs/full \
            --save-plots
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: bash graphqa/run_tasi.sh [test|single|quick|full]"
        exit 1
        ;;
esac

echo "[+] Done."
