#!/usr/bin/env python3
"""
10개 infill 실험 결과를 집계하여 EM/F1 비교표를 출력합니다.
각 infill 결과 JSON에 answer.py를 적용한 후 통계를 모읍니다.

사용법:
  python aggregate_infill_results.py --output_dir /path/to/infill/output
  # 또는 answer.py를 먼저 실행한 결과가 있을 때
  python aggregate_infill_results.py --answer_dir /path/to/answer/output
"""

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_GRAPHCHECK_ROOT = _SCRIPT_DIR.parent.parent
if str(_GRAPHCHECK_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRAPHCHECK_ROOT))

from utils.metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="/data3/seonhoyoo/graphcheck-qa/infilling/output",
        help="infill/answer 결과 JSON이 있는 루트 디렉토리")
    p.add_argument("--dataset", type=str, default="all", choices=["all"] + list(DATASETS),
        help="데이터셋: all | 2wikimultihopqa | hotpotqa | musique")
    return p.parse_args()


def load_answer_results(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(samples: list) -> tuple:
    em_scores, f1_scores = [], []
    for s in samples:
        pred = s.get("predicted_answer", "")
        gt_list = [s.get("answer", "")] + s.get("answer_aliases", [])
        gt_list = [x for x in gt_list if isinstance(x, str) and x.strip()]
        if gt_list:
            em = metric_max_over_ground_truths(compute_exact, pred, gt_list)
            f1 = metric_max_over_ground_truths(compute_f1, pred, gt_list)
            em_scores.append(em)
            f1_scores.append(f1)
    n = len(em_scores)
    avg_em = sum(em_scores) / n if n else 0.0
    avg_f1 = sum(f1_scores) / n if n else 0.0
    return avg_em, avg_f1, n


def main():
    args = parse_args()
    base_dir = Path(args.output_dir)

    # 데이터셋별 서브디렉토리 또는 루트
    if args.dataset == "all":
        search_dirs = [base_dir / ds for ds in DATASETS if (base_dir / ds).exists()]
        if not search_dirs:
            search_dirs = [base_dir]
    else:
        search_dirs = [base_dir / args.dataset] if (base_dir / args.dataset).exists() else [base_dir]

    # answer_*.json 우선 (메트릭 있음), 없으면 infill_*.json
    patterns = ("answer_*.json", "infill_*.json")
    files = []
    seen = set()
    for d in search_dirs:
        for pat in patterns:
            for f in sorted(d.glob(pat)):
                if f.name not in seen:
                    seen.add(f.name)
                    files.append(f)
    files.sort(key=lambda x: (str(x.parent), x.name))

    if not files:
        print(f"No files matching {pattern} in {base_dir}")
        print("Run infill_graphs.py and answer.py first.")
        return

    rows = []
    for f in files:
        dataset = f.parent.name if f.parent.name in DATASETS else "-"
        try:
            data = load_answer_results(f)
        except Exception as e:
            print(f"Skip {f.name}: {e}")
            continue

        if any("predicted_answer" in s for s in data):
            avg_em, avg_f1, n = compute_metrics(data)
            rows.append({
                "dataset": dataset,
                "file": f.name,
                "n": n,
                "avg_em": avg_em,
                "avg_f1": avg_f1,
            })
        else:
            rows.append({
                "dataset": dataset,
                "file": f.name,
                "n": len(data),
                "avg_em": None,
                "avg_f1": None,
            })

    print("\n" + "=" * 95)
    print("Infill Experiment Results Summary")
    print("=" * 95)
    print(f"{'Dataset':<18} {'File':<55} {'N':>6} {'EM':>8} {'F1':>8}")
    print("-" * 95)
    for r in rows:
        em_str = f"{r['avg_em']:.4f}" if r["avg_em"] is not None else "N/A"
        f1_str = f"{r['avg_f1']:.4f}" if r["avg_f1"] is not None else "N/A"
        print(f"{r['dataset']:<18} {r['file']:<55} {r['n']:>6} {em_str:>8} {f1_str:>8}")
    print("=" * 95)


if __name__ == "__main__":
    main()
