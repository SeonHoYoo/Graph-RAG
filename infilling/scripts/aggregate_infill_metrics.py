#!/usr/bin/env python3
"""
Answer 정답 기준 EM/F1을 집계하여 정리합니다.

사용법:
  # 1. answer.py 실행 후 (run_answer_experiments.sh 또는 수동)
  python aggregate_infill_metrics.py --output_dir /data3/seonhoyoo/graphcheck-qa/infilling/output

  # 2. answer를 먼저 실행하고 바로 집계
  python aggregate_infill_metrics.py --output_dir ... --run_answer --model_name gpt-4o-mini
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_GRAPHCHECK_ROOT = _SCRIPT_DIR.parent.parent
if str(_GRAPHCHECK_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRAPHCHECK_ROOT))

from utils.metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths

DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
ANSWER_PY = _SCRIPT_DIR / "scripts" / "answer.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="/data3/seonhoyoo/graphcheck-qa/infilling/output")
    p.add_argument("--run_answer", action="store_true", help="각 infill에 대해 answer.py 실행 후 집계")
    p.add_argument("--model_name", type=str, default="gpt-4o-mini", help="--run_answer 시 사용할 모델")
    p.add_argument("--ent_exist_flag", type=str, default="all", choices=["all", "false"])
    p.add_argument("--out_csv", type=str, default=None, help="결과 CSV 저장 경로")
    return p.parse_args()


def parse_infill_filename(name: str) -> dict:
    """infill_triplets_triplet_only_triplet_only_gold_fulldoc_all.json -> use_full_doc, triplet_selection"""
    # 패턴: infill_*_gold_(triplet|fulldoc|doconly)_(all|top1|top3|top5|top10).json
    m = re.search(r"gold_(triplet|fulldoc|doconly)_(all|top\d+)$", name.replace(".json", ""))
    if m:
        tag = m.group(1)
        if tag == "doconly":
            return {"use_full_doc": 2, "triplet_selection": "doconly"}
        return {
            "use_full_doc": 1 if tag == "fulldoc" else 0,
            "triplet_selection": m.group(2),
        }
    return {"use_full_doc": -1, "triplet_selection": "?"}


def compute_metrics(samples: list) -> tuple:
    em_scores, f1_scores = [], []
    for s in samples:
        # Prefer per-sample answer metrics precomputed by answer.py / single_model_pipeline.
        em_raw = s.get("em_score")
        f1_raw = s.get("f1_score")
        if isinstance(em_raw, (int, float)) and isinstance(f1_raw, (int, float)):
            em_scores.append(float(em_raw))
            f1_scores.append(float(f1_raw))
            continue

        # Backward-compatible fallback for legacy outputs without em_score/f1_score.
        pred = s.get("predicted_answer", "")
        aliases = s.get("answer_aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        # Guard against contaminated outputs where prediction was appended to answer_aliases.
        gt_list = [s.get("answer", "")] + [x for x in aliases if isinstance(x, str) and x.strip() != str(pred).strip()]
        gt_list = [x for x in gt_list if isinstance(x, str) and x.strip()]
        if gt_list:
            em_scores.append(metric_max_over_ground_truths(compute_exact, pred, gt_list))
            f1_scores.append(metric_max_over_ground_truths(compute_f1, pred, gt_list))
    n = len(em_scores)
    avg_em = sum(em_scores) / n if n else 0.0
    avg_f1 = sum(f1_scores) / n if n else 0.0
    return avg_em, avg_f1, n


def run_answer(infill_path: Path, output_dir: Path, model: str, ent_flag: str) -> Path:
    """answer.py 실행 후 결과 파일 경로 반환"""
    out_name = f"answer_{infill_path.stem[len('infill_'):]}_{ent_flag}.json"
    out_path = output_dir / out_name
    if out_path.exists():
        return out_path
    cmd = [
        sys.executable, "-u", str(ANSWER_PY),
        "--model_name", model,
        "--data_file", str(infill_path),
        "--output_dir", str(output_dir),
        "--ent_exist_flag", ent_flag,
        "--max_trials", "3",
    ]
    subprocess.run(cmd, check=True)
    return out_path


def main():
    args = parse_args()
    base = Path(args.output_dir)
    model_short = args.model_name.split("/")[-1] if args.model_name else None

    # infill 파일 수집 (데이터셋별)
    # 우선순위:
    # 1) output/infill/{model_short}/{ds}/
    # 2) output/infill/{ds}/ (legacy)
    rows = []
    for ds in DATASETS:
        infill_dir = base / "infill" / model_short / ds
        answer_dir = base / "answer" / model_short / ds
        if not infill_dir.is_dir():
            infill_dir = base / "infill" / ds
            answer_dir = base / "answer" / ds
        if not infill_dir.is_dir():
            continue
        for f in sorted(infill_dir.glob("infill_*.json")):
            meta = parse_infill_filename(f.name)
            # answer.py 출력: answer_{input_stem without infill_}_{ent_exist_flag}.json
            stem = f.stem[len("infill_"):] if f.stem.startswith("infill_") else f.stem
            answer_path = answer_dir / f"answer_{stem}_{args.ent_exist_flag}.json"

            if args.run_answer:
                answer_dir.mkdir(parents=True, exist_ok=True)
                try:
                    answer_path = run_answer(f, answer_dir, args.model_name, args.ent_exist_flag)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: answer failed for {f.name}: {e}")
                    rows.append({
                        "dataset": ds,
                        "use_full_doc": meta["use_full_doc"],
                        "triplet_selection": meta["triplet_selection"],
                        "n": 0, "em": None, "f1": None,
                    })
                    continue

            if not answer_path.exists():
                rows.append({
                    "dataset": ds,
                    "use_full_doc": meta["use_full_doc"],
                    "triplet_selection": meta["triplet_selection"],
                    "n": 0, "em": None, "f1": None,
                })
                continue

            try:
                with open(answer_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception as e:
                print(f"Skip {answer_path.name}: {e}")
                continue

            avg_em, avg_f1, n = compute_metrics(data)
            rows.append({
                "dataset": ds,
                "use_full_doc": meta["use_full_doc"],
                "triplet_selection": meta["triplet_selection"],
                "n": n,
                "em": avg_em,
                "f1": avg_f1,
            })

    if not rows:
        print("No results to aggregate. Run answer.py first: sbatch run_answer_experiments.sh")
        return

    # CSV 저장
    csv_path = args.out_csv or str(base / "infill_metrics_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("dataset,use_full_doc,triplet_selection,n,em,f1\n")
        for r in rows:
            em_s = f"{r['em']:.4f}" if r["em"] is not None else ""
            f1_s = f"{r['f1']:.4f}" if r["f1"] is not None else ""
            fp.write(f"{r['dataset']},{r['use_full_doc']},{r['triplet_selection']},{r['n']},{em_s},{f1_s}\n")
    print(f"\nCSV saved: {csv_path}")

    # 표 출력 (피벗: dataset x (use_full_doc, triplet_selection))
    print("\n" + "=" * 100)
    print("Answer Correctness EM/F1 Summary")
    print("=" * 100)

    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        print(f"\n--- {ds} ---")
        # use_full_doc 0, 1, 2(doconly) 순 / triplet all, top1, top3, top5, top10, doconly 순
        trip_order = ["all", "top1", "top3", "top5", "top10"]
        for ud in [0, 1, 2]:
            ud_label = "fulldoc" if ud == 1 else ("doconly" if ud == 2 else "triplet")
            print(f"  [{ud_label}] ", end="")
            if ud == 2:
                r = next((x for x in ds_rows if x["use_full_doc"] == 2), None)
                if r and r["em"] is not None:
                    print(f"EM={r['em']:.3f} F1={r['f1']:.3f}")
                else:
                    print("--")
            else:
                for ts in trip_order:
                    r = next((x for x in ds_rows if x["use_full_doc"] == ud and x["triplet_selection"] == ts), None)
                    if r and r["em"] is not None:
                        print(f"{ts}: EM={r['em']:.3f} F1={r['f1']:.3f}  ", end="")
                    else:
                        print(f"{ts}: --  ", end="")
                print()
    print("=" * 100)

    # 마크다운 테이블 (간단 버전)
    print("\n### Markdown Table ###\n")
    print("| dataset | use_full_doc | triplet | n | EM | F1 |")
    print("|---------|--------------|---------|---|-----|-----|")
    for r in sorted(rows, key=lambda x: (x["dataset"], x["use_full_doc"], x["triplet_selection"])):
        em_s = f"{r['em']:.3f}" if r["em"] is not None else "-"
        f1_s = f"{r['f1']:.3f}" if r["f1"] is not None else "-"
        ud = "doconly" if r["use_full_doc"] == 2 else ("O" if r["use_full_doc"] == 1 else "X")
        print(f"| {r['dataset']} | {ud} | {r['triplet_selection']} | {r['n']} | {em_s} | {f1_s} |")


if __name__ == "__main__":
    main()
