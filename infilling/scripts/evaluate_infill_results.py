#!/usr/bin/env python3
"""
predicted_answer가 있는 JSON을 로드하여 EM/F1만 집계합니다.
모델 호출 없음. answer.py 실행 결과(answer_*.json) 또는 predicted_answer가 있는 파일을 대상으로 합니다.
"""

import argparse
import json
import re
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
    p.add_argument("--output_dir", type=str, default="/data3/seonhoyoo/graphcheck-qa/infilling/output")
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--quiet", "-q", action="store_true", help="결과 테이블만 출력")
    return p.parse_args()


def parse_filename(name: str) -> dict:
    stem = name.replace(".json", "")
    # answer_triplets_..._gold_fulldoc_all_all.json 등에서 gold_fulldoc_all 추출
    m = re.search(r"gold_(triplet|fulldoc|doconly)_(all|top\d+|doconly)(?:_all)?$", stem)
    if m:
        tag = m.group(1)
        if tag == "doconly":
            return {"use_full_doc": 2, "triplet_selection": "doconly"}
        return {"use_full_doc": 1 if tag == "fulldoc" else 0, "triplet_selection": m.group(2)}
    return {"use_full_doc": -1, "triplet_selection": "?"}


def compute_metrics(samples: list) -> tuple:
    em_list, f1_list = [], []
    for s in samples:
        pred = s.get("predicted_answer", "")
        if pred is None:
            pred = ""
        gt = [s.get("answer", "")] + s.get("answer_aliases", [])
        gt = [x for x in gt if isinstance(x, str) and x.strip()]
        if gt:
            em_list.append(metric_max_over_ground_truths(compute_exact, pred, gt))
            f1_list.append(metric_max_over_ground_truths(compute_f1, pred, gt))
    n = len(em_list)
    avg_em = sum(em_list) / n if n else 0.0
    avg_f1 = sum(f1_list) / n if n else 0.0
    return n, avg_em, avg_f1


def main():
    args = parse_args()
    base = Path(args.output_dir)

    # answer 파일: output/answer/{ds}/answer_*.json
    rows = []
    for ds in DATASETS:
        answer_dir = base / "answer" / ds
        if not answer_dir.is_dir():
            continue
        for f in sorted(answer_dir.glob("answer_*.json")):
            meta = parse_filename(f.name)
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception as e:
                print(f"Skip {f.name}: {e}")
                continue
            if not any("predicted_answer" in s for s in data):
                continue
            n, em, f1 = compute_metrics(data)
            rows.append({
                "dataset": ds,
                "use_full_doc": meta["use_full_doc"],
                "triplet_selection": meta["triplet_selection"],
                "n": n,
                "em": em,
                "f1": f1,
            })

    if not rows:
        print("No files with predicted_answer found.\nRun answer.py first: sbatch run_answer_and_aggregate.sh")
        return

    csv_path = args.out_csv or str(base / "infill_metrics_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("dataset,use_full_doc,triplet_selection,n,em,f1\n")
        for r in rows:
            fp.write(f"{r['dataset']},{r['use_full_doc']},{r['triplet_selection']},{r['n']},{r['em']:.4f},{r['f1']:.4f}\n")
    if not getattr(args, "quiet", False):
        print(f"CSV saved: {csv_path}")

    print("\n" + "=" * 100)
    print("Infill EM/F1 Summary")
    print("=" * 100)
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        print(f"\n--- {ds} ---")
        for ud in [0, 1, 2]:
            lbl = "fulldoc" if ud == 1 else ("doconly" if ud == 2 else "triplet")
            print(f"  [{lbl}] ", end="")
            if ud == 2:
                r = next((x for x in ds_rows if x["use_full_doc"] == 2), None)
                if r:
                    print(f"EM={r['em']:.3f} F1={r['f1']:.3f}")
                else:
                    print("--")
            else:
                for ts in ["all", "top1", "top3", "top5", "top10"]:
                    r = next((x for x in ds_rows if x["use_full_doc"] == ud and x["triplet_selection"] == ts), None)
                    if r:
                        print(f"{ts}: EM={r['em']:.3f} F1={r['f1']:.3f}  ", end="")
                    else:
                        print(f"{ts}: --  ", end="")
                print()
    print("=" * 100)

    print("\n| dataset | use_full_doc | triplet | n | EM | F1 |")
    print("|---------|--------------|---------|---|-----|-----|")
    for r in sorted(rows, key=lambda x: (x["dataset"], x["use_full_doc"], x["triplet_selection"])):
        ud = "doconly" if r["use_full_doc"] == 2 else ("O" if r["use_full_doc"] == 1 else "X")
        print(f"| {r['dataset']} | {ud} | {r['triplet_selection']} | {r['n']} | {r['em']:.3f} | {r['f1']:.3f} |")


if __name__ == "__main__":
    main()
