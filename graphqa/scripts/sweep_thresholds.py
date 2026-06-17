"""기존에 저장된 verifier eval CSV 로 post-hoc threshold sweep + risk-coverage 분석.

LLM 을 다시 호출하지 않고, run_eval.py 가 만들어 둔 per-sample CSV 의
verify_score / is_correct / em / f1 컬럼만 가지고 selective prediction 메트릭을
재계산한다.

사용 예:
    python -m graphqa.scripts.sweep_thresholds \
        --input-dir graphqa/outputs/full_llm_verify \
        --thresholds 0.0 0.30 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
        --save-plots
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from graphqa.evaluate import (
    DEFAULT_SWEEP_THRESHOLDS,
    print_selective_report,
    selective_report,
    _save_plots,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True,
                   help="run_eval.py 가 만든 dataset 별 출력 dir 의 부모 (e.g. outputs/full_llm_verify)")
    p.add_argument("--csv-glob", type=str, default="**/tasi_eval_*.csv",
                   help="CSV 검색 패턴 (input-dir 기준).")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=list(DEFAULT_SWEEP_THRESHOLDS))
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--output-suffix", type=str, default="_selective",
                   help="JSON 출력 파일 suffix (csv 옆에 같은 stem 으로 저장).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def process_csv(csv_path: Path,
                thresholds: List[float],
                save_plots: bool,
                output_suffix: str,
                quiet: bool) -> dict:
    df = pd.read_csv(csv_path)
    if "verify_score" not in df.columns:
        logger.warning(f"[sweep] no verify_score column in {csv_path}, skipping")
        return {}
    report = selective_report(df, thresholds=thresholds)

    if not quiet:
        print(f"\n=== {csv_path}  (n={len(df)}) ===")
        if "is_correct" in df.columns:
            print(f"  overall acc (no abstain) = {df['is_correct'].mean():.3f}")
        print_selective_report(report)

    out_json = csv_path.with_name(csv_path.stem + output_suffix + ".json")
    summary_extras = {
        "csv": str(csv_path),
        "n_samples": int(len(df)),
        "overall_accuracy_no_abstain": float(df["is_correct"].mean())
            if "is_correct" in df.columns else float("nan"),
        "selective_prediction": report,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_extras, f, indent=2, ensure_ascii=False)
    logger.info(f"[sweep] saved {out_json}")

    if save_plots:
        _save_plots(df, csv_path.parent, csv_path.stem + output_suffix)

    return summary_extras


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    in_dir = args.input_dir.resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {in_dir}")
    csvs = sorted(in_dir.glob(args.csv_glob))
    csvs = [p for p in csvs
            if not p.name.endswith("_selective.csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSVs matching {args.csv_glob} in {in_dir}")

    print(f"[sweep] {len(csvs)} CSV(s) found under {in_dir}")
    aggregate = []
    for csv in csvs:
        rep = process_csv(csv, args.thresholds, args.save_plots,
                          args.output_suffix, args.quiet)
        if rep:
            aggregate.append(rep)

    agg_path = in_dir / f"sweep_aggregate{args.output_suffix}.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({"runs": aggregate}, f, indent=2, ensure_ascii=False)
    print(f"\n[sweep] aggregate written to {agg_path}")


if __name__ == "__main__":
    main()
