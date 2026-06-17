"""Graph 비교 결과(=TASI score)를 abstain threshold 로 사용했을 때
서로 다른 score recipe 의 selective-prediction 성능을 직접 비교.

연구 narrative:
  '답을 후처리로 검증' 하는 verify_score (grounding/chain/type) 가 아니라,
  Module 1 의 그래프 정렬 점수 자체 (relevance/consistency/alignment/
  search_quality/retrieval/total_sum) 가 question 의 confident 여부를
  결정하도록 한다.

이 스크립트는 LLM 을 다시 호출하지 않고, 이미 저장된 evaluator CSV 의
score column 만 가지고 percentile sweep + risk-coverage curve 를 산출한다.

사용 예:
    python -m graphqa.scripts.sweep_recipes \
        --input-dir graphqa/outputs/full_llm_verify \
        --score-cols relevance_score consistency_score alignment_score \
                     search_quality_score retrieval_score \
                     total_sum verify_score \
        --save-plots
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from graphqa.evaluate import (
    DEFAULT_SWEEP_THRESHOLDS,
    selective_report,
)


logger = logging.getLogger(__name__)


DEFAULT_SCORE_COLS = [
    "relevance_score",
    "consistency_score",
    "alignment_score",
    "search_quality_score",
    "retrieval_score",
    "total_sum",
    "verify_score",
]


# ---------------------------------------------------------------------------
# Composite recipes — TASI score 들을 합쳐 새 column 으로 등록
# ---------------------------------------------------------------------------
def _safe_norm(s: pd.Series) -> pd.Series:
    """min-max 정규화 (0~1). 분산 0이면 그대로."""
    s = s.astype(float)
    lo = float(s.min(skipna=True))
    hi = float(s.max(skipna=True))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _per_dataset_norm(df: pd.DataFrame, col: str) -> pd.Series:
    if "dataset" not in df.columns:
        return _safe_norm(df[col])
    return df.groupby("dataset")[col].transform(_safe_norm)


def add_composite_scores(df: pd.DataFrame) -> List[str]:
    """TASI 5-score 를 가공한 합성 신호 column 들을 in-place 로 추가.

    추가되는 column:
      - tasi_mean5      : 5-score 단순 평균 (per-dataset min-max norm 후)
      - tasi_min5       : 5-score 의 min  (가장 약한 신호 = 의심)
      - tasi_geo5       : 5-score 의 기하 평균
      - tasi_relcons    : (relevance + consistency)/2  (직관적으로 가장 핵심)
      - tasi_align_ret  : (alignment + retrieval)/2
    """
    added: List[str] = []
    base = ["relevance_score", "consistency_score", "alignment_score",
            "search_quality_score", "retrieval_score"]
    if not all(c in df.columns for c in base):
        return added
    norm_cols = []
    for c in base:
        nc = f"{c}__norm"
        df[nc] = _per_dataset_norm(df, c)
        norm_cols.append(nc)

    df["tasi_mean5"] = df[norm_cols].mean(axis=1)
    df["tasi_min5"] = df[norm_cols].min(axis=1)
    eps = 1e-6
    df["tasi_geo5"] = np.exp(np.log(df[norm_cols].clip(lower=eps)).mean(axis=1))
    df["tasi_relcons"] = (df["relevance_score__norm"] + df["consistency_score__norm"]) / 2.0
    df["tasi_align_ret"] = (df["alignment_score__norm"] + df["retrieval_score__norm"]) / 2.0
    added += ["tasi_mean5", "tasi_min5", "tasi_geo5", "tasi_relcons", "tasi_align_ret"]
    return added


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _plot_overlay(reports: Dict[str, Dict[str, object]],
                  overall_acc: float,
                  out_path: Path,
                  title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping overlay plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rand_risk = 1.0 - overall_acc
    for name, rep in reports.items():
        curve = rep.get("risk_coverage_curve", {}) or {}
        cov = curve.get("coverage", [])
        risk = curve.get("risk", [])
        acc = curve.get("acc_when_answered", [])
        if not cov:
            continue
        aurc = rep.get("aurc", float("nan"))
        axes[0].plot(cov, risk, lw=1.6, label=f"{name} (AURC={aurc:.3f})")
        axes[1].plot(cov, acc, lw=1.6, label=f"{name}")
    axes[0].plot([0, 1], [rand_risk, rand_risk], "--", color="gray",
                 label=f"random ({rand_risk:.2f})")
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Risk = 1 − acc(answered)")
    axes[0].set_title("Risk–Coverage")
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(overall_acc, color="gray", ls="--",
                    label=f"overall acc = {overall_acc:.3f}")
    axes[1].set_xlabel("Coverage")
    axes[1].set_ylabel("Selective accuracy (answered)")
    axes[1].set_title("Selective Accuracy vs Coverage")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"saved overlay plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True,
                   help="evaluation 결과 dir 의 부모 (e.g. outputs/full_llm_verify).")
    p.add_argument("--csv-glob", type=str, default="**/tasi_eval_*.csv")
    p.add_argument("--score-cols", type=str, nargs="+", default=DEFAULT_SCORE_COLS,
                   help="비교할 score column 들. add_composite_scores 가 추가하는 "
                        "tasi_mean5/min5/geo5/relcons/align_ret 도 사용 가능.")
    p.add_argument("--use-composites", action="store_true", default=True,
                   help="합성 score (tasi_mean5 등) 자동 생성 후 비교에 포함.")
    p.add_argument("--no-composites", dest="use_composites",
                   action="store_false")
    p.add_argument("--coverage-grid", type=float, nargs="+",
                   default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
                            0.80, 0.90, 1.00])
    p.add_argument("--abs-thresholds", type=float, nargs="+",
                   default=list(DEFAULT_SWEEP_THRESHOLDS))
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def process_one_csv(csv_path: Path,
                    score_cols: List[str],
                    coverage_grid: Sequence[float],
                    abs_thresholds: Sequence[float],
                    add_composites: bool,
                    save_plots: bool,
                    quiet: bool) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    if "is_correct" not in df.columns:
        logger.warning(f"no is_correct in {csv_path}, skipping")
        return {}
    if add_composites:
        add_composite_scores(df)

    available = [c for c in score_cols if c in df.columns]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        logger.warning(f"[{csv_path.name}] missing score cols: {missing}")

    overall_acc = float(df["is_correct"].mean())
    print("\n" + "=" * 90)
    print(f"# {csv_path}")
    print(f"  n={len(df)}  overall_acc(no-abstain)={overall_acc:.3f}")
    print("=" * 90)

    reports: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    for col in available:
        rep = selective_report(df,
                               thresholds=abs_thresholds,
                               score_col=col,
                               coverage_grid=coverage_grid)
        reports[col] = rep
        summary_rows.append({
            "score_col": col,
            "n": len(df),
            "overall_acc": overall_acc,
            "score_auc": rep.get("score_auc"),
            "aurc": rep.get("aurc"),
            "e_aurc": rep.get("e_aurc"),
            "acc@cov=0.30": rep.get("acc_at_coverage", {}).get("acc@cov=0.30"),
            "acc@cov=0.50": rep.get("acc_at_coverage", {}).get("acc@cov=0.50"),
            "acc@cov=0.70": rep.get("acc_at_coverage", {}).get("acc@cov=0.70"),
        })

    if not quiet:
        sdf = pd.DataFrame(summary_rows).sort_values("e_aurc", ascending=False)
        # 사람이 읽기 좋은 표
        print(f"  {'score_col':>22s}  {'AUC':>5s}  {'AURC':>6s}  {'E-AURC':>7s}  "
              f"{'acc@30%':>8s}  {'acc@50%':>8s}  {'acc@70%':>8s}")
        for _, r in sdf.iterrows():
            print(f"  {r['score_col']:>22s}  "
                  f"{(r['score_auc'] if r['score_auc'] is not None else float('nan')):>5.3f}  "
                  f"{(r['aurc'] if r['aurc'] is not None else float('nan')):>6.3f}  "
                  f"{(r['e_aurc'] if r['e_aurc'] is not None else float('nan')):>+7.3f}  "
                  f"{(r['acc@cov=0.30'] if r['acc@cov=0.30'] is not None else float('nan')):>8.3f}  "
                  f"{(r['acc@cov=0.50'] if r['acc@cov=0.50'] is not None else float('nan')):>8.3f}  "
                  f"{(r['acc@cov=0.70'] if r['acc@cov=0.70'] is not None else float('nan')):>8.3f}")

    if save_plots and reports:
        out_png = csv_path.with_name(csv_path.stem + "_recipes_overlay.png")
        _plot_overlay(reports, overall_acc, out_png,
                      title=f"{csv_path.stem}  (n={len(df)})")

    out_json = csv_path.with_name(csv_path.stem + "_recipes.json")
    payload = {
        "csv": str(csv_path),
        "n_samples": len(df),
        "overall_acc_no_abstain": overall_acc,
        "summary_per_recipe": summary_rows,
        "reports": reports,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
    logger.info(f"saved {out_json}")
    return payload


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    in_dir = args.input_dir.resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {in_dir}")
    csvs = sorted(in_dir.glob(args.csv_glob))
    csvs = [p for p in csvs
            if not p.name.endswith("_selective.csv")
            and not p.name.endswith("_recipes.csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSVs matching {args.csv_glob} in {in_dir}")

    print(f"[recipes] {len(csvs)} CSV(s) under {in_dir}")
    payloads = []
    for csv in csvs:
        rep = process_one_csv(
            csv,
            score_cols=args.score_cols,
            coverage_grid=args.coverage_grid,
            abs_thresholds=args.abs_thresholds,
            add_composites=args.use_composites,
            save_plots=args.save_plots,
            quiet=args.quiet,
        )
        if rep:
            payloads.append(rep)

    agg = in_dir / "recipes_aggregate.json"
    with open(agg, "w", encoding="utf-8") as f:
        json.dump({"runs": payloads}, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n[recipes] aggregate written to {agg}")


if __name__ == "__main__":
    main()
