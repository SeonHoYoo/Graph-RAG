"""실험 1 — alignment binary × QA-correct 의 2×2 confusion matrix 분석.

CSV 에 이미 저장된 align_QD / align_QSr / align_QT 점수와 is_correct 컬럼을
사용해, 3개 페어 × 여러 threshold 에서 confusion matrix 를 만든다.
(align_* 점수는 **min-of-max(AND)** 집계: Q의 모든 triple 이 각각 τ 이상으로
맞아야 샘플이 align Yes.  참고용 mean 집계는 align_*_mean 컬럼.)
LLM 재호출 0회. encoder 도 안 씀.

사용 예:
    python -m graphqa.scripts.alignment_confusion \
        --input-dir graphqa/outputs/full_llm_gated \
        --thresholds 0.30 0.40 0.50 0.60 0.70 \
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

from graphqa.alignment import (
    DEFAULT_ALIGN_THRESHOLDS,
    confusion_matrix,
)


logger = logging.getLogger(__name__)

PAIRS = [
    ("Q-D", "align_QD"),
    ("Q-Sr", "align_QSr"),
    ("Q-T", "align_QT"),
]


def _matrix_for(df: pd.DataFrame, score_col: str, tau: float) -> Dict[str, object]:
    if score_col not in df.columns or "is_correct" not in df.columns:
        return {}
    s = df[score_col].fillna(-1.0)
    align_bin = (s >= tau).astype(int).tolist()
    correct_bin = df["is_correct"].astype(int).tolist()
    cm = confusion_matrix(align_bin, correct_bin)
    cm["threshold"] = float(tau)
    cm["score_col"] = score_col
    # 행=QA 정답/오답, 열=align Yes/No  (TP=정답∩alignYes, FN=정답∩alignNo, …)
    tp, fn, fp, tn = cm["TP"], cm["FN"], cm["FP"], cm["TN"]
    cm["qa_row_align_col"] = {
        "row_order": ["qa_correct", "qa_incorrect"],
        "col_order": ["align_yes", "align_no"],
        "counts": {
            "qa_correct": {"align_yes": int(tp), "align_no": int(fn)},
            "qa_incorrect": {"align_yes": int(fp), "align_no": int(tn)},
        },
    }
    return cm


def _print_qa_align_grid(cm: Dict[str, object], tau: float) -> None:
    """콘솔용: 행=QA, 열=align."""
    q = cm["qa_row_align_col"]["counts"]
    c_y, c_n = q["qa_correct"]["align_yes"], q["qa_correct"]["align_no"]
    i_y, i_n = q["qa_incorrect"]["align_yes"], q["qa_incorrect"]["align_no"]
    print(f"  τ={tau:.2f}   (rows=QA, cols=align)")
    print(f"                  {'align Yes':>10}  {'align No':>10}")
    print(f"    QA 정답       {c_y:>10d}  {c_n:>10d}")
    print(f"    QA 오답       {i_y:>10d}  {i_n:>10d}")
    ar = cm.get("align_rate", 0.0) * 100
    aa = cm.get("acc_when_aligned", 0.0)
    an = cm.get("acc_when_not_aligned", 0.0)
    phi = cm.get("phi", 0.0)
    print(f"    (align%={ar:.1f}%  acc|align={aa:.3f}  acc|¬align={an:.3f}  phi={phi:+.3f})")


def analyze_csv(csv_path: Path,
                thresholds: Sequence[float],
                save_plots: bool) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    if "is_correct" not in df.columns:
        logger.warning(f"skip (no is_correct): {csv_path}")
        return {}

    overall_acc = float(df["is_correct"].mean())
    print("\n" + "=" * 95)
    print(f"# {csv_path}  (n={len(df)}, overall acc = {overall_acc:.3f})")
    print("=" * 95)

    out: Dict[str, object] = {
        "csv": str(csv_path),
        "n": int(len(df)),
        "overall_acc": overall_acc,
        "by_pair": {},
    }
    for pair_name, col in PAIRS:
        if col not in df.columns:
            print(f"  -- pair {pair_name}: column '{col}' missing, skipping --")
            continue
        score_min = float(df[col].min(skipna=True))
        score_max = float(df[col].max(skipna=True))
        score_mean = float(df[col].mean(skipna=True))
        score_med = float(df[col].median(skipna=True))
        print(f"\n[{pair_name}]   col={col}   min={score_min:.3f}  "
              f"median={score_med:.3f}  mean={score_mean:.3f}  max={score_max:.3f}")
        print("  (각 τ마다: 행=QA 정답/오답, 열=align Yes/No ;  괄호=기존 파생지표)")
        rows = []
        for tau in thresholds:
            cm = _matrix_for(df, col, tau)
            if not cm:
                continue
            _print_qa_align_grid(cm, tau)
            rows.append(cm)
        out["by_pair"][pair_name] = {
            "score_col": col,
            "score_stats": {
                "min": score_min, "median": score_med,
                "mean": score_mean, "max": score_max,
            },
            "matrices": rows,
        }

    if save_plots:
        _plot(df, csv_path, thresholds)

    out_json = csv_path.with_name(csv_path.stem + "_align_confusion.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    logger.info(f"saved {out_json}")
    return out


def _plot(df: pd.DataFrame,
          csv_path: Path,
          thresholds: Sequence[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cor = df["is_correct"].astype(bool)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # row 1: score 분포 by correct/incorrect
    for ax, (pair_name, col) in zip(axes[0], PAIRS):
        if col not in df.columns:
            ax.axis("off")
            continue
        ax.hist(df.loc[cor, col].dropna(), bins=30, alpha=0.6,
                label=f"correct (n={int(cor.sum())})", color="C2")
        ax.hist(df.loc[~cor, col].dropna(), bins=30, alpha=0.6,
                label=f"incorrect (n={int((~cor).sum())})", color="C3")
        ax.set_title(f"{pair_name}: {col} distribution")
        ax.set_xlabel("cosine alignment")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # row 2: phi vs threshold
    for ax, (pair_name, col) in zip(axes[1], PAIRS):
        if col not in df.columns:
            ax.axis("off")
            continue
        phi_vals = []
        aa_vals = []
        an_vals = []
        for tau in thresholds:
            cm = _matrix_for(df, col, tau)
            phi_vals.append(cm.get("phi", float("nan")))
            aa_vals.append(cm.get("acc_when_aligned", float("nan")))
            an_vals.append(cm.get("acc_when_not_aligned", float("nan")))
        ax.plot(thresholds, phi_vals, "-o", label="phi corr", color="C0")
        ax.plot(thresholds, aa_vals, "-s", label="acc | aligned", color="C2")
        ax.plot(thresholds, an_vals, "-^", label="acc | NOT aligned", color="C3")
        ax.set_title(f"{pair_name}: metrics vs τ")
        ax.set_xlabel("threshold τ")
        ax.set_ylim(-0.2, 1.0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Alignment ✕ Correctness — {csv_path.stem}")
    fig.tight_layout()
    out_png = csv_path.with_name(csv_path.stem + "_align_confusion.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logger.info(f"saved {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   help="단일 CSV 파일")
    p.add_argument("--input-dir", type=Path,
                   help="여러 CSV 가 있는 dir 자동 처리")
    p.add_argument("--csv-glob", type=str, default="**/tasi_eval_*.csv")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=list(DEFAULT_ALIGN_THRESHOLDS))
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")

    targets: List[Path] = []
    if args.csv:
        targets.append(args.csv.resolve())
    if args.input_dir:
        in_dir = args.input_dir.resolve()
        for p in sorted(in_dir.glob(args.csv_glob)):
            if any(p.name.endswith(suf) for suf in
                   ("_selective.csv", "_recipes.csv", "_failure.csv",
                    "_align_confusion.csv")):
                continue
            targets.append(p)
    if not targets:
        raise SystemExit("Provide --csv or --input-dir with at least one CSV")

    results = []
    for p in targets:
        r = analyze_csv(p, args.thresholds, args.save_plots)
        if r:
            results.append(r)

    if args.input_dir and results:
        agg = args.input_dir.resolve() / "align_confusion_aggregate.json"
        with open(agg, "w", encoding="utf-8") as f:
            json.dump({"runs": results}, f, indent=2,
                      ensure_ascii=False, default=float)
        print(f"\n[align-confusion] aggregate written to {agg}")


if __name__ == "__main__":
    main()
