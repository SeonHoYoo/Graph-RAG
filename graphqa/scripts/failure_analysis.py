"""verifier 가 *왜* 실패하는지 진단하는 사후 분석 스크립트.

이미 저장된 evaluator CSV 만 가지고 (LLM 재호출 없음):

  1) score column 의 median 으로 sample 을 high/low 로 나누고,
     {correct, incorrect} × {high, low} 의 4 그룹별 통계를 비교한다.
  2) 그룹간 차이로 다음 세 가설 중 어느 것이 가장 잘 뒷받침되는지
     자동으로 점수화한다.
        (i)   paraphrase 문제   ← grounding 분포 vs is_correct
        (ii)  chain 문제        ← alignment/chain 분포 vs is_correct
        (iii) graph score 자체 무력 ← 모든 graph score 의 Δ(correct−incorrect) ≈ 0
  3) 결과 표 / 자동 진단 메시지 / PNG 두 장 (분포 + 4-그룹 bar) 출력.

사용 예:
    python -m graphqa.scripts.failure_analysis \
        --csv graphqa/outputs/full_llm_verify/musique/tasi_eval_musique.csv \
        --score-col verify_score \
        --save-plots
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# 분석 시 그룹별로 평균낼 후보 column 들. CSV 에 없으면 자동으로 skip.
NUMERIC_COLS_OF_INTEREST: Sequence[str] = (
    # answer-level signals
    "verify_score", "verify_grounding", "verify_chain", "verify_type",
    # graph signals
    "relevance_score", "consistency_score", "alignment_score",
    "search_quality_score", "retrieval_score",
    "total_sum", "total_product",
    # candidate accuracies (행마다 0/1)
    "tasi_em", "extract_em", "reason_em", "em", "f1",
    "always_yes_em", "llmpred_em",
    # graph topology
    "n_hops", "n_Q", "n_T", "n_Sr", "n_D", "n_steps", "n_slot_fillings",
    # binary
    "is_yesno",
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def _norm_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    return s


def _gold_token_count(row: pd.Series) -> int:
    g = _norm_text(row.get("ground_truth_answer", ""))
    if not g:
        return 0
    return len([t for t in g.split() if t])


def _candidate_agreement(row: pd.Series) -> float:
    """extract / reason / tasi 답 (정규화 후) 의 pairwise 일치 비율 (0~1)."""
    cands = [
        _norm_text(row.get("tasi_answer", "")),
        _norm_text(row.get("extract_answer", "")),
        _norm_text(row.get("reason_answer", "")),
    ]
    cands = [c for c in cands if c]
    if len(cands) < 2:
        return float("nan")
    n = len(cands)
    pairs = 0
    matches = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if cands[i] == cands[j]:
                matches += 1
    return float(matches / pairs) if pairs else float("nan")


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ground_truth_answer" in df.columns:
        df["gold_n_tokens"] = df.apply(_gold_token_count, axis=1)
    if all(c in df.columns for c in ("tasi_answer", "extract_answer", "reason_answer")):
        df["candidate_agreement"] = df.apply(_candidate_agreement, axis=1)
    return df


# ---------------------------------------------------------------------------
# 4-그룹 분석
# ---------------------------------------------------------------------------
def four_group_analysis(df: pd.DataFrame,
                        score_col: str = "verify_score",
                        ) -> Dict[str, pd.DataFrame]:
    """median split → 4 그룹별 평균 통계."""
    if score_col not in df.columns:
        raise KeyError(f"score column '{score_col}' not in CSV")
    if "is_correct" not in df.columns:
        raise KeyError("'is_correct' column missing")

    median = float(df[score_col].median(skipna=True))
    high = df[score_col] >= median
    cor = df["is_correct"].astype(bool)

    groups = {
        "correct ∩ high (TP)": df[cor & high],
        "correct ∩ low  (FN)": df[cor & ~high],
        "incorrect ∩ high (FP, dangerous)": df[~cor & high],
        "incorrect ∩ low  (TN)": df[~cor & ~high],
    }
    cols = [c for c in NUMERIC_COLS_OF_INTEREST if c in df.columns]
    extra = ["gold_n_tokens", "candidate_agreement"]
    cols += [c for c in extra if c in df.columns]

    rows: List[Dict[str, object]] = []
    for name, sub in groups.items():
        rec: Dict[str, object] = {"group": name, "n": int(len(sub))}
        for c in cols:
            rec[c] = float(sub[c].mean()) if len(sub) else float("nan")
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary["share"] = summary["n"] / max(1, len(df))

    return {"median": median, "summary": summary, "groups": groups}


# ---------------------------------------------------------------------------
# 가설 진단
# ---------------------------------------------------------------------------
def diagnose(df: pd.DataFrame,
             score_col: str = "verify_score") -> Dict[str, object]:
    """세 가설 (i/ii/iii) 의 strength 를 점수화."""
    out: Dict[str, object] = {"score_col": score_col}
    cor = df["is_correct"].astype(bool)
    incor = ~cor

    def _delta(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        a = float(df.loc[cor, col].mean()) if cor.any() else float("nan")
        b = float(df.loc[incor, col].mean()) if incor.any() else float("nan")
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        return float(a - b)

    grounding_delta = _delta("verify_grounding")
    chain_delta = _delta("verify_chain")
    type_delta = _delta("verify_type")
    relevance_delta = _delta("relevance_score")
    alignment_delta = _delta("alignment_score")
    consistency_delta = _delta("consistency_score")

    # (i) paraphrase 문제: gold answer 의 token 개수 ↑ + grounding signal 약함 + grounding Δ ≈ 0
    multi_token_share = float((df.get("gold_n_tokens", pd.Series([0])) > 1).mean()) \
        if "gold_n_tokens" in df.columns else float("nan")
    paraphrase_score = 0.0
    if multi_token_share == multi_token_share:
        paraphrase_score += min(multi_token_share, 1.0)
    if grounding_delta == grounding_delta:
        # grounding 이 답을 거의 못 가르면 (Δ ≈ 0 이거나 음수) 가설 (i) 가능성 ↑
        paraphrase_score += max(0.0, 0.05 - abs(grounding_delta)) * 10.0
    out["hypothesis_paraphrase_score"] = float(paraphrase_score)

    # (ii) chain 문제: alignment / chain Δ 가 양수 (정답인 sample 에서 더 높음) 면 chain 신호 살아있음
    #      반대로 정답/오답이 거의 같으면 chain 노이즈
    chain_score = 0.0
    for d in [chain_delta, alignment_delta, consistency_delta]:
        if d == d:
            # |Δ| 가 작을수록 chain noise ↑
            chain_score += max(0.0, 0.05 - abs(d)) * 6.66  # 0~~~0.33
    out["hypothesis_chain_score"] = float(chain_score)

    # (iii) graph 본체 무력: 모든 graph score Δ 의 평균 |Δ|
    graph_deltas = [d for d in [relevance_delta, alignment_delta, consistency_delta]
                    if d == d]
    out["graph_delta_mean_abs"] = float(np.mean([abs(x) for x in graph_deltas])) \
        if graph_deltas else float("nan")
    inert_score = 0.0
    if graph_deltas:
        inert_score = float(max(0.0, 0.05 - np.mean([abs(x) for x in graph_deltas])) * 20.0)
    out["hypothesis_graph_inert_score"] = inert_score

    out.update({
        "delta_grounding": grounding_delta,
        "delta_chain": chain_delta,
        "delta_type": type_delta,
        "delta_relevance": relevance_delta,
        "delta_alignment": alignment_delta,
        "delta_consistency": consistency_delta,
        "multi_token_gold_share": multi_token_share,
    })

    # 가장 강한 가설 선언
    cands = {
        "(i) paraphrase mismatch": out["hypothesis_paraphrase_score"],
        "(ii) chain noise": out["hypothesis_chain_score"],
        "(iii) graph inertia": out["hypothesis_graph_inert_score"],
    }
    out["dominant_hypothesis"] = max(cands, key=lambda k: cands[k])
    out["hypothesis_strengths"] = cands
    return out


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def _save_plots(df: pd.DataFrame,
                score_col: str,
                out_dir: Path,
                stem: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    cols = [c for c in [
        "verify_score", "verify_grounding", "verify_chain", "verify_type",
        "relevance_score", "consistency_score", "alignment_score",
        "total_sum",
    ] if c in df.columns]
    if not cols:
        return
    cor = df["is_correct"].astype(bool)
    n_cols = 4
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).ravel()
    for ax, col in zip(axes, cols):
        ax.hist(df.loc[cor, col].dropna(), bins=25, alpha=0.6,
                label=f"correct (n={int(cor.sum())})", color="C2")
        ax.hist(df.loc[~cor, col].dropna(), bins=25, alpha=0.6,
                label=f"incorrect (n={int((~cor).sum())})", color="C3")
        ax.set_title(col)
        ax.legend(fontsize=8)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"score distribution by correctness — {stem}")
    fig.tight_layout()
    p1 = out_dir / f"{stem}_failure_dist.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    logger.info(f"saved {p1}")

    # 4-group bar plot
    median = float(df[score_col].median(skipna=True))
    high = df[score_col] >= median
    correct = cor
    groups = {
        "C∩High": df[correct & high],
        "C∩Low":  df[correct & ~high],
        "I∩High": df[~correct & high],
        "I∩Low":  df[~correct & ~high],
    }
    plot_metrics = [c for c in [
        "verify_grounding", "verify_chain", "verify_type",
        "relevance_score", "consistency_score", "alignment_score",
    ] if c in df.columns]
    if plot_metrics:
        fig2, ax2 = plt.subplots(1, 1, figsize=(11, 5))
        x = np.arange(len(plot_metrics))
        width = 0.2
        for i, (gname, sub) in enumerate(groups.items()):
            vals = [float(sub[m].mean()) if len(sub) else 0.0 for m in plot_metrics]
            ax2.bar(x + i * width - 1.5 * width, vals, width=width,
                    label=f"{gname} (n={len(sub)})")
        ax2.set_xticks(x)
        ax2.set_xticklabels(plot_metrics, rotation=20, ha="right")
        ax2.set_ylabel("mean")
        ax2.set_title(f"4-group means — {stem}  (split @{score_col} median={median:.3f})")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, axis="y")
        fig2.tight_layout()
        p2 = out_dir / f"{stem}_failure_4group.png"
        fig2.savefig(p2, dpi=120)
        plt.close(fig2)
        logger.info(f"saved {p2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   help="단일 CSV. --input-dir 와 둘 중 하나는 필요.")
    p.add_argument("--input-dir", type=Path,
                   help="여러 CSV 가 있는 dir 를 자동 처리.")
    p.add_argument("--csv-glob", type=str, default="**/tasi_eval_*.csv")
    p.add_argument("--score-col", type=str, default="verify_score")
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--output-suffix", type=str, default="_failure")
    return p.parse_args()


def process_one(csv_path: Path,
                score_col: str,
                save_plots: bool,
                suffix: str) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    if "is_correct" not in df.columns:
        logger.warning(f"skip (no is_correct): {csv_path}")
        return {}
    df = enrich_df(df)
    if score_col not in df.columns:
        logger.warning(f"skip (no {score_col}): {csv_path}")
        return {}

    print("\n" + "=" * 95)
    print(f"# {csv_path}")
    print(f"  n={len(df)}  overall_acc={df['is_correct'].mean():.3f}  score_col={score_col}")
    print("=" * 95)

    res = four_group_analysis(df, score_col=score_col)
    summary: pd.DataFrame = res["summary"]
    median = res["median"]
    print(f"\n[4-group split @median({score_col})={median:.3f}]")
    show_cols = ["group", "n", "share"]
    for c in ["verify_grounding", "verify_chain", "verify_type",
              "relevance_score", "consistency_score", "alignment_score",
              "total_sum",
              "tasi_em", "extract_em", "reason_em",
              "n_hops", "n_D", "gold_n_tokens", "candidate_agreement",
              "is_yesno"]:
        if c in summary.columns:
            show_cols.append(c)
    print(summary[show_cols].to_string(
        index=False,
        float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)))

    diag = diagnose(df, score_col=score_col)
    print("\n[Diagnosis]")
    for k in ["delta_grounding", "delta_chain", "delta_type",
              "delta_relevance", "delta_alignment", "delta_consistency",
              "multi_token_gold_share", "graph_delta_mean_abs"]:
        v = diag.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            print(f"  {k:>26s}  = {v:+.3f}" if v == v else f"  {k:>26s}  = nan")
        else:
            print(f"  {k:>26s}  = {v}")
    print("\n  hypothesis strengths (higher = more likely):")
    for k, v in diag["hypothesis_strengths"].items():
        marker = "  <-- dominant" if k == diag["dominant_hypothesis"] else ""
        print(f"    {k:>32s}  : {v:.3f}{marker}")

    out_dir = csv_path.parent
    stem = csv_path.stem + suffix
    out_json = out_dir / f"{stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "csv": str(csv_path),
            "n_samples": int(len(df)),
            "overall_acc": float(df["is_correct"].mean()),
            "score_col": score_col,
            "median": median,
            "groups": summary.to_dict(orient="records"),
            "diagnosis": diag,
        }, f, indent=2, ensure_ascii=False, default=float)
    logger.info(f"saved {out_json}")

    if save_plots:
        _save_plots(df, score_col, out_dir, csv_path.stem + suffix)

    return {"csv": str(csv_path), "diagnosis": diag, "groups": summary.to_dict(orient="records")}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    if not args.csv and not args.input_dir:
        raise SystemExit("Provide --csv or --input-dir")

    targets: List[Path] = []
    if args.csv:
        targets.append(args.csv.resolve())
    if args.input_dir:
        in_dir = args.input_dir.resolve()
        for p in sorted(in_dir.glob(args.csv_glob)):
            if any(p.name.endswith(suf) for suf in
                   ("_selective.csv", "_recipes.csv", "_failure.csv")):
                continue
            targets.append(p)
    if not targets:
        raise SystemExit("No CSV targets found")

    out_aggregate = []
    for p in targets:
        r = process_one(p, score_col=args.score_col,
                        save_plots=args.save_plots,
                        suffix=args.output_suffix)
        if r:
            out_aggregate.append(r)

    if args.input_dir and out_aggregate:
        agg = args.input_dir.resolve() / f"failure_aggregate{args.output_suffix}.json"
        with open(agg, "w", encoding="utf-8") as f:
            json.dump({"runs": out_aggregate}, f, indent=2,
                      ensure_ascii=False, default=float)
        print(f"\n[failure] aggregate written to {agg}")


if __name__ == "__main__":
    main()
