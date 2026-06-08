"""Triplet-fill doc/think OK group analysis.

Reads CSVs produced by qa-mode=triplet_fill and summarizes the relationship
between QA correctness and the four doc/think validation states:

  0/0, 1/0, 0/1, 1/1

The first bit is query-vs-document field validation, the second bit is
query-vs-think field validation. LLM is not called here.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


logger = logging.getLogger(__name__)

PAIR_ORDER = ["0/0", "1/0", "0/1", "1/1"]


def _derive_pair(df: pd.DataFrame) -> pd.Series:
    if "triplet_ok_pair" in df.columns:
        s = df["triplet_ok_pair"].fillna("").astype(str)
        if s.str.len().gt(0).any():
            return s
    if {"triplet_doc_ok", "triplet_think_ok"}.issubset(df.columns):
        d = df["triplet_doc_ok"].fillna(0).astype(float).astype(int).clip(0, 1)
        t = df["triplet_think_ok"].fillna(0).astype(float).astype(int).clip(0, 1)
        return d.astype(str) + "/" + t.astype(str)
    return pd.Series([""] * len(df), index=df.index)


def _summarize_pairs(df: pd.DataFrame, *, label: str) -> List[Dict[str, object]]:
    print(f"\n[{label}]")
    print(f"{'pair':<6s} {'n':>6s} {'qa_correct':>10s} {'qa_wrong':>9s} "
          f"{'acc':>8s} {'f1':>8s} {'abstain':>9s}")
    print("-" * 72)

    rows: List[Dict[str, object]] = []
    for p in PAIR_ORDER:
        sub = df.loc[df["_triplet_pair"] == p]
        n = int(len(sub))
        correct = int(sub["is_correct"].astype(bool).sum()) if n else 0
        wrong = n - correct
        acc = float(sub["is_correct"].mean()) if n else float("nan")
        f1 = float(sub["f1"].mean()) if n and "f1" in sub.columns else float("nan")
        abst = float(sub["abstained"].mean()) if n and "abstained" in sub.columns else float("nan")
        row = {
            "pair": p,
            "n": n,
            "qa_correct": correct,
            "qa_wrong": wrong,
            "accuracy": acc,
            "f1_mean": f1,
            "abstain_rate": abst,
        }
        rows.append(row)
        print(f"{p:<6s} {n:>6d} {correct:>10d} {wrong:>9d} "
              f"{acc:>8.3f} {f1:>8.3f} {abst:>9.3f}")
    return rows


def _expand_step_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if "triplet_step_ok_pairs" not in df.columns:
        return pd.DataFrame()
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        raw = str(row.get("triplet_step_ok_pairs", "") or "")
        for p in raw.split("|"):
            if p not in PAIR_ORDER:
                continue
            records.append({
                "_triplet_pair": p,
                "is_correct": bool(row.get("is_correct", False)),
                "f1": float(row.get("f1", 0.0)),
                "abstained": bool(row.get("abstained", False)),
            })
    return pd.DataFrame(records)


def analyze_csv(csv_path: Path, save_plots: bool = False) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    if "is_correct" not in df.columns:
        logger.warning("skip (no is_correct): %s", csv_path)
        return {}
    pair = _derive_pair(df)
    if not pair.str.len().gt(0).any():
        logger.warning("skip (no triplet ok columns): %s", csv_path)
        return {}
    df = df.copy()
    df["_triplet_pair"] = pair

    print("\n" + "=" * 88)
    print(f"# {csv_path}  (n={len(df)}, overall acc={float(df['is_correct'].mean()):.3f})")
    print("=" * 88)
    rows = _summarize_pairs(df, label="sample-level all-step AND")
    step_rows: List[Dict[str, object]] = []
    step_df = _expand_step_pairs(df)
    if not step_df.empty:
        step_rows = _summarize_pairs(step_df, label="query-triple step-level")

    out = {
        "csv": str(csv_path),
        "n": int(len(df)),
        "overall_accuracy": float(df["is_correct"].mean()),
        "groups": rows,
        "step_groups": step_rows,
    }

    out_json = csv_path.with_name(csv_path.stem + "_triplet_ok_summary.json")
    out_csv = csv_path.with_name(csv_path.stem + "_triplet_ok_summary.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("saved %s and %s", out_json, out_csv)

    if save_plots:
        _plot(rows, csv_path)

    return out


def _plot(rows: List[Dict[str, object]], csv_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [str(r["pair"]) for r in rows]
    ns = [int(r["n"]) for r in rows]
    accs = [float(r["accuracy"]) if r["n"] else 0.0 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, ns, color="C0", alpha=0.8)
    axes[0].set_title("count by doc/think OK pair")
    axes[0].set_xlabel("doc_ok/think_ok")
    axes[0].set_ylabel("n")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, accs, color="C2", alpha=0.8)
    axes[1].set_title("QA accuracy by pair")
    axes[1].set_xlabel("doc_ok/think_ok")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(csv_path.stem)
    fig.tight_layout()
    out_png = csv_path.with_name(csv_path.stem + "_triplet_ok_summary.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logger.info("saved %s", out_png)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, help="single CSV file")
    p.add_argument("--input-dir", type=Path, help="directory containing CSVs")
    p.add_argument("--csv-glob", type=str, default="**/tasi_eval_*.csv")
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    targets: List[Path] = []
    if args.csv:
        targets.append(args.csv.resolve())
    if args.input_dir:
        in_dir = args.input_dir.resolve()
        for p in sorted(in_dir.glob(args.csv_glob)):
            if any(p.name.endswith(suf) for suf in (
                "_selective.csv",
                "_recipes.csv",
                "_failure.csv",
                "_align_confusion.csv",
                "_triplet_death_steps.csv",
                "_triplet_ok_summary.csv",
                "_triplet_route_alignment.csv",
                "_triplet_route_alignment_summary.csv",
            )):
                continue
            targets.append(p)
    if not targets:
        raise SystemExit("Provide --csv or --input-dir with at least one CSV")

    results = []
    for p in targets:
        r = analyze_csv(p, save_plots=args.save_plots)
        if r:
            results.append(r)

    if args.input_dir and results:
        agg = args.input_dir.resolve() / "triplet_ok_aggregate.json"
        with open(agg, "w", encoding="utf-8") as f:
            json.dump({"runs": results}, f, indent=2,
                      ensure_ascii=False, default=float)
        print(f"\n[triplet-ok] aggregate written to {agg}")


if __name__ == "__main__":
    main()
