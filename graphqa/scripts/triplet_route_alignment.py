"""Summarize triplet-fill route/order alignment diagnostics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


logger = logging.getLogger(__name__)


def _route_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("triplet_route_")]


def analyze_csv(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    cols = _route_columns(df)
    if not cols:
        logger.warning("skip (no triplet_route columns): %s", csv_path)
        return {}

    scope = ""
    if "triplet_evidence_scope" in df.columns and len(df):
        vals = df["triplet_evidence_scope"].dropna().astype(str).unique().tolist()
        scope = vals[0] if len(vals) == 1 else "|".join(vals)

    out: Dict[str, Any] = {
        "csv": str(csv_path),
        "scope": scope,
        "n": int(len(df)),
    }
    print("\n" + "=" * 88)
    print(f"# route alignment: {csv_path}  (n={len(df)}, scope={scope or 'n/a'})")
    print("=" * 88)

    for kind in ("doc", "think"):
        prefix = f"triplet_route_{kind}_"
        summary: Dict[str, Any] = {}
        for name in (
            "strict_available_steps",
            "prefix_available_steps",
            "future_only_steps",
            "unavailable_steps",
            "anywhere_available_steps",
        ):
            col = prefix + name
            if col in df.columns:
                summary[name] = int(df[col].fillna(0).astype(float).sum())
        mismatch_col = prefix + "order_mismatch_rate"
        if mismatch_col in df.columns:
            summary["mean_order_mismatch_rate"] = float(
                df[mismatch_col].fillna(0).astype(float).mean()
            )
        longest_col = prefix + "longest_exact_prefix"
        if longest_col in df.columns:
            vals = df[longest_col].fillna(0).astype(int)
            summary["mean_longest_exact_prefix"] = float(vals.mean())
            summary["longest_exact_prefix_counts"] = {
                str(int(k)): int(v) for k, v in vals.value_counts().sort_index().items()
            }
        out[kind] = summary
        print(
            f"[{kind}] strict={summary.get('strict_available_steps', 0)} "
            f"prefix={summary.get('prefix_available_steps', 0)} "
            f"future_only={summary.get('future_only_steps', 0)} "
            f"unavailable={summary.get('unavailable_steps', 0)} "
            f"mean_mismatch={summary.get('mean_order_mismatch_rate', 0.0):.3f} "
            f"mean_exact_prefix={summary.get('mean_longest_exact_prefix', 0.0):.2f}"
        )

    both_col = "triplet_route_both_longest_exact_prefix"
    if both_col in df.columns:
        vals = df[both_col].fillna(0).astype(int)
        out["both_longest_exact_prefix_counts"] = {
            str(int(k)): int(v) for k, v in vals.value_counts().sort_index().items()
        }
    out_json = csv_path.with_name(csv_path.stem + "_triplet_route_alignment_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, help="single CSV file")
    p.add_argument("--input-dir", type=Path, help="directory containing CSVs")
    p.add_argument("--csv-glob", type=str, default="**/tasi_eval_*.csv")
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
        r = analyze_csv(p)
        if r:
            results.append(r)

    if args.input_dir and results:
        agg = args.input_dir.resolve() / "triplet_route_alignment_aggregate.json"
        with open(agg, "w", encoding="utf-8") as f:
            json.dump({"runs": results}, f, indent=2, ensure_ascii=False)
        print(f"\n[triplet-route] aggregate written to {agg}")


if __name__ == "__main__":
    main()
