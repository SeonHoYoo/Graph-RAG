#!/usr/bin/env python
"""Collect baseline metrics for VeriGraph feedback experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    val = row.get(key, "")
    if val in ("", "nan", "None", None):
        return None
    try:
        return float(val)
    except Exception:
        return None


def _boolish(row: Dict[str, str], key: str) -> Optional[float]:
    val = str(row.get(key, "")).strip().lower()
    if val in ("true", "1", "yes"):
        return 1.0
    if val in ("false", "0", "no"):
        return 0.0
    return _float(row, key)


def _avg(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return round(mean(xs), 6) if xs else None


def _summarize_rows(rows: List[Dict[str, str]], spec: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}
    for out_key, col in spec.items():
        if out_key.endswith("_rate") or out_key.endswith("_acc") or out_key.startswith("is_"):
            out[out_key] = _avg(_boolish(r, col) for r in rows)
        else:
            out[out_key] = _avg(_float(r, col) for r in rows)
    return out


def summarize_dataset(run_dir: Path, dataset: str) -> Dict[str, Any]:
    fallback_csv = (
        run_dir / "fallback_verigraph" / dataset / f"online_corrector_{dataset}.csv"
    )
    analyze_csv = (
        run_dir / "analyze_verigraph" / dataset / f"online_eval_{dataset}.csv"
    )

    fallback_rows = _read_csv(fallback_csv)
    analyze_rows = _read_csv(analyze_csv)

    vanilla = _summarize_rows(
        fallback_rows,
        {
            "em": "vanilla_em",
            "f1": "vanilla_f1",
            "turns": "vanilla_num_turns",
        },
    )
    analyze = _summarize_rows(
        analyze_rows,
        {
            "em": "em",
            "f1": "f1",
            "acc": "is_correct",
            "searchr1_em": "searchr1_em",
            "searchr1_f1": "searchr1_f1",
            "final_q_hint_em": "final_q_hint_em",
            "final_q_hint_f1": "final_q_hint_f1",
            "abstain_rate": "abstained",
            "n_slot_steps": "n_slot_steps",
            "triplet_min_doc_score": "triplet_min_doc_score",
            "triplet_min_think_score": "triplet_min_think_score",
        },
    )
    fallback = _summarize_rows(
        fallback_rows,
        {
            "em": "em",
            "f1": "f1",
            "turns": "final_num_turns",
            "trigger_rate": "trigger_verigraph",
            "control_em": "control_em",
            "control_f1": "control_f1",
            "system_d_em": "system_d_em",
            "system_d_f1": "system_d_f1",
        },
    )

    return {
        "dataset": dataset,
        "files": {
            "vanilla_searchr1_from_fallback_csv": str(fallback_csv),
            "analyze_verigraph_csv": str(analyze_csv),
            "fallback_verigraph_csv": str(fallback_csv),
        },
        "vanilla_searchr1": vanilla,
        "analyze_verigraph": analyze,
        "fallback_verigraph": fallback,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    per_dataset = [summarize_dataset(run_dir, ds) for ds in args.datasets]

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "datasets": per_dataset,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
