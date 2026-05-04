#!/usr/bin/env python3
"""Aggregate per-hop EM/F1 stats for answer result files.

Usage:
  python scripts/analysis/perhop.py \
    --root /home/hyeseojeon/data/Graph-RAG/results/answer/0303 \
    --out /home/hyeseojeon/data/Graph-RAG/results/answer/0303/perhop_summary.txt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-file per-hop EM/F1 aggregation")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/hyeseojeon/data/Graph-RAG/results/baseline/0308"),
        help="Root directory to recursively scan (default: results/answer/0308)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/hyeseojeon/data/Graph-RAG/results/baseline/0308/perhop_summary.txt"),
        help="Output text file path",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Filename pattern to scan recursively (default: *.json)",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load a file as JSON list/object, or JSONL fallback."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
        return []
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
        return records


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_per_hop(records: Iterable[Dict[str, Any]]) -> Dict[int, Tuple[int, float, float]]:
    """Return {num_hops: (count, mean_em, mean_f1)}."""
    bucket: Dict[int, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "em": 0.0, "f1": 0.0})

    for rec in records:
        num_hops_raw = rec.get("num_hops")
        em = to_float(rec.get("em_score"))
        f1 = to_float(rec.get("f1_score"))

        if em is None or f1 is None:
            continue

        try:
            num_hops = int(num_hops_raw)
        except (TypeError, ValueError):
            continue

        bucket[num_hops]["count"] += 1
        bucket[num_hops]["em"] += em
        bucket[num_hops]["f1"] += f1

    result: Dict[int, Tuple[int, float, float]] = {}
    for hop, stats in sorted(bucket.items()):
        count = int(stats["count"])
        if count == 0:
            continue
        result[hop] = (count, stats["em"] / count, stats["f1"] / count)
    return result


def build_report(root: Path, files: List[Path]) -> str:
    lines: List[str] = []
    lines.append(f"Root: {root}")
    lines.append(f"Scanned files: {len(files)}")
    lines.append("")

    processed = 0
    skipped = 0

    for path in files:
        records = load_records(path)
        if not records:
            skipped += 1
            continue

        per_hop = aggregate_per_hop(records)
        if not per_hop:
            skipped += 1
            continue

        processed += 1
        rel = path.relative_to(root)
        lines.append(f"File: {rel}")
        lines.append("num_hops\tsamples\tem\tf1")

        for hop, (count, em_avg, f1_avg) in per_hop.items():
            lines.append(f"{hop}\t{count}\t{em_avg:.4f}\t{f1_avg:.4f}")

        lines.append("")

    lines.append(f"Processed files: {processed}")
    lines.append(f"Skipped files (no usable records): {skipped}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root: Path = args.root
    out_path: Path = args.out

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Invalid --root directory: {root}")

    files = sorted(p for p in root.rglob(args.pattern) if p.is_file())
    report = build_report(root, files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Scanned: {len(files)} files")


if __name__ == "__main__":
    main()
