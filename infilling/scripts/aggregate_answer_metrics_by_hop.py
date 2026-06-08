#!/usr/bin/env python3
"""
기존 answer 결과 JSON을 기반으로 EM/F1을 집계합니다.

집계 단위:
- dataset
- use_full_doc (triplet / fulldoc / doconly)
- triplet_selection (all / top1 / top3 / top5 / top10)
- ent_scope (파일명 마지막 토큰: all/false 등)
- num_hops (개별 hop + total)

입력 예시:
  /data3/seonhoyoo/graphcheck-qa/infilling/output/answer/Qwen2.5-7B-Instruct

출력:
- CSV: output/answer_em_f1_by_hop_<model>.csv
- MD : output/answer_em_f1_by_hop_<model>.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ANSWER_ROOT = Path(
    "/data3/seonhoyoo/graphcheck-qa/infilling/output/answer/Qwen2.5-7B-Instruct"
)
DEFAULT_OUTPUT_DIR = Path("/data3/seonhoyoo/graphcheck-qa/infilling/output")
DATASET_ORDER = ("2wikimultihopqa", "hotpotqa", "musique")
USE_FULL_DOC_ORDER = ("triplet", "fulldoc", "doconly")
TRIPLET_ORDER = ("all", "top1", "top3", "top5", "top10")
HOP_UNKNOWN = "unknown"
HOP_TOTAL = "total"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--answer_root", type=str, default=str(DEFAULT_ANSWER_ROOT))
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_md", type=str, default=None)
    return p.parse_args()


def parse_answer_filename(path: Path) -> tuple[str, str, str] | None:
    """
    answer_*.json 파일명에서 (use_full_doc, triplet_selection, ent_scope)를 파싱합니다.
    예:
      answer_triplets_triplet_only_triplet_only_gold_fulldoc_top3_all.json
      -> ("fulldoc", "top3", "all")
    """
    stem = path.stem
    if not stem.startswith("answer_"):
        return None

    m = re.search(r"(?:gold|all)_(doconly|fulldoc|triplet)_(all|top\d+)_(\w+)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Baseline output:
    # answer_baseline_qonly_openbook_top{k}_{ent_scope}.json
    m2 = re.search(r"baseline_qonly_openbook_top(\d+)_(\w+)$", stem)
    if m2:
        k = m2.group(1)
        ent_scope = m2.group(2)
        return "baseline", f"qonly_top{k}", ent_scope

    return None


def detect_dataset(path: Path, answer_root: Path) -> str | None:
    rel = path.relative_to(answer_root)
    if len(rel.parts) < 2:
        return None
    ds = rel.parts[0]
    if ds in DATASET_ORDER:
        return ds
    return None


def to_float_or_none(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def hop_sort_key(h: int | str) -> tuple[int, int | str]:
    if h == HOP_TOTAL:
        return (0, 0)
    if isinstance(h, int):
        return (1, h)
    return (2, str(h))


def build_sort_key(row: tuple) -> tuple:
    dataset, use_full_doc, triplet_selection, ent_scope, hop, _, _, _ = row
    ds_idx = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 999
    if use_full_doc in USE_FULL_DOC_ORDER:
        ufd_idx = USE_FULL_DOC_ORDER.index(use_full_doc)
    elif use_full_doc == "baseline":
        ufd_idx = 998
    else:
        ufd_idx = 999

    if triplet_selection in TRIPLET_ORDER:
        ts_idx = TRIPLET_ORDER.index(triplet_selection)
    elif triplet_selection.startswith("qonly_top"):
        try:
            ts_idx = 100 + int(triplet_selection.replace("qonly_top", ""))
        except ValueError:
            ts_idx = 999
    else:
        ts_idx = 999
    return (ds_idx, ufd_idx, ts_idx, ent_scope, hop_sort_key(hop))


def main() -> None:
    args = parse_args()
    answer_root = Path(args.answer_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not answer_root.is_dir():
        raise FileNotFoundError(f"answer_root not found: {answer_root}")

    model_name = answer_root.name
    out_csv = Path(args.out_csv) if args.out_csv else output_dir / f"answer_em_f1_by_hop_{model_name}.csv"
    out_md = Path(args.out_md) if args.out_md else output_dir / f"answer_em_f1_by_hop_{model_name}.md"

    # key: (dataset, use_full_doc, triplet_selection, ent_scope, hop)
    # val: list[(em, f1)]
    grouped: dict[tuple[str, str, str, str, int | str], list[tuple[float, float]]] = defaultdict(list)

    files = sorted(answer_root.glob("**/answer_*.json"))
    if not files:
        raise RuntimeError(f"No answer_*.json found under {answer_root}")

    used_files = 0
    skipped_files = 0
    for path in files:
        dataset = detect_dataset(path, answer_root)
        parsed = parse_answer_filename(path)
        if dataset is None or parsed is None:
            skipped_files += 1
            continue

        use_full_doc, triplet_selection, ent_scope = parsed
        try:
            with open(path, "r", encoding="utf-8") as f:
                samples = json.load(f)
        except Exception:
            skipped_files += 1
            continue

        if not isinstance(samples, list):
            skipped_files += 1
            continue

        used_files += 1
        for s in samples:
            em = to_float_or_none(s.get("em_score"))
            f1 = to_float_or_none(s.get("f1_score"))
            if em is None or f1 is None:
                continue

            hop_raw = s.get("num_hops")
            hop: int | str = HOP_UNKNOWN
            if isinstance(hop_raw, int):
                hop = hop_raw
            elif isinstance(hop_raw, str) and hop_raw.isdigit():
                hop = int(hop_raw)

            key_total = (dataset, use_full_doc, triplet_selection, ent_scope, HOP_TOTAL)
            key_hop = (dataset, use_full_doc, triplet_selection, ent_scope, hop)
            grouped[key_total].append((em, f1))
            grouped[key_hop].append((em, f1))

    rows: list[tuple[str, str, str, str, int | str, int, float, float]] = []
    for key, scores in grouped.items():
        if not scores:
            continue
        n = len(scores)
        avg_em = sum(x[0] for x in scores) / n
        avg_f1 = sum(x[1] for x in scores) / n
        rows.append((*key, n, avg_em, avg_f1))

    rows.sort(key=build_sort_key)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("dataset,use_full_doc,triplet_selection,ent_scope,num_hops,n,em,f1\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]:.4f},{r[7]:.4f}\n")

    md_lines = [
        f"# Answer EM/F1 by Hop ({model_name})",
        "",
        f"- source: `{answer_root}`",
        f"- files used: {used_files} (skipped: {skipped_files})",
        "",
        "| dataset | use_full_doc | triplet_selection | ent_scope | num_hops | n | EM | F1 |",
        "|---------|--------------|-------------------|-----------|----------|---|-----|-----|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]:.4f} | {r[7]:.4f} |"
        )
    md_lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Answer root : {answer_root}")
    print(f"Files used  : {used_files} (skipped: {skipped_files})")
    print(f"CSV saved   : {out_csv}")
    print(f"MD saved    : {out_md}")


if __name__ == "__main__":
    main()
