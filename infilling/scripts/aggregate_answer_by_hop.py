#!/usr/bin/env python3
"""
Answer JSON 파일에서 dataset × use_full_doc × triplet_selection × num_hops별 EM/F1 추출
출력: CSV, Markdown (hop 차원 포함)
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ANSWER_DIR = Path(__file__).resolve().parents[1] / "output" / "answer"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

# answer_*.json 파일명 → (dataset, use_full_doc, triplet_selection) 파싱
def parse_filename(path: Path) -> tuple[str, str, str] | None:
    name = path.stem  # answer_triplets_triplet_only_triplet_only_gold_X_Y_all
    if not name.startswith("answer_"):
        return None
    # answer_triplets_triplet_only_triplet_only_gold_{doconly|fulldoc|triplet}_{all|top1|top3|top5|top10}_all
    m = re.search(r"gold_(doconly|fulldoc|triplet)_(all|top\d+)_all", name)
    if not m:
        return None
    use_full_doc, triplet_selection = m.group(1), m.group(2)
    dataset = path.parent.name
    return dataset, use_full_doc, triplet_selection


def load_answer_file(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_by_hop(samples: list[dict]) -> dict:
    """num_hops별 EM/F1 집계. num_hops 없는 샘플은 'total'로."""
    by_hop: dict[int | str, list[tuple[float, float]]] = defaultdict(list)
    for s in samples:
        em = s.get("em_score")
        f1 = s.get("f1_score")
        if em is None and f1 is None:
            continue
        em = float(em) if em is not None else 0.0
        f1 = float(f1) if f1 is not None else 0.0
        hop = s.get("num_hops")
        by_hop["total"].append((em, f1))
        if hop is not None:
            by_hop[int(hop)].append((em, f1))
    return by_hop


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # (dataset, use_full_doc, triplet_selection) → hop → (n, em, f1)
    results: dict[tuple, dict] = defaultdict(dict)

    for dataset in ("2wikimultihopqa", "hotpotqa", "musique"):
        ddir = ANSWER_DIR / dataset
        if not ddir.exists():
            continue
        for path in sorted(ddir.glob("answer_*.json")):
            parsed = parse_filename(path)
            if not parsed:
                continue
            ds, ufd, ts = parsed
            samples = load_answer_file(path)
            by_hop = aggregate_by_hop(samples)

            for hop_key, scores in by_hop.items():
                n = len(scores)
                if n == 0:
                    continue
                avg_em = sum(s[0] for s in scores) / n
                avg_f1 = sum(s[1] for s in scores) / n
                results[(ds, ufd, ts)][hop_key] = (n, avg_em, avg_f1)

    # CSV (hop 포함)
    rows = []
    for (ds, ufd, ts) in sorted(results.keys()):
        hop_data = results[(ds, ufd, ts)]
        for hop in sorted(hop_data.keys(), key=lambda x: (0 if x == "total" else 1, x if isinstance(x, int) else 0)):
            n, em, f1 = hop_data[hop]
            rows.append((ds, ufd, ts, hop, n, em, f1))

    csv_path = OUTPUT_DIR / "answer_em_f1_by_hop.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dataset,use_full_doc,triplet_selection,num_hops,n,em,f1\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]:.4f}\n")
    print(f"CSV: {csv_path}")

    # Markdown
    md_path = OUTPUT_DIR / "answer_em_f1_by_hop.md"
    lines = [
        "# Answer EM/F1 by num_hops",
        "",
        "**출처**: `output/answer/**/answer_*.json` (각 샘플 em_score, f1_score, num_hops)",
        "",
        "---",
        "",
        "## 표 (dataset × use_full_doc × triplet_selection × num_hops)",
        "",
        "| dataset | use_full_doc | triplet_selection | num_hops | n | EM | F1 |",
        "|---------|--------------|-------------------|----------|---|-------|-------|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.4f} | {r[6]:.4f} |")
    lines.extend(["", "---", ""])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
