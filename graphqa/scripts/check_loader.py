"""데이터 로더 sanity check."""
from __future__ import annotations

import argparse
import logging
import sys

# 상위 디렉터리에서 실행되도록 sys.path 조정
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from graphqa.data import load_dataset, DATASETS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2wikimultihopqa", choices=list(DATASETS))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--show-triples", type=int, default=3, help="\uadf8\ub798\ud504\ub2f9 \ucd9c\ub825 \uc81c\ud55c")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    samples = load_dataset(args.dataset, limit=args.limit)
    print(f"\n=== Loaded {len(samples)} samples from {args.dataset} ===")

    for s in samples:
        print("\n" + "=" * 80)
        print(s.summary())
        print(f"  Q  question  : {s.question}")
        print(f"  Q  answer/GT : {s.answer}")
        print(f"  Q  pred(LLM) : {s.predicted_answer}")
        print(f"  Q  num_hops  : {s.num_hops}")
        print(f"  Q  search_qs : {s.search_queries[:3]}{'...' if len(s.search_queries) > 3 else ''}")

        for label, triples in [
            ("Q (question)", s.Q),
            ("Q_def       ", s.Q_def),
            ("T (think)   ", s.T),
            ("Sr(search)  ", s.Sr),
            ("D (doc)     ", s.D),
        ]:
            print(f"  -- {label} ({len(triples)} triples) --")
            for t in triples[: args.show_triples]:
                print(f"     {t.head} | {t.relation} | {t.tail}"
                      + (f" [PREP] {t.context}" if t.context else ""))

        if s.T_steps:
            print(f"  -- T_steps ({len(s.T_steps)}) --")
            for st in s.T_steps[:2]:
                print(f"     step{st.step_index} ({len(st.triples)} triples): {st.step_text[:70]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
