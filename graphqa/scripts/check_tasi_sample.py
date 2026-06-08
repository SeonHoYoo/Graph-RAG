"""실제 데이터셋 샘플 1개로 TASI 동작 확인."""
from __future__ import annotations

import argparse
import logging
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from graphqa.data import load_dataset, DATASETS
from graphqa.tasi import (
    get_default_encoder,
    tasi,
    free_matching,
    pairwise_alignment_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2wikimultihopqa", choices=list(DATASETS))
    parser.add_argument("--n", type=int, default=2, help="\uc0d8\ud50c \uac1c\uc218")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)

    enc = get_default_encoder(device=args.device)
    samples = load_dataset(args.dataset, limit=args.n)

    for s in samples:
        print("\n" + "=" * 90)
        print(s.summary())
        print(f"  Q: {s.question}")
        print(f"  GT: {s.answer} (LLM pred: {s.predicted_answer})")
        print()

        # 5가지 비교
        # Q-D : doc이 question을 커버 → relevance
        # T-D : think 가 doc과 일치
        # T-Q : think 방향이 question 과 맞음
        # Sr-Q: search query가 의도 반영
        # Sr-D: search 결과 doc이 유용
        results = {}
        results["Q-D (relevance)  "] = tasi(s.Q, s.D, enc, steps=None)
        results["T-D (consistency)"] = tasi(s.T, s.D, enc, steps=s.T_steps)
        results["T-Q (alignment)  "] = tasi(s.T, s.Q, enc, steps=s.T_steps)
        results["Sr-Q (search_qual)"] = tasi(s.Sr, s.Q, enc)
        results["Sr-D (retrieval) "] = tasi(s.Sr, s.D, enc)

        prod = 1.0
        for name, r in results.items():
            print(f"  {name}: tasi={r.tasi:.3f}  wa={r.wa:.3f}  pc={r.pc:.3f}  "
                  f"|A|={r.n_a}, |B|={r.n_b}")
            prod *= r.tasi
        print(f"  --- product (total) = {prod:.4f}")

        # Q-D matched pairs 일부 출력 (debug)
        r_qd = results["Q-D (relevance)  "]
        print("\n  [Q→D] best matches (top 4 by weight):")
        if r_qd.weights_a is not None:
            order = np.argsort(-np.asarray(r_qd.weights_a))[:4]
            for i in order:
                j = r_qd.matched_pairs[i][1]
                score = r_qd.matched_pairs[i][2]
                print(f"     w={r_qd.weights_a[i]:.3f} score={score:.3f}")
                print(f"       Q : {s.Q[i].head} | {s.Q[i].relation} | {s.Q[i].tail}")
                print(f"       D : {s.D[j].head} | {s.D[j].relation} | {s.D[j].tail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
