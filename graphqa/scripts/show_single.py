"""단일 샘플 풀이 결과 (5 TASI score + slot filling + answer) 자세히 출력."""
from __future__ import annotations

import argparse
import logging
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from graphqa.data import DATASETS, load_dataset
from graphqa.pipeline import TASIPipeline
from graphqa.qa import answer_question
from graphqa.tasi.embedding import get_default_encoder


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="2wikimultihopqa", choices=list(DATASETS))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)

    encoder = get_default_encoder(device=args.device)
    pipe = TASIPipeline(encoder=encoder)

    samples = load_dataset(args.dataset, limit=args.n)
    for s in samples:
        scores = pipe.score_sample(s)
        qa = answer_question(s, encoder)

        print("\n" + "=" * 90)
        print(s.summary())
        print(f"  Q  : {s.question}")
        print(f"  GT : {s.answer}  (LLM={s.predicted_answer})")
        print(f"  PRED(TASI-QA) : {qa.predicted_answer}  | EM={qa.em:.0f} F1={qa.f1:.2f} | yesno={qa.is_yesno}")

        print("  TASI scores:")
        for name in ("relevance", "consistency", "alignment", "search_quality", "retrieval"):
            r = getattr(scores, name)
            print(f"    {name:>15s} : tasi={r.tasi:.3f}  wa={r.wa:.3f}  pc={r.pc:.3f}")
        print(f"    {'TOTAL (Π)':>15s} : {scores.total:.5f}")

        print("  slot fillings:")
        for slot, cand in sorted(qa.slot_fillings.items()):
            print(f"    {slot:>6s} = {cand.value!r}  (cum_score={cand.score:.3f}, |evidence|={len(cand.support_pairs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
