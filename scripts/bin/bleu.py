"""
Infill 결과 트리플과 문서 트리플의 BLEU 기반 정합성을 검증하는 스크립트
"""

import argparse
import json
import os
import re
from typing import List, Tuple

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--bleu_n",
        type=int,
        default=2,
        choices=[1, 2],
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().strip(",").strip('"').strip("'")


def parse_response_triples(response: str) -> List[str]:
    if not isinstance(response, str):
        return []
    text = response.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean_text(x) for x in parsed if clean_text(x)]
        if isinstance(parsed, str):
            text = parsed
    except Exception:
        pass

    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return [clean_text(x) for x in quoted if clean_text(x)]

    if "\n" in text:
        lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
        if lines:
            return lines

    parts = re.split(r'"\s*,\s*"', text)
    if len(parts) > 1:
        parsed_parts = [clean_text(p) for p in parts if clean_text(p)]
        if parsed_parts:
            return parsed_parts

    fallback = [clean_text(x) for x in re.split(r"(?<=[.!?])\s+", text) if clean_text(x)]
    return fallback


def tokenize_triplet(triple_text: str) -> List[str]:
    text = clean_text(triple_text)
    text = re.sub(r"\s*\[\s*sep\s*\]\s*", " [SEP] ", text, flags=re.IGNORECASE)
    return text.split()


def bleu_score(candidate_text: str, reference_text: str, bleu_n: int = 2) -> float:
    candidate = tokenize_triplet(candidate_text)
    reference = tokenize_triplet(reference_text)

    if not candidate or not reference:
        return 0.0

    if bleu_n == 1:
        weights = (1.0, 0.0, 0.0, 0.0)
    else:
        weights = (0.5, 0.5, 0.0, 0.0)

    return float(
        sentence_bleu(
            references=[reference],
            hypothesis=candidate,
            weights=weights,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def best_bleu_match(query_triple: str, doc_triples: List[str], bleu_n: int) -> Tuple[str, float]:
    best_triple = ""
    best_score = -1.0
    for d in doc_triples:
        score = bleu_score(query_triple, d, bleu_n=bleu_n)
        if score > best_score:
            best_score = score
            best_triple = d
    return best_triple, max(best_score, 0.0)


def main():
    args = parse_args()

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    if args.output_file is None:
        input_stem = os.path.splitext(os.path.basename(args.data_file))[0]
        output_dir = os.path.dirname(args.data_file)
        tau_str = str(args.tau).replace(".", "p")
        args.output_file = os.path.join(output_dir, f"{input_stem}_bleu{args.bleu_n}_tau{tau_str}.json")

    results = []

    for sample in tqdm(samples, desc="BLEU verify"):
        index = sample.get("index")
        response = sample.get("infill_result", {}).get("response", "")
        question_triples_filled = parse_response_triples(response)
        doc_triples = sample.get("document_graph", {}).get("triples", [])

        if not isinstance(doc_triples, list):
            doc_triples = []

        # [SEP] 개수/위치를 포함해 원래 트리플 형태를 그대로 유지합니다.
        doc_triples = [clean_text(x) for x in doc_triples if clean_text(x)]
        question_triples_filled = [clean_text(x) for x in question_triples_filled if clean_text(x)]

        triple_results = []
        for q in question_triples_filled:
            best_triple, best_score = best_bleu_match(q, doc_triples, args.bleu_n) if doc_triples else ("", 0.0)
            triple_results.append(
                {
                    "question_triple_filled": q,
                    "best_document_triple": best_triple,
                    "bleu_score": round(float(best_score), 6),
                    "is_true": bool(best_score >= args.tau),
                }
            )

        all_true = bool(triple_results) and all(x["is_true"] for x in triple_results)
        results.append(
            {
                "index": index,
                "verification": {
                    "bleu_n": args.bleu_n,
                    "tau": args.tau,
                    "num_question_triples": len(question_triples_filled),
                    "num_document_triples": len(doc_triples),
                    "triple_results": triple_results,
                    "all_triples_true": all_true,
                    "claim_supported": all_true,
                },
            }
        )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output_file}")
    print(f"Processed samples: {len(results)}")


if __name__ == "__main__":
    main()
