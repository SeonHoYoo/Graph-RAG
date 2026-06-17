"""
Question + Document 기반 baseline 답변 생성 스크립트
GraphCheck와 동일한 EM/F1 metric 함수를 사용합니다...
"""

import argparse
import json
import logging
import os
import sys
from typing import Any

from tqdm import tqdm

# 경로 설정: graphcheck-qa 루트(utils.metrics, model_library), infilling/scripts(scripts.utils)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_GRAPHCHECK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
for _p in (_GRAPHCHECK_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.utils.preprocess import extract_documents, extract_triples, select_graph
from scripts.utils.model import load_model
from utils.metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--context_strategy", type=str, choices=["doc", "docgraph", "combined"], default="combined")
    parser.add_argument("--use_gold_graph", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_gold_doc", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_trials", type=int, default=1)
    return parser.parse_args()


def build_raw_document_context(sample: dict, use_gold_graph: int) -> str:
    """raw_document 불러오기 전처리"""
    graph_key, graph = select_graph(sample, use_gold_graph)
    documents = extract_documents(graph)
    if not documents:
        raise ValueError(f"{graph_key}.per_document 문서가 비어 있습니다.")
    return "\n".join(documents).strip()


def build_document_graph_context(sample: dict, use_gold_doc: int) -> str:
    """document_graph 불러오기 전처리"""
    graph_key, graph = select_graph(sample, use_gold_doc)
    triples = extract_triples(graph)
    if not triples:
        raise ValueError(f"{graph_key}.triples가 비어 있습니다.")
    graph_evidence = "\n".join(triples)
    graph_evidence = graph_evidence.replace("[SEP] ", "").replace("[PREP] ", "")
    graph_evidence = graph_evidence.replace("[SEP]", " ").replace("[PREP]", " ")
    return graph_evidence.strip()


def build_context(
    context_strategy: str,
    raw_document_context: str = "",
    document_graph_context: str = "",
) -> str:
    if context_strategy == "doc":
        return raw_document_context
    if context_strategy == "docgraph":
        return document_graph_context
    return f"{raw_document_context}\n{document_graph_context}".strip()


def build_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using the context. "
        "Return only a short answer phrase.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def normalize_predicted_answer(answer: Any) -> str:
    if not isinstance(answer, str):
        return ""
    first_line = answer.splitlines()[0].strip() if answer.splitlines() else ""
    return first_line.strip().strip('"').strip("'")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    model_client = load_model(args.model_name)
    logger.info(
        "Loaded model=%s, context_strategy=%s, use_gold_graph=%d, use_gold_doc=%d, total_samples=%d",
        args.model_name,
        args.context_strategy,
        args.use_gold_graph,
        args.use_gold_doc,
        len(samples),
    )

    for sample in tqdm(samples, desc="Baseline..."):
        question = sample.get("question", "")
        raw_document_context = ""
        document_graph_context = ""
        if args.context_strategy in ["doc", "combined"]:
            raw_document_context = build_raw_document_context(sample, args.use_gold_graph)
        if args.context_strategy in ["docgraph", "combined"]:
            document_graph_context = build_document_graph_context(sample, args.use_gold_doc)

        context = build_context(
            args.context_strategy,
            raw_document_context=raw_document_context,
            document_graph_context=document_graph_context,
        )
        prompt = build_prompt(question, context)

        answer = ""
        for _ in range(max(1, args.max_trials)):
            answer = model_client.generate(
                user_message=prompt,
                max_tokens=64,
                temperature=0.0,
            )
            answer = normalize_predicted_answer(answer)
            if answer:
                break

        ground_truth_answers = [sample.get("answer", "")] + sample.get("answer_aliases", [])
        ground_truth_answers = [x for x in ground_truth_answers if isinstance(x, str)]

        em_score = metric_max_over_ground_truths(compute_exact, answer, ground_truth_answers) if ground_truth_answers else 0.0
        f1_score = metric_max_over_ground_truths(compute_f1, answer, ground_truth_answers) if ground_truth_answers else 0.0

        sample["baseline_context_strategy"] = args.context_strategy
        sample["baseline_use_gold_graph"] = args.use_gold_graph
        sample["baseline_use_gold_doc"] = args.use_gold_doc
        sample["baseline_prompt"] = prompt
        sample["baseline_context"] = context
        sample["baseline_raw_document_context"] = raw_document_context
        sample["baseline_document_graph_context"] = document_graph_context
        sample["predicted_answer"] = answer
        sample["em_score"] = em_score
        sample["f1_score"] = f1_score

    input_stem = os.path.splitext(os.path.basename(args.data_file))[0]
    raw_doc_tag = "gold" if args.use_gold_graph == 1 else "all"
    doc_graph_tag = "gold" if args.use_gold_doc == 1 else "all"
    output_name_parts = [f"baseline_{input_stem}", args.context_strategy]
    if args.context_strategy in ["doc", "combined"]:
        output_name_parts.append(f"rawdoc_{raw_doc_tag}")
    if args.context_strategy in ["docgraph", "combined"]:
        output_name_parts.append(f"docgraph_{doc_graph_tag}")
    output_name = "_".join(output_name_parts) + ".json"
    output_path = os.path.join(args.output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    avg_em = sum(s.get("em_score", 0.0) for s in samples) / len(samples) if samples else 0.0
    avg_f1 = sum(s.get("f1_score", 0.0) for s in samples) / len(samples) if samples else 0.0
    logger.info("Saved: %s", output_path)
    logger.info("Average EM=%.4f, Average F1=%.4f", avg_em, avg_f1)


if __name__ == "__main__":
    main()
