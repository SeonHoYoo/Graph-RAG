"""
Infill된 Question Triplet을 기반으로 최종 답변을 생성하고, GraphCheck 방식의 EM/F1을 계산하는 스크립트

입력은 아래 데이터를 포함한 sample list 형태를 가정합니다...
- question / answer / answer_aliases
- response ((ENT)가 채워진 question triples 문자열)
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, List

from tqdm import tqdm

# 경로 설정: graphcheck-qa 루트(utils.metrics, model_library), infilling/scripts(scripts.utils)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_GRAPHCHECK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
for _p in (_GRAPHCHECK_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--ent_exist_flag", type=str, choices=["all", "false"], default="false")
    args = parser.parse_args()
    if bool(args.data_file) == bool(args.data_dir):
        raise ValueError("Exactly one of --data_file or --data_dir must be provided.")
    return args


def parse_response_triples(response: Any) -> List[str]:
    """infilled triplet 전처리"""
    if not isinstance(response, str):
        return []
    text = response.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    if text.startswith('"') and text.endswith('"') and '", "' in text:
        parts = text.split('", "')
        parts[0] = parts[0][1:] if parts[0].startswith('"') else parts[0]
        parts[-1] = parts[-1][:-1] if parts[-1].endswith('"') else parts[-1]
        parsed = [p.strip() for p in parts if p.strip()]
        if parsed:
            return parsed

    lines = [line.strip().strip(",") for line in text.splitlines()]
    parsed = [line.strip().strip('"') for line in lines if line]
    return [p for p in parsed if p]


def build_graph_evidence(sample: dict) -> str:
    """infilled triplet 전처리 2 (graphcheck 방식)"""
    response = sample.get("infill_result", {}).get("response", "")
    infilled_triples = parse_response_triples(response)
    if not infilled_triples:
        raise ValueError("Failed to parse infill_result.response into triples.")

    graph_evidence = "\n".join(infilled_triples)
    graph_evidence = graph_evidence.replace("[SEP] ", "").replace("[PREP] ", "")
    return graph_evidence.strip()


def normalize_predicted_answer(answer: Any) -> str:
    """answer 전처리"""
    if not isinstance(answer, str):
        return ""
    first_line = answer.splitlines()[0].strip() if answer.splitlines() else ""
    return first_line.strip().strip('"').strip("'")


def build_answer_prompt(question: str, graph_evidence: str) -> str:
    """{question + infill triplet} 제공하는 프롬프트"""
    if graph_evidence:
        return (
            "Answer the question using the context. "
            "Return only a short answer phrase.\n"
            f"Context: {graph_evidence}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return f"Question: {question}\nAnswer:"


def process_one_file(args: argparse.Namespace, model_client, data_file: str) -> None:
    with open(data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    total_samples = len(samples)
    resolved_samples = sum(
        1 for s in samples if s.get("infill_result", {}).get("ent_exist_flag", None) is False
    )
    processed_all = total_samples
    processed_false = resolved_samples
    processed_current = processed_all if args.ent_exist_flag == "all" else processed_false

    logger.info("data_file=%s", data_file)
    logger.info("total_samples=%d", total_samples)
    logger.info("processed_samples[all]=%d", processed_all)
    logger.info("processed_samples[false]=%d", processed_false)
    logger.info("processed_samples[current_scope]=%d", processed_current)

    processed_count = 0
    skipped_count = 0
    for sample in tqdm(samples, desc=f"Answering:{os.path.basename(data_file)}"):
        # ent_exist_flag = true인 경우 모든 (ENT)가 infill 되지 않은 경우임
        ent_exist_flag = sample.get("infill_result", {}).get("ent_exist_flag", None)
        if args.ent_exist_flag == "false" and ent_exist_flag is not False:
            skipped_count += 1
            sample["answer_skipped"] = True
            sample["answer_skip_reason"] = "ent_exist_flag_is_not_false"
            continue

        question = sample.get("question", "")
        try:
            graph_evidence = build_graph_evidence(sample)
        except ValueError as e:
            idx = sample.get("index", "unknown")
            raise ValueError(f"Sample index={idx}: {e}") from e
        prompt = build_answer_prompt(question, graph_evidence)

        answer = ""
        for _ in range(max(1, args.max_trials)):
            answer = model_client.generate(
                user_message=prompt,
                max_tokens=256,
                temperature=0.0,
            )
            answer = normalize_predicted_answer(answer)
            if answer:
                break

        ground_truth_answers = [sample.get("answer", "")] + sample.get("answer_aliases", [])
        ground_truth_answers = [x for x in ground_truth_answers if isinstance(x, str)]

        em_score = metric_max_over_ground_truths(compute_exact, answer, ground_truth_answers) if ground_truth_answers else 0.0
        f1_score = metric_max_over_ground_truths(compute_f1, answer, ground_truth_answers) if ground_truth_answers else 0.0

        # Keep gold aliases untouched; do not inject model prediction into gold set.
        sample["predicted_answer"] = answer
        sample["graph_evidence"] = graph_evidence
        sample["answer_prompt"] = prompt
        sample["em_score"] = em_score
        sample["f1_score"] = f1_score
        sample["answer_skipped"] = False
        sample["answer_skip_reason"] = None
        processed_count += 1

    input_stem = os.path.splitext(os.path.basename(data_file))[0]
    # 단순 파일명 수정
    if input_stem.startswith("infill_"):
        input_stem = input_stem[len("infill_"):]
    output_name = f"answer_{input_stem}_{args.ent_exist_flag}.json"
    output_path = os.path.join(args.output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # score 출력
    valid = [s for s in samples if isinstance(s.get("em_score"), (int, float))]
    avg_em = sum(s.get("em_score", 0.0) for s in valid) / len(valid) if valid else 0.0
    avg_f1 = sum(s.get("f1_score", 0.0) for s in valid) / len(valid) if valid else 0.0
    logger.info("Saved: %s", output_path)
    logger.info("Processed=%d, Skipped=%d", processed_count, skipped_count)
    logger.info("Average EM=%.4f, Average F1=%.4f", avg_em, avg_f1)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_client = load_model(args.model_name)
    logger.info("Loaded model=%s, ent_exist_flag=%s", args.model_name, args.ent_exist_flag)
    if args.data_file:
        process_one_file(args, model_client, args.data_file)
        return

    infill_files = sorted(
        f for f in os.listdir(args.data_dir)
        if f.startswith("infill_") and f.endswith(".json")
    )
    if not infill_files:
        logger.warning("No infill_*.json found in %s", args.data_dir)
        return

    for fname in infill_files:
        process_one_file(args, model_client, os.path.join(args.data_dir, fname))


if __name__ == "__main__":
    main()
