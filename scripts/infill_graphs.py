"""
Query Graph에 대한 Entity를 LLM이 infill하게 하는 스크립트
이미 Triplet 생성이 완료된 데이터 형태를 가정합니다...
"""

import argparse
import json
import logging
import os
import re
import sys
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt import (INFILL_PROMPT_BASE, 
                    TRIPLET_ONLY_TRIPLET_ONLY_EXAMPLES, TRIPLET_ONLY_DOC_ONLY_EXAMPLES, TRIPLET_ONLY_COMBINED_EXAMPLES, 
                    COMBINED_DOC_ONLY_EXAMPLES, COMBINED_TRIPLET_ONLY_EXAMPLES, COMBINED_COMBINED_EXAMPLES)
from scripts.utils.preprocess import extract_documents, extract_triples, select_graph
from scripts.utils.model import load_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)

# response에 (ENT) 존재 시 재시도를 위함
ENT_PLACEHOLDER_PATTERN = re.compile(r"\(ENT\d+\)|\bENT\d*\b", re.IGNORECASE)     # 강한 버전(예: ENT, ENT1이 남아 있으면 재시도)
# ENT_PLACEHOLDER_PATTERN = re.compile(r"\(ENT\d+\)", re.IGNORECASE)              # 약한 버전(예: (ENT1), (ENT2)가 남아 있으면 재시도)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", required=True) 
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--question_strategy', type=str, choices=["triplet_only", "combined"], required=True)
    parser.add_argument('--infill_strategy', type=str, choices=["triplet_only", "doc_only", "combined"], required=True)
    parser.add_argument('--use_gold_only', type=int, choices=[0, 1], default=0)
    parser.add_argument('--max_trials', type=int, default=3)

    return parser.parse_args()


def normalize_graph_text(text: str) -> str:
    """baseline.py와 동일하게 [SEP], [PREP] 토큰 제거"""
    if not isinstance(text, str):
        return ""
    normalized = text.replace("[SEP] ", "").replace("[PREP] ", "")
    normalized = normalized.replace("[SEP]", " ").replace("[PREP]", " ")
    return normalized.strip()


def infill_prompt(args, sample) -> str:
    """각 infill strategy에 따른 prompt 형식 추출"""
    question_triples = sample.get("question_graph", {}).get("triples", [])
    question_text = sample.get("question", "")

    _, source_graph = select_graph(sample, args.use_gold_only)
    documents = extract_documents(source_graph)
    document_triples = extract_triples(source_graph)

    question_triples_text = "\n".join(question_triples) if isinstance(question_triples, list) else str(question_triples)
    documents_text = "\n".join(documents) if isinstance(documents, list) else str(documents)
    document_triples_text = "\n".join(document_triples) if isinstance(document_triples, list) else str(document_triples)

    # document triplets만 baseline.py와 맞추기 위해 [SEP]/[PREP] 제거
    # target question triplets는 원본 포맷([SEP] 포함)을 유지
    document_triples_text = normalize_graph_text(document_triples_text)

    if args.question_strategy == "triplet_only":        # {Triplet Question}만
        if args.infill_strategy == "triplet_only":          # {Triplet Documment}만
            PROMPT = INFILL_PROMPT_BASE + TRIPLET_ONLY_TRIPLET_ONLY_EXAMPLES
            TRIPLET_ONLY_TRIPLET_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n" +
                                                "<<Document Triplets>>" + "\n" + document_triples_text + "\n" +
                                                "<<Answer>>")

            return TRIPLET_ONLY_TRIPLET_ONLY_PROMPT

        if args.infill_strategy == "doc_only":              # {Original Document}만
            PROMPT = INFILL_PROMPT_BASE + TRIPLET_ONLY_DOC_ONLY_EXAMPLES
            TRIPLET_ONLY_DOC_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                            "<<Documents>>" "\n" + documents_text + "\n"+
                                            "<<Answer>>")

            return TRIPLET_ONLY_DOC_ONLY_PROMPT
        
        if args.infill_strategy == "combined":              # {Original + Triplet Document}
            PROMPT = INFILL_PROMPT_BASE + TRIPLET_ONLY_COMBINED_EXAMPLES
            TRIPLET_ONLY_COMBINED_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                            "<<Documents>>" + "\n"+ documents_text + "\n"+
                                            "<<Document Triplets>>" + "\n"+ document_triples_text + "\n"+
                                            "<<Answer>>")
            
            return TRIPLET_ONLY_COMBINED_PROMPT

    if args.question_strategy == "combined":            # {Triplet Question + Original Question} (not recommended)
        if args.infill_strategy == "triplet_only":          # {Triplet Documment}만
            PROMPT = INFILL_PROMPT_BASE + COMBINED_TRIPLET_ONLY_EXAMPLES
            COMBINED_TRIPLET_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                            "<<Question>>" + "\n"+ question_text + "\n"+
                                            "<<Document Triplets>>" + "\n"+ document_triples_text + "\n"+
                                            "<<Answer>>")

            return COMBINED_TRIPLET_ONLY_PROMPT

        if args.infill_strategy == "doc_only":              # {Original Document}만
            PROMPT = INFILL_PROMPT_BASE + COMBINED_DOC_ONLY_EXAMPLES
            COMBINED_DOC_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                        "<<Question>>" + "\n"+ question_text + "\n"+
                                        "<<Documents>>" + "\n"+ documents_text + "\n"+
                                        "<<Answer>>")
            
            return COMBINED_DOC_ONLY_PROMPT
        
        
        if args.infill_strategy == "combined":              # {Original + Triplet Document}
            PROMPT = INFILL_PROMPT_BASE + COMBINED_COMBINED_EXAMPLES
            COMBINED_COMBINED_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                        "<<Question>>" + "\n"+ question_text + "\n"+
                                        "<<Documents>>" + "\n"+ documents_text +"\n"+
                                        "<<Document Triplets>>" + "\n"+ document_triples_text + "\n"+
                                        "<<Answer>>")
            
            return COMBINED_COMBINED_PROMPT

    raise ValueError(
        f"Unsupported strategy combination: question_strategy={args.question_strategy}, "
        f"infill_strategy={args.infill_strategy}"
    )


def contains_ent_placeholder(text: str) -> bool:
    """infill 했는데도 ENT가 남아있는지 확인"""
    if not isinstance(text, str):
        return False
    return ENT_PLACEHOLDER_PATTERN.search(text) is not None


def main():
    args = parse_args()

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    model_client = load_model(args.model_name)

    results = []
    logger.info("Starting infill for %d samples with model=%s", len(samples), args.model_name)

    for sample in tqdm(samples, desc="Infilling!"):
        prompt = None
        try:
            prompt = infill_prompt(args, sample)
            response = None
            used_trials = 0

            for trial in range(1, args.max_trials + 1):
                used_trials = trial
                response = model_client.generate(
                    user_message=prompt,
                    max_tokens=512,
                    temperature=0.0,
                )

                if not contains_ent_placeholder(response):
                    break

                if trial < args.max_trials:
                    logger.info(
                        "Sample index=%s still contains (ENT). Retrying (%d/%d).",
                        sample.get("index"),
                        trial + 1,
                        args.max_trials,
                    )

            sample["infill_result"] = {
                "question_strategy": args.question_strategy,
                "infill_strategy": args.infill_strategy,
                "trials": used_trials,
                "ent_exist_flag": contains_ent_placeholder(response),
                "response": response,
                "prompt": prompt,
            }
        except Exception as e:
            logger.exception("Failed infill for sample index=%s", sample.get("index"))
            sample["infill_result"] = {
                "question_strategy": args.question_strategy,
                "infill_strategy": args.infill_strategy,
                "response": None,
                "prompt": prompt,
                "error": str(e),
            }
        results.append(sample)

    input_stem = os.path.splitext(os.path.basename(args.data_file))[0]
    dataset_name = input_stem.split("_")[0]
    source_tag = "gold" if args.use_gold_only == 1 else "all"
    output_name = (
        f"infill_{dataset_name}_{args.question_strategy}_{args.infill_strategy}_{source_tag}.json"
    )
    output_path = os.path.join(args.output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Infill result 결과 저장: %s", output_path)


if __name__ == "__main__":
    main()
