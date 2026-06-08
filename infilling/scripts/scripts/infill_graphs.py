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
from copy import deepcopy
from types import SimpleNamespace
from tqdm import tqdm

# 경로 설정: graphcheck-qa 루트(model_library), infilling/scripts(scripts.utils), infilling/scripts/scripts(prompt)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_GRAPHCHECK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))  # graphcheck-qa/
for _p in (_GRAPHCHECK_ROOT, _PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompt import (INFILL_PROMPT_BASE, 
                    TRIPLET_ONLY_TRIPLET_ONLY_EXAMPLES, TRIPLET_ONLY_DOC_ONLY_EXAMPLES, TRIPLET_ONLY_COMBINED_EXAMPLES, 
                    COMBINED_DOC_ONLY_EXAMPLES, COMBINED_TRIPLET_ONLY_EXAMPLES, COMBINED_COMBINED_EXAMPLES)
from scripts.utils.preprocess import extract_documents, extract_triples, select_graph
from scripts.utils.model import load_model
from utils.graph import select_topk_doc_triplets_by_ensemble

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)

# response에 (ENT) 존재 시 재시도를 위함
ENT_PLACEHOLDER_PATTERN = re.compile(r"\(ENT\d+\)|\bENT\d*\b", re.IGNORECASE)     # 강한 버전(예: ENT, ENT1이 남아 있으면 재시도)
# ENT_PLACEHOLDER_PATTERN = re.compile(r"\(ENT\d+\)", re.IGNORECASE)              # 약한 버전(예: (ENT1), (ENT2)가 남아 있으면 재시도)


TRIPLET_SELECTION_CHOICES = ("all", "top1", "top3", "top5", "top10")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", required=True) 
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--question_strategy', type=str, choices=["triplet_only", "combined"], required=True)
    parser.add_argument('--infill_strategy', type=str, choices=["triplet_only", "doc_only", "combined"], required=True)
    parser.add_argument('--use_gold_only', type=int, choices=[0, 1], default=0)
    parser.add_argument('--max_trials', type=int, default=3)
    # 실험: 전체 문서 O/X
    parser.add_argument('--use_full_doc', type=int, choices=[0, 1], default=0,
        help="1: 원문 문서 포함(doc_only/combined), 0: triplet만(triplet_only)")
    # 실험: triplet 후보 선택 (all | top1 | top3 | top5 | top10)
    parser.add_argument('--triplet_selection', type=str, choices=TRIPLET_SELECTION_CHOICES, default="all",
        help="all: 전체 doc triplet, top-k: ensemble 유사도로 상위 k개")
    # 실험: document만 사용 (triplet 제외). 1이면 <<Documents>>만 제공, <<Document Triplets>> 미제공
    parser.add_argument('--force_doc_only', type=int, choices=[0, 1], default=0,
        help="1: 원문 문서만 사용, document triplet 미사용 (doc_only 모드)")
    parser.add_argument('--max_samples', type=int, default=None,
        help="테스트용: 처리할 샘플 수 제한 (None=전체)")
    parser.add_argument(
        '--run_all_settings',
        type=int,
        choices=[0, 1],
        default=0,
        help="1이면 triplet/all~top10 + fulldoc/all~top10 + doconly를 한 번에 실행",
    )

    return parser.parse_args()


def _triplet_selection_to_k(selection: str) -> int:
    """triplet_selection 문자열을 top-k 정수로 변환"""
    if selection == "all":
        return -1
    if selection.startswith("top"):
        try:
            return int(selection[3:])
        except ValueError:
            pass
    return -1


def infill_prompt(args, sample) -> str:
    """각 infill strategy에 따른 prompt 형식 추출"""
    question_triples = sample.get("question_graph", {}).get("triples", [])
    question_text = sample.get("question", "")

    _, source_graph = select_graph(sample, args.use_gold_only)
    documents = extract_documents(source_graph)
    all_document_triples = extract_triples(source_graph)

    # triplet_selection: all | top1 | top3 | top5 | top10
    top_k = _triplet_selection_to_k(getattr(args, "triplet_selection", "all"))
    if top_k >= 1 and question_triples and all_document_triples:
        document_triples = select_topk_doc_triplets_by_ensemble(
            question_triples, all_document_triples, top_k=top_k
        )
    else:
        document_triples = all_document_triples

    # use_full_doc: 1이면 원문 문서 포함, 0이면 triplet만
    # force_doc_only: 1이면 document만 사용 (triplet 무시)
    use_full_doc = getattr(args, "use_full_doc", 0)
    force_doc_only = getattr(args, "force_doc_only", 0)
    if force_doc_only == 1:
        effective_infill = "doc_only"
        document_triples = []  # triplet 미사용
    elif use_full_doc == 1:
        effective_infill = "combined" if document_triples else "doc_only"
    else:
        effective_infill = "triplet_only"

    question_triples_text = "\n".join(question_triples) if isinstance(question_triples, list) else str(question_triples)
    documents_text = "\n".join(documents) if isinstance(documents, list) else str(documents)
    document_triples_text = "\n".join(document_triples) if isinstance(document_triples, list) else str(document_triples)

    if args.question_strategy == "triplet_only":        # {Triplet Question}만
        if effective_infill == "triplet_only":          # {Triplet Documment}만
            PROMPT = INFILL_PROMPT_BASE + TRIPLET_ONLY_TRIPLET_ONLY_EXAMPLES
            TRIPLET_ONLY_TRIPLET_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n" +
                                                "<<Document Triplets>>" + "\n" + document_triples_text + "\n" +
                                                "<<Answer>>")

            return TRIPLET_ONLY_TRIPLET_ONLY_PROMPT

        if effective_infill == "doc_only":              # {Original Document}만
            PROMPT = INFILL_PROMPT_BASE + TRIPLET_ONLY_DOC_ONLY_EXAMPLES
            TRIPLET_ONLY_DOC_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                            "<<Documents>>" "\n" + documents_text + "\n"+
                                            "<<Answer>>")

            return TRIPLET_ONLY_DOC_ONLY_PROMPT
        
        if effective_infill == "combined":              # {Original + Triplet Document}
            PROMPT = INFILL_PROMPT_BASE + TRIPLET_ONLY_COMBINED_EXAMPLES
            TRIPLET_ONLY_COMBINED_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                            "<<Documents>>" + "\n"+ documents_text + "\n"+
                                            "<<Document Triplets>>" + "\n"+ document_triples_text + "\n"+
                                            "<<Answer>>")
            
            return TRIPLET_ONLY_COMBINED_PROMPT

    if args.question_strategy == "combined":            # {Triplet Question + Original Question} (not recommended)
        if effective_infill == "triplet_only":          # {Triplet Documment}만
            PROMPT = INFILL_PROMPT_BASE + COMBINED_TRIPLET_ONLY_EXAMPLES
            COMBINED_TRIPLET_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                            "<<Question>>" + "\n"+ question_text + "\n"+
                                            "<<Document Triplets>>" + "\n"+ document_triples_text + "\n"+
                                            "<<Answer>>")

            return COMBINED_TRIPLET_ONLY_PROMPT

        if effective_infill == "doc_only":              # {Original Document}만
            PROMPT = INFILL_PROMPT_BASE + COMBINED_DOC_ONLY_EXAMPLES
            COMBINED_DOC_ONLY_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                        "<<Question>>" + "\n"+ question_text + "\n"+
                                        "<<Documents>>" + "\n"+ documents_text + "\n"+
                                        "<<Answer>>")
            
            return COMBINED_DOC_ONLY_PROMPT
        
        
        if effective_infill == "combined":              # {Original + Triplet Document}
            PROMPT = INFILL_PROMPT_BASE + COMBINED_COMBINED_EXAMPLES
            COMBINED_COMBINED_PROMPT = (PROMPT + "\n"+ question_triples_text + "\n"+
                                        "<<Question>>" + "\n"+ question_text + "\n"+
                                        "<<Documents>>" + "\n"+ documents_text +"\n"+
                                        "<<Document Triplets>>" + "\n"+ document_triples_text + "\n"+
                                        "<<Answer>>")
            
            return COMBINED_COMBINED_PROMPT

    raise ValueError(
        f"Unsupported strategy combination: question_strategy={args.question_strategy}, "
        f"effective_infill={effective_infill}"
    )


def contains_ent_placeholder(text: str) -> bool:
    """infill 했는데도 ENT가 남아있는지 확인"""
    if not isinstance(text, str):
        return False
    return ENT_PLACEHOLDER_PATTERN.search(text) is not None


def build_output_path(args) -> str:
    input_stem = os.path.splitext(os.path.basename(args.data_file))[0]
    dataset_name = input_stem.split("_")[0]
    source_tag = "gold" if args.use_gold_only == 1 else "all"
    force_doc_only = getattr(args, "force_doc_only", 0)
    if force_doc_only == 1:
        use_full_doc_tag = "doconly"
        triplet_sel = "all"
    else:
        use_full_doc_tag = "fulldoc" if getattr(args, "use_full_doc", 0) == 1 else "triplet"
        triplet_sel = getattr(args, "triplet_selection", "all")
    output_name = (
        f"infill_{dataset_name}_{args.question_strategy}_{args.infill_strategy}_"
        f"{source_tag}_{use_full_doc_tag}_{triplet_sel}.json"
    )
    return os.path.join(args.output_dir, output_name)


def run_one_setting(args, base_samples, model_client):
    results = []
    logger.info(
        "Starting infill: samples=%d, model=%s, use_full_doc=%s, force_doc_only=%s, triplet_selection=%s",
        len(base_samples),
        args.model_name,
        getattr(args, "use_full_doc", 0),
        getattr(args, "force_doc_only", 0),
        getattr(args, "triplet_selection", "all"),
    )

    for sample in tqdm(base_samples, desc="Infilling!"):
        sample_out = deepcopy(sample)
        prompt = None
        try:
            prompt = infill_prompt(args, sample_out)
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
                        sample_out.get("index"),
                        trial + 1,
                        args.max_trials,
                    )

            sample_out["infill_result"] = {
                "question_strategy": args.question_strategy,
                "infill_strategy": args.infill_strategy,
                "use_full_doc": getattr(args, "use_full_doc", 0),
                "force_doc_only": getattr(args, "force_doc_only", 0),
                "triplet_selection": getattr(args, "triplet_selection", "all"),
                "trials": used_trials,
                "ent_exist_flag": contains_ent_placeholder(response),
                "response": response,
                "prompt": prompt,
            }
        except Exception as e:
            logger.exception("Failed infill for sample index=%s", sample_out.get("index"))
            sample_out["infill_result"] = {
                "question_strategy": args.question_strategy,
                "infill_strategy": args.infill_strategy,
                "use_full_doc": getattr(args, "use_full_doc", 0),
                "force_doc_only": getattr(args, "force_doc_only", 0),
                "triplet_selection": getattr(args, "triplet_selection", "all"),
                "response": None,
                "prompt": prompt,
                "error": str(e),
            }
        results.append(sample_out)

    output_path = build_output_path(args)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Infill result 결과 저장: %s", output_path)


def main():
    args = parse_args()

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if getattr(args, "max_samples", None) is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
        logger.info("Limited to %d samples (--max_samples)", len(samples))

    os.makedirs(args.output_dir, exist_ok=True)
    model_client = load_model(args.model_name)
    logger.info("Model loaded once: %s", args.model_name)

    if args.run_all_settings == 1:
        settings = []
        for sel in TRIPLET_SELECTION_CHOICES:
            settings.append((0, sel, 0))
        for sel in TRIPLET_SELECTION_CHOICES:
            settings.append((1, sel, 0))
        settings.append((1, "all", 1))

        for use_full_doc, triplet_selection, force_doc_only in settings:
            setting_args = SimpleNamespace(**vars(args))
            setting_args.use_full_doc = use_full_doc
            setting_args.triplet_selection = triplet_selection
            setting_args.force_doc_only = force_doc_only
            run_one_setting(setting_args, samples, model_client)
        return

    run_one_setting(args, samples, model_client)


if __name__ == "__main__":
    main()
