import argparse
import json
import logging
import os
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from direct import Direct


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--input_filename", type=str, default=None)
    parser.add_argument("--input_file_path", type=str, default=None)
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--evidence_setting", type=str, choices=["open-book", "open-book+gold", "gold"], default="open-book")
    parser.add_argument("--use_searchr1", action="store_true")
    parser.add_argument("--nudge_searchr1", action="store_true")
    parser.add_argument("--use_total_search_results", action="store_true")
    parser.add_argument("--searchr1_top_k", type=int, default=3)
    parser.add_argument("--searchr1_max_turns", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_filename", type=str, default=None)
    return parser.parse_args()


def resolve_input_args(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.input_file_path:
        input_path = os.path.abspath(args.input_file_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if args.dataset:
            dataset = args.dataset
        else:
            parts = os.path.normpath(input_path).split(os.sep)
            if "datasets" in parts:
                dataset_idx = parts.index("datasets") + 1
                if dataset_idx < len(parts):
                    dataset = parts[dataset_idx]
                else:
                    raise ValueError(f"Could not infer dataset from input path: {input_path}")
            else:
                path_lower = input_path.lower()
                if "2wiki" in path_lower or "2wikimultihopqa" in path_lower:
                    dataset = "2wikimultihopqa"
                elif "hotpotqa" in path_lower:
                    dataset = "hotpotqa"
                elif "musique" in path_lower:
                    dataset = "musique"
                else:
                    raise ValueError("When using --input_file_path outside datasets/<dataset>/claims/, --dataset must also be provided.")

        input_filename = os.path.basename(input_path)
        return dataset, input_filename, input_path

    if not args.dataset or not args.input_filename:
        raise ValueError("Provide either --input_file_path or both --dataset and --input_filename.")

    input_path = os.path.join("datasets", args.dataset, "claims", args.input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return args.dataset, args.input_filename, input_path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    return " ".join(text.split())


def retrieve_with_reasoning(
    sample: Dict[str, Any],
    direct_model: Direct,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Any]]:
    retrieved_documents = []
    doc_id_list = []
    gold_ids = {unicodedata.normalize("NFC", doc_id).strip() for doc_id in sample.get("gold_id_list", [])}
    is_gold_list = []
    search_info: Dict[str, Any] = {}

    if args.evidence_setting in {"open-book", "open-book+gold"}:
        hit_list, search_info = direct_model.retrieve(
            sample["question"],
            top_k=args.bm25_top_k,
            use_searchr1=args.use_searchr1,
            use_total_search_results=args.use_total_search_results,
            nudge_searchr1=args.nudge_searchr1,
        )
        for hit in hit_list:
            doc_id = unicodedata.normalize("NFC", hit["doc_id"]).strip()
            text = unicodedata.normalize("NFC", hit["text"]).strip()
            evidence = f"(Title: {doc_id}) {text}"
            if evidence not in retrieved_documents:
                retrieved_documents.append(evidence)
                doc_id_list.append(doc_id)
                is_gold_list.append(1 if doc_id in gold_ids else 0)

    if args.evidence_setting in {"gold", "open-book+gold"}:
        for doc_id, text in zip(sample.get("gold_id_list", []), sample.get("gold_evidence_list", [])):
            normalized_doc_id = unicodedata.normalize("NFC", doc_id).strip()
            normalized_text = unicodedata.normalize("NFC", text).strip()
            evidence = f"(Title: {normalized_doc_id}) {normalized_text}"
            if evidence not in retrieved_documents:
                retrieved_documents.append(evidence)
                doc_id_list.append(normalized_doc_id)
                is_gold_list.append(1)

    retrieval_info: Dict[str, Any] = {
        "query": sample["question"],
        "evidence_setting": args.evidence_setting,
        "doc_id_list": doc_id_list,
        "is_gold_list": is_gold_list,
    }
    if args.use_searchr1:
        retrieval_info["full_response"] = search_info.get("full_response", "")
        retrieval_info["searchr1_answer"] = search_info.get("predicted_answer", "")
        retrieval_info["searchr1_reasoning_path"] = search_info.get("reasoning_path", "")
        retrieval_info["searchr1_reasoning_steps"] = search_info.get("reasoning_steps", [])
        retrieval_info["num_turns"] = search_info.get("num_turns", 0)
        retrieval_info["retrieval_turns"] = search_info.get("retrieval_turns", [])
    return retrieved_documents, retrieval_info


def process_sample(
    sample: Dict[str, Any],
    direct_model: Direct,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    retrieved_documents, retrieval_info = retrieve_with_reasoning(sample, direct_model, args)
    predicted_answer = retrieval_info.get("searchr1_answer", "")
    gold_answer = normalize_text(sample.get("answer", ""))

    result = {
        "index": sample.get("index"),
        "uid": sample.get("uid"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "answer_aliases": sample.get("answer_aliases", []),
        "num_hops": sample.get("num_hops"),
        "gold_id_list": sample.get("gold_id_list", []),
        "retrieved_documents": retrieved_documents,
        "retrieval_info": retrieval_info,
    }
    if args.use_searchr1:
        result["predicted_answer"] = predicted_answer
        result["reasoning_path"] = retrieval_info.get("searchr1_reasoning_path", "")
        result["reasoning_steps"] = retrieval_info.get("searchr1_reasoning_steps", [])
        result["answer_matches_gold"] = normalize_text(predicted_answer) == gold_answer if predicted_answer else None
    return result


def main() -> None:
    args = parse_args()
    args.dataset, args.input_filename, input_path = resolve_input_args(args)

    direct_args = argparse.Namespace(
        dataset=args.dataset,
        input_filename=args.input_filename,
        direct_filename=None,
        base_model_name=args.base_model_name,
        setting=args.evidence_setting,
        bm25_top_k=args.bm25_top_k,
        use_searchr1=args.use_searchr1,
        searchr1_top_k=args.searchr1_top_k,
        searchr1_max_turns=args.searchr1_max_turns,
        use_total_search_results=args.use_total_search_results,
        retriever_url=args.retriever_url,
    )
    direct_model = Direct(direct_args)

    with open(input_path, "r") as handle:
        input_list = json.load(handle)
    if args.max_samples is not None:
        input_list = input_list[:args.max_samples]

    result_list = []
    for sample in tqdm(input_list):
        try:
            result_list.append(process_sample(sample, direct_model, args))
        except Exception as exc:
            logger.error(f"Failed to process sample {sample.get('index')}: {exc}")
            result_list.append({
                "index": sample.get("index"),
                "uid": sample.get("uid"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "error": str(exc),
            })

    base_output_name = args.output_filename or f"vanilla_{args.input_filename}"
    output_stem, output_ext = os.path.splitext(base_output_name)
    if not output_ext:
        output_ext = ".json"
    slurm_job_id = os.getenv("SLURM_JOB_ID", os.getenv("JOB_ID", "local"))
    mode_name = "searchr1" if args.use_searchr1 else "retriever"
    output_filename = f"{output_stem}_{args.evidence_setting}_{mode_name}_{slurm_job_id}_{len(result_list)}{output_ext}"
    output_dir = args.output_dir or os.path.join("results", args.dataset, "vanilla")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as handle:
        json.dump(result_list, handle, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
