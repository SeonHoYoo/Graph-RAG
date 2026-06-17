import argparse
import json
import logging
import os
import random
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
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before slicing max_samples.")
    parser.add_argument("--seed", type=int, default=42)
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
        "uid": sample.get("uid") or sample.get("id"),
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
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(input_list)
    if args.max_samples is not None:
        input_list = input_list[:args.max_samples]

    base_output_name = args.output_filename or f"vanilla_{args.input_filename}"
    output_stem, output_ext = os.path.splitext(base_output_name)
    slurm_job_id = os.getenv("SLURM_JOB_ID", os.getenv("JOB_ID", "local"))
    mode_name = "searchr1" if args.use_searchr1 else "retriever"
    output_filename = f"{output_stem}_{args.evidence_setting}_{mode_name}_{slurm_job_id}.jsonl"
    output_dir = args.output_dir or os.path.join("results", args.dataset, "vanilla")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # resume: 이미 저장된 uid 로드
    done_uids: set = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = obj.get("uid") or obj.get("question", "")
                    if key:
                        done_uids.add(key)
                except Exception:
                    pass
        logger.info(f"Resuming: {len(done_uids)} samples already done")

    result_list = []
    with open(output_path, "a") as out_f:
        for sample in tqdm(input_list):
            uid = sample.get("uid") or sample.get("id") or sample.get("question", "")
            if uid in done_uids:
                continue
            try:
                result = process_sample(sample, direct_model, args)
            except Exception as exc:
                logger.error(f"Failed to process sample {sample.get('index')}: {exc}")
                result = {
                    "index": sample.get("index"),
                    "uid": sample.get("uid") or sample.get("id"),
                    "question": sample.get("question"),
                    "answer": sample.get("answer"),
                    "error": str(exc),
                }
            result_list.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    logger.info(f"Results saved to: {output_path} ({len(result_list)} new samples)")


if __name__ == "__main__":
    main()
