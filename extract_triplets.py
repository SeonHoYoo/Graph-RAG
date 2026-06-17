"""
Question / Retrieved Doc / Gold Evidence에서 triplet만 추출하는 스크립트.

사용법:
    python extract_triplets.py \
        --dataset musique \
        --input_filename train_sampled.json \
        --construct_model_name Qwen/Qwen2.5-7B-Instruct \
        --bm25_top_k 5 \
        --setting open-book+gold
"""

import argparse
import json
import logging
import os
import time
from typing import *
from tqdm import tqdm

from model_library.construct_model import ConstructModel
from direct import Direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name (musique, hotpotqa, 2wikimultihopqa)")
    parser.add_argument("--input_filename", type=str, required=True,
        help="Input JSON filename (e.g., train_sampled.json)")
    parser.add_argument("--construct_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for triplet extraction")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--setting", type=str, default="open-book+gold",
        choices=["open-book", "open-book+gold", "gold"],
        help="Retrieval setting: open-book, open-book+gold, gold")
    parser.add_argument("--bm25_top_k", type=int, default=5,
        help="Number of top documents to retrieve using BM25")
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--max_samples", type=int, default=None,
        help="Limit number of samples (for debug runs)")
    parser.add_argument("--output_filename", type=str, default=None,
        help="Output JSON filename (default: triplets_{input_filename})")
    parser.add_argument("--checkpoint_every", type=int, default=10,
        help="Save checkpoint every N samples")
    return parser.parse_args()


def extract_question_triplets(
    sample: Dict[str, Any],
    construct_model: ConstructModel,
) -> Dict[str, Any]:
    """question에서 triplet 추출 (2-step: latent entity detection → triplet extraction)"""
    question = sample.get("question", "")
    question_sample = {"question": question}
    question_sample = construct_model.process_sample(question_sample)
    def_triples = question_sample.get("definition_triples", [])
    triples = question_sample.get("triples", [])
    triples = construct_model.normalize_casting_triples(question, triples)
    return {"definition_triples": def_triples, "triples": triples}


def extract_doc_triplets(
    documents: List[str],
    construct_model: ConstructModel,
) -> Dict[str, Any]:
    """검색된 문서 리스트에서 triplet 추출"""
    all_def_triples: List[str] = []
    all_triples: List[str] = []
    per_doc: List[Dict[str, Any]] = []

    for doc in documents:
        try:
            def_t, t = construct_model.extract_triplets_from_document(doc)
            all_def_triples.extend(def_t)
            all_triples.extend(t)
            per_doc.append({"document": doc, "definition_triples": def_t, "triples": t})
        except Exception as e:
            logger.warning(f"Doc triplet extraction failed: {e}")
            per_doc.append({"document": doc, "definition_triples": [], "triples": [], "error": str(e)})

    return {
        "definition_triples": all_def_triples,
        "triples": all_triples,
        "per_document": per_doc,
    }


def process_sample(
    sample: Dict[str, Any],
    construct_model: ConstructModel,
    direct_model: Direct,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question = sample.get("question", "")
    if not question:
        logger.warning(f"Sample {sample.get('index')} has no question. Skipping.")
        return None

    gold_id_list = sample.get("gold_id_list", [])
    gold_evidence_list = sample.get("gold_evidence_list", [])

    # ── 1) Question triplets ──
    logger.info(f"[{sample.get('index')}] Extracting question triplets ...")
    question_graph = extract_question_triplets(sample, construct_model)

    # ── 2) Retrieved doc triplets ──
    logger.info(f"[{sample.get('index')}] Retrieving documents & extracting doc triplets ...")
    evidence_list, retrieval_info = direct_model.retrieve_evidence(
        question, gold_id_list, gold_evidence_list, top_k=direct_model.bm25_top_k,
    )
    doc_graph = extract_doc_triplets(evidence_list, construct_model)

    # ── 3) Gold evidence triplets (optional) ──
    if "gold" in args.setting:
        logger.info(f"[{sample.get('index')}] Extracting gold triplets ...")
        gold_docs = [f"(Title: {did}) {txt}" for did, txt in zip(gold_id_list, gold_evidence_list)]
        gold_graph = extract_doc_triplets(gold_docs, construct_model)
    else:
        gold_graph = {
            "definition_triples": [],
            "triples": [],
            "per_document": [],
        }

    # ── meta ──
    meta: Dict[str, Any] = {
        "index": sample.get("index"),
        "uid": sample.get("uid"),
        "num_hops": sample.get("num_hops"),
        "question": question,
        "answer": sample.get("answer"),
        "answer_aliases": sample.get("answer_aliases", []),
        "gold_id_list": gold_id_list,
    }
    if "level" in sample:
        meta["level"] = sample["level"]

    result = {
        **meta,
        "question_graph": question_graph,
        "doc_graph": {
            "retrieval_info": {
                "doc_id_list": retrieval_info.get("doc_id_list", []),
                "is_gold_list": retrieval_info.get("is_gold_list", []),
            },
            "definition_triples": doc_graph["definition_triples"],
            "triples": doc_graph["triples"],
            "per_document": doc_graph["per_document"],
        },
        "gold_graph": {
            "definition_triples": gold_graph["definition_triples"],
            "triples": gold_graph["triples"],
            "per_document": gold_graph["per_document"],
        },
    }

    logger.info(
        f"[{sample.get('index')}] "
        f"question_triples={len(question_graph['triples'])}, "
        f"doc_triples={len(doc_graph['triples'])}, "
        f"gold_triples={len(gold_graph['triples'])}"
    )
    return result


def main():
    args = parse_args()

    input_path = os.path.join("datasets", args.dataset, "claims", args.input_filename)
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    if args.output_filename:
        output_filename = args.output_filename
    else:
        output_filename = f"triplets_{args.input_filename}"

    model_short = args.construct_model_name.split("/")[-1]
    output_path = os.path.join("results", args.dataset, "triplets", model_short, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── 모델 초기화 ──
    logger.info(f"Loading construct model: {args.construct_model_name}")
    construct_model = ConstructModel(
        construct_model_name=args.construct_model_name,
        dataset_name=args.dataset,
        api_key=args.api_key,
    )

    from argparse import Namespace
    direct_args = Namespace(
        dataset=args.dataset,
        input_filename=args.input_filename,
        direct_filename=None,
        base_model_name="google/flan-t5-xl",
        setting=args.setting,
        bm25_top_k=args.bm25_top_k,
        use_searchr1=False,
        searchr1_top_k=3,
        searchr1_max_turns=3,
        use_total_search_results=False,
        retriever_url=args.retriever_url,
    )
    direct_model = Direct(direct_args)

    with open(input_path, "r") as f:
        input_list = json.load(f)
    logger.info(f"Loaded {len(input_list)} samples from {input_path}")

    if args.max_samples is not None:
        input_list = input_list[: args.max_samples]
        logger.info(f"Limiting to {len(input_list)} samples")

    # ── 기존 checkpoint 로드 ──
    result_list: List[Dict[str, Any]] = []
    done_indices: Set[int] = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            result_list = json.load(f)
        done_indices = {r["index"] for r in result_list}
        logger.info(f"Loaded checkpoint with {len(done_indices)} already-processed samples")

    t0 = time.time()
    new_count = 0
    for sample in tqdm(input_list, desc=f"{args.dataset}"):
        if sample.get("index") in done_indices:
            continue
        try:
            result = process_sample(sample, construct_model, direct_model, args)
            if result is not None:
                result_list.append(result)
                new_count += 1
        except Exception as e:
            logger.error(f"Sample {sample.get('index')} failed: {e}")
            continue

        if new_count % args.checkpoint_every == 0 and new_count > 0:
            result_list.sort(key=lambda x: x["index"])
            with open(output_path, "w") as f:
                json.dump(result_list, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved ({len(result_list)} samples) → {output_path}")

    result_list.sort(key=lambda x: x["index"])
    with open(output_path, "w") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    logger.info(f"Done. {len(result_list)} samples saved → {output_path}  ({elapsed:.1f}s)")

    # ── 간단 통계 ──
    q_counts = [len(r["question_graph"]["triples"]) for r in result_list]
    d_counts = [len(r["doc_graph"]["triples"]) for r in result_list]
    g_counts = [len(r["gold_graph"]["triples"]) for r in result_list]
    logger.info(
        f"Avg triplets — question: {sum(q_counts)/max(len(q_counts),1):.1f}, "
        f"doc: {sum(d_counts)/max(len(d_counts),1):.1f}, "
        f"gold: {sum(g_counts)/max(len(g_counts),1):.1f}"
    )


if __name__ == "__main__":
    main()
