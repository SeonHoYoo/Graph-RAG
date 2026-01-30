"""
CoT Reasoning Path와 Retrieved Documents의 Triplet Graph 비교 스크립트

사용법:
    python compare_graphs.py \
        --dataset musique \
        --input_filename train_sampled.json \
        --construct_model_name Qwen/Qwen2.5-72B-Instruct \
        --retriever_url http://127.0.0.1:8000/retrieve \
        --bm25_top_k 5
"""

import argparse
import json
import logging
import os
from typing import *
from tqdm import tqdm

from model_library.construct_model import ConstructModel
from utils.graph import Graph
from direct import Direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name (musique, hotpotqa, 2wikimultihopqa)"
    )
    parser.add_argument("--input_filename", type=str, required=True,
        help="Input JSON filename (e.g., train_sampled.json)"
    )
    parser.add_argument("--construct_model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct",
        help="Model name for triplet extraction (GPT/Claude/Qwen). Default: Qwen/Qwen2.5-72B-Instruct"
    )
    parser.add_argument("--api_key", type=str, default=None,
        help="API key for OpenAI or Anthropic (not needed for Qwen)"
    )
    parser.add_argument("--construct_batch_size", type=int, default=1,
        help="Batch size for construction (for batch API)"
    )
    parser.add_argument("--force_cot_regen", action="store_true",
        help="Force regeneration of CoT reasoning/triplets even if present"
    )
    parser.add_argument("--cot_retry", type=int, default=1,
        help="Number of retries when CoT comes back empty (generation is attempted at most 1 + cot_retry times)"
    )
    parser.add_argument("--max_samples", type=int, default=None,
        help="Limit number of samples to process (for quick/debug runs)"
    )
    parser.add_argument("--output_filename", type=str, default=None,
        help="Output JSON filename (default: compare_graphs_{input_filename})"
    )
    parser.add_argument("--generate_cot", action="store_true",
        help="Generate CoT reasoning if not present in input data"
    )
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve",
        help="URL for BM25 retriever server (default: http://127.0.0.1:8000/retrieve, same node)"
    )
    parser.add_argument("--bm25_top_k", type=int, default=5,
        help="Number of top documents to retrieve using BM25"
    )
    parser.add_argument("--setting", type=str, default="open-book",
        choices=["open-book", "open-book+gold", "gold"],
        help="Retrieval setting mode"
    )
    parser.add_argument("--retrieval_strategy", type=str, default="cot_reasoning",
        choices=["question", "cot_reasoning", "triplets", "combined", "multihop_triplets", "question_triplets"],
        help="Retrieval strategy: question (question only), cot_reasoning (use CoT reasoning as thinking), triplets (use CoT triplets as query), combined (use both question and triplets), multihop_triplets (search for each triplet separately), question_triplets (use GraphCheck-style question triplets as query)"
    )
    parser.add_argument("--use_searchr1", action="store_true",
        help="Use SearchR1 model for retrieval (enables thinking parameter)"
    )
    parser.add_argument("--nudge_searchr1", action="store_true",
        help="Use nudge model with SearchR1 (requires use_searchr1)"
    )
    parser.add_argument("--multihop_top_k_per_triplet", type=int, default=2,
        help="Number of documents to retrieve per triplet in multihop_triplets strategy"
    )
    parser.add_argument("--compare_question_graph", action="store_true",
        help="Also build a graph directly from the question and compare with doc/gold graphs"
    )
    
    return parser.parse_args()


def process_sample(
    sample: Dict[str, Any],
    construct_model: ConstructModel,
    direct_model: Direct,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    각 샘플에 대해 CoT reasoning과 retrieved documents에서 triplet graph를 추출하고 비교합니다.
    
    입력 샘플 구조 (train_sampled.json):
    {
        "index": int,
        "question": str,
        "answer": str,
        "gold_id_list": List[str],
        "gold_evidence_list": List[str],
        ...
    }
    """
    try:
        question = sample.get("question", "")
        if not question:
            logger.warning(f"Sample {sample.get('index')} has no question. Skipping.")
            return sample
        
        # 질문 자체로 만든 그래프 (GraphCheck 기본 방식) 준비
        question_def_triples = sample.get("definition_triples", [])
        question_triples = sample.get("triples", [])
        question_graph = None
        need_question_graph = args.compare_question_graph or args.retrieval_strategy == "question_triplets"
        if need_question_graph:
            try:
                if not question_def_triples and not question_triples:
                    question_sample = {"question": question}
                    question_sample = construct_model.process_sample(question_sample)
                    question_def_triples = question_sample.get("definition_triples", [])
                    question_triples = question_sample.get("triples", [])
                question_graph = Graph(question_def_triples, question_triples)
            except Exception as e:
                logger.warning(f"Sample {sample.get('index')}: failed to build question graph: {e}")
        
        # 1. CoT reasoning 가져오기 또는 생성
        cot_reasoning = sample.get("cot_reasoning", "")
        cot_def_triples = sample.get("cot_def_triples", [])
        cot_triples = sample.get("cot_triples", [])
        
        def regenerate_cot(max_attempts: int) -> Tuple[str, List[Any], List[Any]]:
            regen_reasoning, regen_def, regen_triples = "", [], []
            for attempt in range(max_attempts):
                if attempt > 0:
                    logger.info(f"Sample {sample.get('index')}: retry CoT generation ({attempt}/{max_attempts - 1})")
                regen_reasoning, regen_def, regen_triples = construct_model.generate_cot_reasoning_with_triplets(question)
                # reasoning과 triplets가 모두 있으면 성공으로 간주
                if regen_reasoning and regen_reasoning.strip() and regen_triples:
                    logger.info(f"Sample {sample.get('index')}: Generated reasoning ({len(regen_reasoning)} chars) and {len(regen_triples)} triplets")
                    break
                # triplets만 있어도 일단 사용 (reasoning은 나중에 재시도 가능)
                elif regen_triples:
                    logger.info(f"Sample {sample.get('index')}: Generated {len(regen_triples)} triplets but no reasoning (attempt {attempt + 1}/{max_attempts})")
                    if attempt == max_attempts - 1:  # 마지막 시도면 그냥 사용
                        break
            return regen_reasoning, regen_def, regen_triples
        
        need_regen = args.force_cot_regen or (not cot_triples)
        if need_regen:
            logger.info(f"Sample {sample.get('index')} has no CoT triplets or force regen is set. Generating...")
            cot_reasoning, cot_def_triples, cot_triples = regenerate_cot(args.cot_retry + 1)
            sample["cot_reasoning"] = cot_reasoning
            sample["cot_def_triples"] = cot_def_triples
            sample["cot_triples"] = cot_triples
            if not cot_triples:
                logger.warning(f"Sample {sample.get('index')} CoT generation yielded no triplets after retries.")
            elif not cot_reasoning or not cot_reasoning.strip():
                logger.warning(f"Sample {sample.get('index')} CoT generation yielded triplets but no reasoning.")
        
        if cot_reasoning and not cot_triples and not need_regen:
            # CoT reasoning은 있지만 triplet이 없으면 추출
            logger.info(f"Sample {sample.get('index')} has CoT reasoning but no triplets. Extracting...")
            cot_def_triples, cot_triples = construct_model.extract_triplets_from_cot_reasoning(cot_reasoning)
            sample["cot_def_triples"] = cot_def_triples
            sample["cot_triples"] = cot_triples
            if (not cot_triples) and args.force_cot_regen:
                logger.info(f"Sample {sample.get('index')} extraction empty; regenerating CoT due to force flag.")
                cot_reasoning, cot_def_triples, cot_triples = regenerate_cot(args.cot_retry + 1)
                sample["cot_reasoning"] = cot_reasoning
                sample["cot_def_triples"] = cot_def_triples
                sample["cot_triples"] = cot_triples
        
        cot_graph = Graph(cot_def_triples, cot_triples)
        
        # 2. Retrieved documents 검색 (없으면 자동 검색)
        retrieved_docs = sample.get("retrieved_documents", [])
        if not retrieved_docs:
            # 자동으로 검색 수행
            logger.info(f"Sample {sample.get('index')} has no retrieved documents. Retrieving...")
            gold_id_list = sample.get("gold_id_list", [])
            gold_evidence_list = sample.get("gold_evidence_list", [])
            
            # 검색 전략에 따라 쿼리와 thinking 설정
            retrieval_query = question
            thinking = None
            
            if args.retrieval_strategy == "cot_reasoning":
                # CoT reasoning을 thinking으로 사용
                if cot_reasoning and cot_reasoning.strip():
                    thinking = cot_reasoning
                    logger.info(f"Sample {sample.get('index')}: Using CoT reasoning for retrieval")
                else:
                    logger.warning(f"Sample {sample.get('index')}: CoT reasoning is empty, falling back to question")
            
            elif args.retrieval_strategy == "triplets":
                # Triplets를 검색 쿼리로 사용
                all_triples = cot_def_triples + cot_triples
                if all_triples:
                    # Triplets를 자연어로 변환하여 쿼리 생성
                    triplets_text = "\n".join(all_triples)
                    retrieval_query = f"{question}\n\nTriplets:\n{triplets_text}"
                    logger.info(f"Sample {sample.get('index')}: Using triplets for retrieval")
                else:
                    logger.warning(f"Sample {sample.get('index')}: No triplets available, falling back to question")
            
            elif args.retrieval_strategy == "combined":
                # Question과 triplets를 모두 사용
                all_triples = cot_def_triples + cot_triples
                if all_triples:
                    triplets_text = "\n".join(all_triples)
                    retrieval_query = f"{question}\n\nTriplets:\n{triplets_text}"
                if cot_reasoning and cot_reasoning.strip():
                    thinking = cot_reasoning
                logger.info(f"Sample {sample.get('index')}: Using combined strategy (question + triplets + CoT reasoning)")
            
            elif args.retrieval_strategy == "question_triplets":
                # 질문 그래프에서 뽑은 triplets를 검색 쿼리로 사용
                all_triples = question_def_triples + question_triples
                if all_triples:
                    triplets_text = "\n".join(all_triples)
                    retrieval_query = f"{question}\n\nQuestion Triplets:\n{triplets_text}"
                    logger.info(f"Sample {sample.get('index')}: Using question triplets for retrieval")
                else:
                    logger.warning(f"Sample {sample.get('index')}: Question triplets unavailable, falling back to question")
            
            elif args.retrieval_strategy == "multihop_triplets":
                # 각 triplet별로 멀티홉 검색 수행
                all_triples = cot_def_triples + cot_triples
                if not all_triples:
                    logger.warning(f"Sample {sample.get('index')}: No triplets available for multihop search, falling back to question")
                    retrieval_query = question
                else:
                    # 멀티홉 검색: 각 triplet별로 검색
                    all_evidence_list = []
                    all_doc_ids = set()
                    retrieval_turns = []
                    
                    logger.info(f"Sample {sample.get('index')}: Starting multihop retrieval for {len(all_triples)} triplets")
                    
                    for hop_idx, triple in enumerate(all_triples):
                        # Triple을 자연어 쿼리로 변환
                        # "(ENT1) [SEP] is [SEP] a character" -> "a character"
                        # "Adams Township [SEP] is located in [SEP] (ENT1)" -> "Adams Township is located in"
                        if "[SEP]" in triple:
                            parts = triple.split(" [SEP] ")
                            if len(parts) >= 3:
                                subject = parts[0].replace("(ENT", "").replace(")", "").strip()
                                relation = parts[1].strip()
                                obj = parts[2].split("[PREP]")[0].strip() if "[PREP]" in parts[2] else parts[2].strip()
                                
                                # ENT placeholder 제거하고 자연어로 변환
                                if subject.startswith("ENT") or subject.startswith("("):
                                    query_text = f"{relation} {obj}"
                                elif obj.startswith("ENT") or obj.startswith("("):
                                    query_text = f"{subject} {relation}"
                                else:
                                    query_text = f"{subject} {relation} {obj}"
                            else:
                                query_text = triple.replace("[SEP]", " ").replace("(ENT", "").replace(")", "")
                        else:
                            query_text = triple
                        
                        hop_query = f"{question}\n\nSearching for: {query_text}"
                        if cot_reasoning and cot_reasoning.strip():
                            thinking = cot_reasoning
                        
                        hop_evidence, hop_retrieval_info = direct_model.retrieve_evidence(
                            hop_query,
                            gold_id_list,
                            gold_evidence_list,
                            top_k=args.multihop_top_k_per_triplet,
                            use_searchr1=args.use_searchr1,
                            nudge_searchr1=args.nudge_searchr1,
                            thinking=thinking
                        )
                        
                        # 중복 제거하면서 추가
                        for doc in hop_evidence:
                            if doc not in all_evidence_list:
                                all_evidence_list.append(doc)
                        
                        retrieval_turns.append({
                            "hop": hop_idx + 1,
                            "triplet": triple,
                            "query": hop_query,
                            "num_docs": len(hop_evidence),
                            "doc_ids": hop_retrieval_info.get("doc_id_list", [])
                        })
                    
                    retrieved_docs = all_evidence_list
                    retrieval_info = {
                        "query": question,
                        "strategy": args.retrieval_strategy,
                        "doc_id_list": list(set([doc.split(")")[0].replace("(Title: ", "") for doc in all_evidence_list if "(Title:" in doc])),
                        "is_gold_list": [1 if doc_id in gold_id_list else 0 for doc_id in list(set([doc.split(")")[0].replace("(Title: ", "") for doc in all_evidence_list if "(Title:" in doc]))],
                        "retrieval_turns": retrieval_turns,
                        "total_hops": len(all_triples),
                        "total_docs": len(retrieved_docs)
                    }
                    sample["retrieved_documents"] = retrieved_docs
                    sample["retrieval_info"] = retrieval_info
                    logger.info(f"Sample {sample.get('index')}: Multihop retrieval completed - {len(retrieved_docs)} unique documents from {len(all_triples)} hops")
            
            # 단일 검색 전략인 경우
            if args.retrieval_strategy != "multihop_triplets":
                evidence_list, retrieval_info = direct_model.retrieve_evidence(
                    retrieval_query,
                    gold_id_list,
                    gold_evidence_list,
                    top_k=direct_model.bm25_top_k,
                    use_searchr1=args.use_searchr1,
                    nudge_searchr1=args.nudge_searchr1,
                    thinking=thinking
                )
                retrieved_docs = evidence_list
                sample["retrieved_documents"] = retrieved_docs
                sample["retrieval_info"] = retrieval_info
                sample["retrieval_info"]["strategy"] = args.retrieval_strategy
        
        doc_triples_list = []
        doc_def_triples_list = []
        
        for doc_idx, doc in enumerate(retrieved_docs):
            try:
                doc_def_triples, doc_triples = construct_model.extract_triplets_from_document(doc)
                doc_triples_list.extend(doc_triples)
                doc_def_triples_list.extend(doc_def_triples)
            except Exception as e:
                logger.warning(f"Failed to extract triplets from document {doc_idx} in sample {sample.get('index')}: {e}")
        
        # 모든 documents의 triplets를 하나의 그래프로 합침
        doc_graph = Graph(doc_def_triples_list, doc_triples_list)
        
        comparison_question_doc = None
        comparison_question_gold = None
        if question_graph is not None:
            comparison_question_doc = question_graph.compare_with(doc_graph)
        
        # 3. 두 그래프 비교
        comparison_result = cot_graph.compare_with(doc_graph)
        
        # 3-1. Gold evidence 그래프 비교 (있을 경우)
        gold_graph = None
        comparison_gold = None
        comparison_gold_vs_doc = None
        gold_evidence_list = sample.get("gold_evidence_list", [])
        if gold_evidence_list:
            gold_def_triples_list: List[Any] = []
            gold_triples_list: List[Any] = []
            for gold_idx, gold_doc in enumerate(gold_evidence_list):
                try:
                    gold_def_triples, gold_triples = construct_model.extract_triplets_from_document(gold_doc)
                    gold_triples_list.extend(gold_triples)
                    gold_def_triples_list.extend(gold_def_triples)
                except Exception as e:
                    logger.warning(f"Failed to extract triplets from gold document {gold_idx} in sample {sample.get('index')}: {e}")
            gold_graph = Graph(gold_def_triples_list, gold_triples_list)
            comparison_gold = cot_graph.compare_with(gold_graph)
            # Gold triplets vs Doc triplets 비교 추가
            comparison_gold_vs_doc = gold_graph.compare_with(doc_graph)
            if question_graph is not None:
                comparison_question_gold = question_graph.compare_with(gold_graph)
        
        # 4. 결과 저장
        sample.update({
            "cot_graph": {
                "definition_triples": cot_def_triples,
                "triples": cot_triples,
                "num_triplets": len(cot_graph.total_triples)
            },
            "document_graph": {
                "definition_triples": doc_def_triples_list,
                "triples": doc_triples_list,
                "num_triplets": len(doc_graph.total_triples)
            },
            "comparison": comparison_result
        })
        
        if question_graph is not None:
            sample.update({
                "question_graph": {
                    "definition_triples": question_def_triples,
                    "triples": question_triples,
                    "num_triplets": len(question_graph.total_triples)
                },
                "comparison_question_vs_doc": comparison_question_doc
            })
        
        if gold_graph is not None:
            update_dict = {
                "gold_graph": {
                    "definition_triples": gold_def_triples_list,
                    "triples": gold_triples_list,
                    "num_triplets": len(gold_graph.total_triples)
                },
                "comparison_gold": comparison_gold
            }
            if comparison_gold_vs_doc is not None:
                update_dict["comparison_gold_vs_doc"] = comparison_gold_vs_doc
            if comparison_question_gold is not None:
                update_dict["comparison_question_vs_gold"] = comparison_question_gold
            sample.update(update_dict)
        
        logger.info(
            f"Sample {sample.get('index')}: "
            f"CoT triplets={len(cot_graph.total_triples)}, "
            f"Doc triplets={len(doc_graph.total_triples)}, "
            f"Overlap={comparison_result['triplet_overlap']}, "
            f"F1={comparison_result['triplet_f1']:.3f}"
        )
        
        if comparison_gold:
            logger.info(
                f"Sample {sample.get('index')} (CoT vs Gold): "
                f"Gold triplets={len(gold_graph.total_triples)}, "
                f"Overlap={comparison_gold['triplet_overlap']}, "
                f"F1={comparison_gold['triplet_f1']:.3f}"
            )
        if comparison_gold_vs_doc:
            logger.info(
                f"Sample {sample.get('index')} (Gold vs Doc): "
                f"Gold triplets={len(gold_graph.total_triples)}, "
                f"Doc triplets={len(doc_graph.total_triples)}, "
                f"Overlap={comparison_gold_vs_doc['triplet_overlap']}, "
                f"F1={comparison_gold_vs_doc['triplet_f1']:.3f}"
            )
        if comparison_question_doc:
            logger.info(
                f"Sample {sample.get('index')} (Question vs Doc): "
                f"Question triplets={len(question_graph.total_triples)}, "
                f"Doc triplets={len(doc_graph.total_triples)}, "
                f"Overlap={comparison_question_doc['triplet_overlap']}, "
                f"F1={comparison_question_doc['triplet_f1']:.3f}"
            )
        if comparison_question_gold:
            logger.info(
                f"Sample {sample.get('index')} (Question vs Gold): "
                f"Question triplets={len(question_graph.total_triples)}, "
                f"Gold triplets={len(gold_graph.total_triples)}, "
                f"Overlap={comparison_question_gold['triplet_overlap']}, "
                f"F1={comparison_question_gold['triplet_f1']:.3f}"
            )
        
    except Exception as e:
        logger.error(f"Failed to process sample {sample.get('index')}: {e}")
        sample.update({
            "cot_graph": None,
            "document_graph": None,
            "comparison": None,
            "error": str(e)
        })
    
    return sample


def main():
    args = parse_args()
    
    # 입력 파일 경로
    input_path = os.path.join("datasets", args.dataset, "claims", args.input_filename)
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    
    # 출력 파일 경로
    if args.output_filename:
        output_filename = args.output_filename
    else:
        output_filename = f"compare_graphs_{args.input_filename}"
    
    output_path = os.path.join(
        "results", args.dataset, "graph_comparison", 
        args.construct_model_name.split("/")[-1], output_filename
    )
    
    # ConstructModel 초기화
    construct_model = ConstructModel(
        construct_model_name=args.construct_model_name,
        dataset_name=args.dataset,
        api_key=args.api_key,
        batch_size=args.construct_batch_size
    )
    
    # Direct 모델 초기화 (검색용)
    from argparse import Namespace
    direct_args = Namespace(
        dataset=args.dataset,
        input_filename=args.input_filename,
        direct_filename=None,
        base_model_name="google/flan-t5-xl",  # 검색만 하므로 모델 로드 안 함
        setting=args.setting,
        bm25_top_k=args.bm25_top_k,
        use_searchr1=args.use_searchr1,
        searchr1_top_k=3,
        searchr1_max_turns=3,
        use_total_search_results=False,
        retriever_url=args.retriever_url
    )
    direct_model = Direct(direct_args)
    
    # 입력 데이터 로드
    with open(input_path, "r") as f:
        input_list = json.load(f)
    
    if args.max_samples is not None:
        input_list = input_list[:args.max_samples]
    
    logger.info(f"Processing {len(input_list)} samples...")
    logger.info(f"Using model: {args.construct_model_name}")
    logger.info(f"Retriever URL: {args.retriever_url}")
    logger.info(f"BM25 top_k: {args.bm25_top_k}")
    
    # 각 샘플 처리
    result_list = []
    for sample in tqdm(input_list):
        result = process_sample(sample, construct_model, direct_model, args)
        result_list.append(result)
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_list, f, indent=4)
    
    logger.info(f"Results saved to: {output_path}")
    
    # 전체 통계 계산
    valid_results = [r for r in result_list if r.get("comparison") is not None]
    valid_gold_results = [r for r in result_list if r.get("comparison_gold") is not None]
    if valid_results:
        avg_f1 = sum(r["comparison"]["triplet_f1"] for r in valid_results) / len(valid_results)
        avg_precision = sum(r["comparison"]["triplet_precision"] for r in valid_results) / len(valid_results)
        avg_recall = sum(r["comparison"]["triplet_recall"] for r in valid_results) / len(valid_results)
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Total samples: {len(result_list)}")
        logger.info(f"Valid samples: {len(valid_results)}")
        logger.info(f"Average Precision: {avg_precision:.3f}")
        logger.info(f"Average Recall: {avg_recall:.3f}")
        logger.info(f"Average F1: {avg_f1:.3f}")
        
        if valid_gold_results:
            avg_f1_gold = sum(r["comparison_gold"]["triplet_f1"] for r in valid_gold_results) / len(valid_gold_results)
            avg_precision_gold = sum(r["comparison_gold"]["triplet_precision"] for r in valid_gold_results) / len(valid_gold_results)
            avg_recall_gold = sum(r["comparison_gold"]["triplet_recall"] for r in valid_gold_results) / len(valid_gold_results)
            
            logger.info(f"\n=== CoT vs Gold Summary ===")
            logger.info(f"Gold valid samples: {len(valid_gold_results)}")
            logger.info(f"Average Precision: {avg_precision_gold:.3f}")
            logger.info(f"Average Recall: {avg_recall_gold:.3f}")
            logger.info(f"Average F1: {avg_f1_gold:.3f}")
            
            # Gold vs Doc 비교 통계
            valid_gold_vs_doc_results = [r for r in result_list if r.get("comparison_gold_vs_doc") is not None]
            if valid_gold_vs_doc_results:
                avg_f1_gold_vs_doc = sum(r["comparison_gold_vs_doc"]["triplet_f1"] for r in valid_gold_vs_doc_results) / len(valid_gold_vs_doc_results)
                avg_precision_gold_vs_doc = sum(r["comparison_gold_vs_doc"]["triplet_precision"] for r in valid_gold_vs_doc_results) / len(valid_gold_vs_doc_results)
                avg_recall_gold_vs_doc = sum(r["comparison_gold_vs_doc"]["triplet_recall"] for r in valid_gold_vs_doc_results) / len(valid_gold_vs_doc_results)
                
                logger.info(f"\n=== Gold vs Doc Summary ===")
                logger.info(f"Gold vs Doc valid samples: {len(valid_gold_vs_doc_results)}")
                logger.info(f"Average Precision: {avg_precision_gold_vs_doc:.3f}")
                logger.info(f"Average Recall: {avg_recall_gold_vs_doc:.3f}")
                logger.info(f"Average F1: {avg_f1_gold_vs_doc:.3f}")


if __name__ == "__main__":
    main()

