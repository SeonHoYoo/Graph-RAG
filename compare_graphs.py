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
import re
from typing import *
from tqdm import tqdm

from model_library.construct_model import ConstructModel
from utils.graph import Graph, search_query_graph_bindings, ensemble_triplet_matching
from direct import Direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)


def _normalize_rel_key(text: str) -> str:
    if not text:
        return ""
    normalized = text.lower().strip()
    normalized = re.sub(r"[\"'`]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _load_softmatch_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = item.get("key")
            result = item.get("result")
            if key and isinstance(result, dict):
                cache[key] = result
    return cache


def _append_softmatch_cache(path: str, key: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"key": key, "result": result}) + "\n")


def _parse_triple(triple_sent: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    if "[SEP]" not in triple_sent:
        return None
    parts = triple_sent.split(" [SEP] ")
    if len(parts) < 3:
        return None
    head = parts[0].strip()
    rel = parts[1].strip()
    tail = " [SEP] ".join(parts[2:]).strip()
    context = None
    if " [PREP] " in tail:
        tail, context = tail.split(" [PREP] ", 1)
        tail = tail.strip()
        context = context.strip()
    return head, rel, tail, context


def _token_set(text: str) -> Set[str]:
    if not text:
        return set()
    normalized = text.lower()
    normalized = re.sub(r"[\"'`]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = normalized.split()
    return set(t for t in tokens if len(t) >= 3)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_placeholder(entity: str) -> bool:
    normalized = entity.lower()
    return re.search(r"\bent\d+\b", normalized) is not None


def _soft_match_triplets(
    query_triples: List[str],
    fact_triples: List[str],
    construct_model: ConstructModel,
    cache: Dict[str, Dict[str, Any]],
    cache_path: str,
    threshold: float,
    topn: int,
) -> Dict[str, Any]:
    llm_calls = 0
    cache_hits = 0

    query_items = []
    for triple in query_triples:
        parsed = _parse_triple(triple)
        if parsed:
            query_items.append((triple, *parsed))

    fact_items = []
    for triple in fact_triples:
        parsed = _parse_triple(triple)
        if parsed:
            fact_items.append((triple, *parsed))

    if not query_items or not fact_items:
        return {
            "matched_triplets": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "overlap_pairs_topk": [],
            "llm_calls": 0,
            "cache_hits": 0,
        }

    candidate_pairs = []
    for qi, (q_raw, q_head, q_rel, q_tail, q_ctx) in enumerate(query_items):
        q_head_tokens = _token_set(q_head)
        q_tail_tokens = _token_set(q_tail)
        q_rel_tokens = _token_set(q_rel)

        scored = []
        for fi, (f_raw, f_head, f_rel, f_tail, f_ctx) in enumerate(fact_items):
            head_sim = _jaccard(q_head_tokens, _token_set(f_head)) if not _is_placeholder(q_head) else 1.0
            tail_sim = _jaccard(q_tail_tokens, _token_set(f_tail)) if not _is_placeholder(q_tail) else 1.0
            rel_sim = _jaccard(q_rel_tokens, _token_set(f_rel))
            cheap_score = (head_sim + tail_sim + rel_sim) / 3
            if cheap_score == 0.0:
                continue
            scored.append((cheap_score, qi, fi))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidate_pairs.extend(scored[:topn])

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    rel_cache_key = {}
    for _score, qi, fi in candidate_pairs:
        q_rel = query_items[qi][2]
        f_rel = fact_items[fi][2]
        key = f"{_normalize_rel_key(q_rel)}|||{_normalize_rel_key(f_rel)}"
        if key in rel_cache_key:
            continue
        if key in cache:
            cache_hits += 1
            rel_cache_key[key] = cache[key]
            continue
        prompt = (
            "Decide if two relations are equivalent or inverse. "
            "Return JSON only with fields: equivalent, inverse, similarity, "
            "normalized_rel_a, normalized_rel_b, notes. "
            "Ignore [PREP] context details for similarity.\n\n"
            f"rel_a: {q_rel}\n"
            f"rel_b: {f_rel}\n"
        )
        raw = construct_model.construct_model.generate(prompt)
        parsed = _extract_json(raw) or {
            "equivalent": False,
            "inverse": False,
            "similarity": 0.0,
            "normalized_rel_a": _normalize_rel_key(q_rel),
            "normalized_rel_b": _normalize_rel_key(f_rel),
            "notes": "parse_failed",
        }
        rel_cache_key[key] = parsed
        cache[key] = parsed
        _append_softmatch_cache(cache_path, key, parsed)
        llm_calls += 1

    used_query = set()
    used_fact = set()
    overlap_pairs = []

    for _score, qi, fi in candidate_pairs:
        if qi in used_query or fi in used_fact:
            continue
        q_raw, q_head, q_rel, q_tail, _q_ctx = query_items[qi]
        f_raw, f_head, f_rel, f_tail, _f_ctx = fact_items[fi]

        key = f"{_normalize_rel_key(q_rel)}|||{_normalize_rel_key(f_rel)}"
        rel_info = rel_cache_key.get(key)
        if not rel_info:
            continue
        rel_sim = float(rel_info.get("similarity", 0.0))
        inverse = bool(rel_info.get("inverse", False))
        if not rel_info.get("equivalent", False) and rel_sim < threshold:
            continue

        if inverse:
            subj_sim = _jaccard(_token_set(q_head), _token_set(f_tail)) if not _is_placeholder(q_head) else 1.0
            obj_sim = _jaccard(_token_set(q_tail), _token_set(f_head)) if not _is_placeholder(q_tail) else 1.0
        else:
            subj_sim = _jaccard(_token_set(q_head), _token_set(f_head)) if not _is_placeholder(q_head) else 1.0
            obj_sim = _jaccard(_token_set(q_tail), _token_set(f_tail)) if not _is_placeholder(q_tail) else 1.0

        entity_sim = (subj_sim + obj_sim) / 2
        match_score = rel_sim * 0.6 + entity_sim * 0.4
        if match_score < threshold:
            continue

        used_query.add(qi)
        used_fact.add(fi)
        overlap_pairs.append({
            "self_triplet": q_raw,
            "other_triplet": f_raw,
            "score": match_score,
            "inverse": inverse,
            "rel_sim": rel_sim,
        })

    matched = len(overlap_pairs)
    precision = matched / len(query_items) if query_items else 0.0
    recall = matched / len(fact_items) if fact_items else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "matched_triplets": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap_pairs_topk": overlap_pairs[:10],
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
    }
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
    parser.add_argument("--graphcheck_results_path", type=str, default=None,
        help="Optional GraphCheck results JSON path to attach final answers/metrics"
    )
    parser.add_argument("--graphcheck_attach_full", action="store_true",
        help="Attach full GraphCheck verification_process (can be large)"
    )
    parser.add_argument("--enable_binding_search", action="store_true",
        help="Enable query graph binding search between query and fact graphs"
    )
    parser.add_argument("--binding_top_k", type=int, default=5,
        help="Top-K bindings to return for query graph binding search"
    )
    parser.add_argument("--binding_beam_size", type=int, default=50,
        help="Beam size for query graph binding search"
    )
    parser.add_argument("--binding_cand_per_query", type=int, default=50,
        help="Candidate facts per query triple for binding search"
    )
    parser.add_argument("--binding_min_token_jaccard", type=float, default=0.5,
        help="Min token Jaccard threshold for binding search"
    )
    parser.add_argument("--binding_include_definitions", action="store_true",
        help="Include definition_triples in binding search"
    )
    parser.add_argument("--compare_soft_match", action="store_true",
        help="Enable LLM-based soft match for triplets (relation equivalence)"
    )
    parser.add_argument("--soft_match_threshold", type=float, default=0.65,
        help="Threshold for soft match acceptance"
    )
    parser.add_argument("--soft_match_topn_candidates", type=int, default=30,
        help="Top-N candidates per query triple to consider for soft match"
    )
    parser.add_argument("--soft_match_cache_path", type=str, default="./cache/softmatch_rel_pairs.jsonl",
        help="Cache path for relation equivalence results"
    )
    parser.add_argument("--compare_ensemble_match", action="store_true",
        help="Enable ensemble triplet matching (Lemma Jaccard + TF-IDF + Char N-gram)"
    )
    parser.add_argument("--ensemble_w_lemma", type=float, default=0.4,
        help="Ensemble weight for Lemma Token Jaccard (default: 0.4)"
    )
    parser.add_argument("--ensemble_w_tfidf", type=float, default=0.35,
        help="Ensemble weight for TF-IDF Cosine (default: 0.35)"
    )
    parser.add_argument("--ensemble_w_char", type=float, default=0.25,
        help="Ensemble weight for Char N-gram Cosine (default: 0.25)"
    )
    parser.add_argument("--ensemble_threshold", type=float, default=0.3,
        help="Ensemble matching threshold (default: 0.3)"
    )
    parser.add_argument("--compact_output", action="store_true", default=True,
        help="Write compact output JSON (default: true)"
    )
    parser.add_argument("--no_compact_output", action="store_false", dest="compact_output",
        help="Disable compact output JSON"
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
                question_triples = construct_model.normalize_casting_triples(question, question_triples)
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
            cot_triples = construct_model.normalize_casting_triples(question, cot_triples)
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
            cot_triples = construct_model.normalize_casting_triples(question, cot_triples)
            sample["cot_def_triples"] = cot_def_triples
            sample["cot_triples"] = cot_triples
            if (not cot_triples) and args.force_cot_regen:
                logger.info(f"Sample {sample.get('index')} extraction empty; regenerating CoT due to force flag.")
                cot_reasoning, cot_def_triples, cot_triples = regenerate_cot(args.cot_retry + 1)
                cot_triples = construct_model.normalize_casting_triples(question, cot_triples)
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
        
        compare_kwargs = {
            "match_mode": "token_jaccard",
            "min_token_jaccard": 0.5,
            "include_definitions": False,
            "ignore_ent_placeholders": True,
        }
        compare_strict_kwargs = {
            "match_mode": "exact",
            "include_definitions": True,
            "ignore_ent_placeholders": False,
        }

        comparison_question_doc = None
        comparison_question_doc_strict = None
        comparison_question_gold = None
        comparison_question_gold_strict = None
        if question_graph is not None:
            comparison_question_doc = question_graph.compare_with(doc_graph, **compare_kwargs)
            comparison_question_doc_strict = question_graph.compare_with(doc_graph, **compare_strict_kwargs)
        
        # 3. 두 그래프 비교
        comparison_result = cot_graph.compare_with(doc_graph, **compare_kwargs)
        comparison_strict = cot_graph.compare_with(doc_graph, **compare_strict_kwargs)
        
        # 3-1. Gold evidence 그래프 비교 (있을 경우)
        gold_graph = None
        comparison_gold = None
        comparison_gold_strict = None
        comparison_gold_vs_doc = None
        comparison_gold_vs_doc_strict = None
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
            comparison_gold = cot_graph.compare_with(gold_graph, **compare_kwargs)
            # Gold triplets vs Doc triplets 비교 추가
            comparison_gold_vs_doc = gold_graph.compare_with(doc_graph, **compare_kwargs)
            comparison_gold_strict = cot_graph.compare_with(gold_graph, **compare_strict_kwargs)
            comparison_gold_vs_doc_strict = gold_graph.compare_with(doc_graph, **compare_strict_kwargs)
            if question_graph is not None:
                comparison_question_gold = question_graph.compare_with(gold_graph, **compare_kwargs)
                comparison_question_gold_strict = question_graph.compare_with(gold_graph, **compare_strict_kwargs)
        
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
            "comparison": comparison_result,
            "comparison_strict": comparison_strict
        })

        if args.compare_soft_match:
            soft_match = _soft_match_triplets(
                cot_triples,
                doc_triples_list,
                construct_model,
                args._softmatch_cache,
                args.soft_match_cache_path,
                args.soft_match_threshold,
                args.soft_match_topn_candidates,
            )
            sample["comparison_soft_match"] = soft_match

        # --- Ensemble Triplet Matching (question_graph 기준만) ---
        if args.compare_ensemble_match:
            ensemble_kwargs = dict(
                w_lemma=args.ensemble_w_lemma,
                w_tfidf=args.ensemble_w_tfidf,
                w_char=args.ensemble_w_char,
                threshold=args.ensemble_threshold,
                top_n_per_query=5,
            )

            # Question vs Doc
            if question_graph is not None:
                ensemble_question_doc = ensemble_triplet_matching(
                    question_triples, doc_triples_list, **ensemble_kwargs,
                )
                sample["comparison_ensemble_question_vs_doc"] = ensemble_question_doc

            if gold_graph is not None:
                # Gold vs Doc
                ensemble_gold_doc = ensemble_triplet_matching(
                    gold_triples_list, doc_triples_list, **ensemble_kwargs,
                )
                sample["comparison_ensemble_gold_vs_doc"] = ensemble_gold_doc

                # Question vs Gold
                if question_graph is not None:
                    ensemble_question_gold = ensemble_triplet_matching(
                        question_triples, gold_triples_list, **ensemble_kwargs,
                    )
                    sample["comparison_ensemble_question_vs_gold"] = ensemble_question_gold

        def build_binding_result(query_def: List[str], query_rel: List[str], fact_def: List[str], fact_rel: List[str]) -> Optional[Dict[str, Any]]:
            if not args.enable_binding_search:
                return None
            query_list = []
            if args.binding_include_definitions:
                query_list.extend(query_def)
            query_list.extend(query_rel)
            fact_list = []
            if args.binding_include_definitions:
                fact_list.extend(fact_def)
            fact_list.extend(fact_rel)
            if not query_list or not fact_list:
                return {"k": args.binding_top_k, "bindings": []}
            result = search_query_graph_bindings(
                query_list,
                fact_list,
                top_k=args.binding_top_k,
                beam_size=args.binding_beam_size,
                cand_per_query=args.binding_cand_per_query,
                min_token_jaccard=args.binding_min_token_jaccard,
                include_definitions=args.binding_include_definitions,
            )
            top1 = result["bindings"][0] if result.get("bindings") else None
            result["top1_score"] = top1["score"] if top1 else 0.0
            result["top1_supported"] = len(top1["supported_pairs"]) if top1 else 0
            result["top1_unmatched"] = len(top1["unmatched_query_triples"]) if top1 else 0
            return result
        
        if question_graph is not None:
            sample.update({
                "question_graph": {
                    "definition_triples": question_def_triples,
                    "triples": question_triples,
                    "num_triplets": len(question_graph.total_triples)
                },
                "comparison_question_vs_doc": comparison_question_doc,
                "comparison_question_vs_doc_strict": comparison_question_doc_strict
            })
            binding_question_doc = build_binding_result(
                question_def_triples,
                question_triples,
                doc_def_triples_list,
                doc_triples_list,
            )
            if binding_question_doc is not None:
                sample["comparison_binding_question_vs_doc"] = binding_question_doc
        
        if gold_graph is not None:
            update_dict = {
                "gold_graph": {
                    "definition_triples": gold_def_triples_list,
                    "triples": gold_triples_list,
                    "num_triplets": len(gold_graph.total_triples)
                },
                "comparison_gold": comparison_gold,
                "comparison_gold_strict": comparison_gold_strict
            }
            if comparison_gold_vs_doc is not None:
                comparison_gold_vs_doc["gold_in_doc_coverage"] = comparison_gold_vs_doc.get("self_coverage")
                update_dict["comparison_gold_vs_doc"] = comparison_gold_vs_doc
                if comparison_gold_vs_doc_strict is not None:
                    comparison_gold_vs_doc_strict["gold_in_doc_coverage"] = comparison_gold_vs_doc_strict.get("self_coverage")
                update_dict["comparison_gold_vs_doc_strict"] = comparison_gold_vs_doc_strict
            if comparison_question_gold is not None:
                update_dict["comparison_question_vs_gold"] = comparison_question_gold
                update_dict["comparison_question_vs_gold_strict"] = comparison_question_gold_strict
            sample.update(update_dict)

        binding_cot_doc = build_binding_result(
            cot_def_triples,
            cot_triples,
            doc_def_triples_list,
            doc_triples_list,
        )
        if binding_cot_doc is not None:
            sample["comparison_binding_cot_vs_doc"] = binding_cot_doc

        if gold_graph is not None:
            binding_cot_gold = build_binding_result(
                cot_def_triples,
                cot_triples,
                gold_def_triples_list,
                gold_triples_list,
            )
            if binding_cot_gold is not None:
                sample["comparison_binding_cot_vs_gold"] = binding_cot_gold
            if question_graph is not None:
                binding_question_gold = build_binding_result(
                    question_def_triples,
                    question_triples,
                    gold_def_triples_list,
                    gold_triples_list,
                )
                if binding_question_gold is not None:
                    sample["comparison_binding_question_vs_gold"] = binding_question_gold
        
        logger.info(
            f"Sample {sample.get('index')}: "
            f"CoT triplets={len(cot_graph.total_triples)}, "
            f"Doc triplets={len(doc_graph.total_triples)}, "
            f"Overlap={comparison_result['triplet_overlap']}, "
            f"SubsetF1={comparison_result.get('subset_f1', 0.0):.3f}"
        )
        
        if comparison_gold:
            logger.info(
                f"Sample {sample.get('index')} (CoT vs Gold): "
                f"Gold triplets={len(gold_graph.total_triples)}, "
                f"Overlap={comparison_gold['triplet_overlap']}, "
                f"SubsetF1={comparison_gold.get('subset_f1', 0.0):.3f}"
            )
        if comparison_gold_vs_doc:
            logger.info(
                f"Sample {sample.get('index')} (Gold vs Doc): "
                f"Gold triplets={len(gold_graph.total_triples)}, "
                f"Doc triplets={len(doc_graph.total_triples)}, "
                f"Overlap={comparison_gold_vs_doc['triplet_overlap']}, "
                f"SubsetF1={comparison_gold_vs_doc.get('subset_f1', 0.0):.3f}"
            )
        if comparison_question_doc:
            logger.info(
                f"Sample {sample.get('index')} (Question vs Doc): "
                f"Question triplets={len(question_graph.total_triples)}, "
                f"Doc triplets={len(doc_graph.total_triples)}, "
                f"Overlap={comparison_question_doc['triplet_overlap']}, "
                f"SubsetF1={comparison_question_doc.get('subset_f1', 0.0):.3f}"
            )
        if comparison_question_gold:
            logger.info(
                f"Sample {sample.get('index')} (Question vs Gold): "
                f"Question triplets={len(question_graph.total_triples)}, "
                f"Gold triplets={len(gold_graph.total_triples)}, "
                f"Overlap={comparison_question_gold['triplet_overlap']}, "
                f"SubsetF1={comparison_question_gold.get('subset_f1', 0.0):.3f}"
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
    
    args._softmatch_cache = {}
    if args.compare_soft_match:
        args._softmatch_cache = _load_softmatch_cache(args.soft_match_cache_path)

    # 입력 데이터 로드
    with open(input_path, "r") as f:
        input_list = json.load(f)

    graphcheck_map: Optional[Dict[int, Dict[str, Any]]] = None
    if args.graphcheck_results_path:
        if not os.path.exists(args.graphcheck_results_path):
            logger.error(f"GraphCheck results not found: {args.graphcheck_results_path}")
        else:
            logger.info(f"Loading GraphCheck results from: {args.graphcheck_results_path}")
            with open(args.graphcheck_results_path, "r") as f:
                graphcheck_list = json.load(f)
            graphcheck_map = {}
            for item in graphcheck_list:
                idx = item.get("index")
                if idx is None:
                    continue
                graphcheck_item = {
                    "predicted_answer": item.get("predicted_answer"),
                    "answering_confidence": item.get("answering_confidence"),
                    "em_score": item.get("em_score"),
                    "f1_score": item.get("f1_score"),
                    "prediction": item.get("prediction"),
                    "best_path_index": item.get("best_path_index"),
                    "best_path_confidence": item.get("best_path_confidence"),
                    "best_path_infilling_confidence": item.get("best_path_infilling_confidence"),
                    "best_infilling_conf_path_index": item.get("best_infilling_conf_path_index"),
                    "best_infilling_conf_path_confidence": item.get("best_infilling_conf_path_confidence"),
                }
                if args.graphcheck_attach_full:
                    graphcheck_item["verification_process"] = item.get("verification_process")
                graphcheck_map[idx] = graphcheck_item
    
    if args.max_samples is not None:
        input_list = input_list[:args.max_samples]
    
    logger.info(f"Processing {len(input_list)} samples...")
    logger.info(f"Using model: {args.construct_model_name}")
    logger.info(f"Retriever URL: {args.retriever_url}")
    logger.info(f"BM25 top_k: {args.bm25_top_k}")
    
    # 각 샘플 처리
    result_list = []
    graphcheck_em_scores: List[float] = []
    graphcheck_f1_scores: List[float] = []
    for sample in tqdm(input_list):
        result = process_sample(sample, construct_model, direct_model, args)
        if graphcheck_map is not None:
            graphcheck_item = graphcheck_map.get(result.get("index"))
            if graphcheck_item:
                result["graphcheck"] = graphcheck_item
                if graphcheck_item.get("em_score") is not None:
                    graphcheck_em_scores.append(graphcheck_item["em_score"])
                if graphcheck_item.get("f1_score") is not None:
                    graphcheck_f1_scores.append(graphcheck_item["f1_score"])
        result_list.append(result)
    
    def compact_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            "uid": sample.get("uid"),
            "num_hops": sample.get("num_hops"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "gold_id_list": sample.get("gold_id_list", []),
        }
        compact["cot_graph"] = {
            "definition_triples": sample.get("cot_graph", {}).get("definition_triples", []),
            "triples": sample.get("cot_graph", {}).get("triples", []),
        }
        compact["question_graph"] = {
            "definition_triples": sample.get("question_graph", {}).get("definition_triples", []),
            "triples": sample.get("question_graph", {}).get("triples", []),
        }
        compact["document_graph"] = {
            "definition_triples": sample.get("document_graph", {}).get("definition_triples", []),
            "triples": sample.get("document_graph", {}).get("triples", []),
        }
        compact["gold_graph"] = {
            "definition_triples": sample.get("gold_graph", {}).get("definition_triples", []),
            "triples": sample.get("gold_graph", {}).get("triples", []),
        }
        retrieval_info = sample.get("retrieval_info", {}) or {}
        compact["retrieval_info"] = {
            "doc_id_list": retrieval_info.get("doc_id_list", []),
            "is_gold_list": retrieval_info.get("is_gold_list", []),
            "strategy": retrieval_info.get("strategy"),
        }
        comparison_gold_vs_doc = sample.get("comparison_gold_vs_doc") or {}
        compact["comparison_gold_vs_doc"] = {
            "triplet_f1": comparison_gold_vs_doc.get("triplet_f1"),
            "triplet_recall": comparison_gold_vs_doc.get("triplet_recall"),
            "gold_in_doc_coverage": comparison_gold_vs_doc.get("gold_in_doc_coverage"),
        }
        binding_question_doc = sample.get("comparison_binding_question_vs_doc") or {}
        compact["comparison_binding_question_vs_doc"] = {
            "top1_score": binding_question_doc.get("top1_score"),
            "top1_supported": binding_question_doc.get("top1_supported"),
            "top1_unmatched": binding_question_doc.get("top1_unmatched"),
        }
        binding_cot_doc = sample.get("comparison_binding_cot_vs_doc") or {}
        compact["comparison_binding_cot_vs_doc"] = {
            "top1_score": binding_cot_doc.get("top1_score"),
            "top1_supported": binding_cot_doc.get("top1_supported"),
            "top1_unmatched": binding_cot_doc.get("top1_unmatched"),
        }
        if args.compare_soft_match:
            soft_match = sample.get("comparison_soft_match") or {}
            compact["comparison_soft_match"] = {
                "precision": soft_match.get("precision"),
                "recall": soft_match.get("recall"),
                "f1": soft_match.get("f1"),
                "llm_calls": soft_match.get("llm_calls"),
                "cache_hits": soft_match.get("cache_hits"),
            }

        def _compact_ensemble(key: str) -> None:
            ens = sample.get(key)
            if ens is None:
                return
            compact[key] = {
                "matched_count": ens.get("matched_count"),
                "precision": ens.get("precision"),
                "recall": ens.get("recall"),
                "f1": ens.get("f1"),
                "per_method_avg": ens.get("per_method_avg"),
                "matched_pairs": ens.get("matched_pairs"),
                "best_match_per_query": ens.get("best_match_per_query"),
            }

        if args.compare_ensemble_match:
            _compact_ensemble("comparison_ensemble_question_vs_doc")
            _compact_ensemble("comparison_ensemble_gold_vs_doc")
            _compact_ensemble("comparison_ensemble_question_vs_gold")

        return compact

    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        output_list = [compact_sample(r) for r in result_list] if args.compact_output else result_list
        json.dump(output_list, f, indent=4)
    
    logger.info(f"Results saved to: {output_path}")
    
    # 전체 통계 계산
    valid_results = [r for r in result_list if r.get("comparison") is not None]
    valid_gold_results = [r for r in result_list if r.get("comparison_gold") is not None]
    valid_question_doc_results = [r for r in result_list if r.get("comparison_question_vs_doc") is not None]
    valid_question_gold_results = [r for r in result_list if r.get("comparison_question_vs_gold") is not None]
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
        avg_subset_f1 = sum(r["comparison"]["subset_f1"] for r in valid_results) / len(valid_results)
        logger.info(f"Average Subset F1: {avg_subset_f1:.3f}")
        
        if valid_gold_results:
            avg_f1_gold = sum(r["comparison_gold"]["triplet_f1"] for r in valid_gold_results) / len(valid_gold_results)
            avg_precision_gold = sum(r["comparison_gold"]["triplet_precision"] for r in valid_gold_results) / len(valid_gold_results)
            avg_recall_gold = sum(r["comparison_gold"]["triplet_recall"] for r in valid_gold_results) / len(valid_gold_results)
            
            logger.info(f"\n=== CoT vs Gold Summary ===")
            logger.info(f"Gold valid samples: {len(valid_gold_results)}")
            logger.info(f"Average Precision: {avg_precision_gold:.3f}")
            logger.info(f"Average Recall: {avg_recall_gold:.3f}")
            logger.info(f"Average F1: {avg_f1_gold:.3f}")
            avg_subset_f1_gold = sum(r["comparison_gold"]["subset_f1"] for r in valid_gold_results) / len(valid_gold_results)
            logger.info(f"Average Subset F1: {avg_subset_f1_gold:.3f}")
            
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
                avg_subset_f1_gold_vs_doc = sum(r["comparison_gold_vs_doc"]["subset_f1"] for r in valid_gold_vs_doc_results) / len(valid_gold_vs_doc_results)
                logger.info(f"Average Subset F1: {avg_subset_f1_gold_vs_doc:.3f}")

        if valid_question_doc_results:
            avg_f1_q_doc = sum(r["comparison_question_vs_doc"]["triplet_f1"] for r in valid_question_doc_results) / len(valid_question_doc_results)
            avg_precision_q_doc = sum(r["comparison_question_vs_doc"]["triplet_precision"] for r in valid_question_doc_results) / len(valid_question_doc_results)
            avg_recall_q_doc = sum(r["comparison_question_vs_doc"]["triplet_recall"] for r in valid_question_doc_results) / len(valid_question_doc_results)
            avg_subset_f1_q_doc = sum(r["comparison_question_vs_doc"]["subset_f1"] for r in valid_question_doc_results) / len(valid_question_doc_results)
            logger.info(f"\n=== Question vs Doc Summary ===")
            logger.info(f"Valid samples: {len(valid_question_doc_results)}")
            logger.info(f"Average Precision: {avg_precision_q_doc:.3f}")
            logger.info(f"Average Recall: {avg_recall_q_doc:.3f}")
            logger.info(f"Average F1: {avg_f1_q_doc:.3f}")
            logger.info(f"Average Subset F1: {avg_subset_f1_q_doc:.3f}")

        if valid_question_gold_results:
            avg_f1_q_gold = sum(r["comparison_question_vs_gold"]["triplet_f1"] for r in valid_question_gold_results) / len(valid_question_gold_results)
            avg_precision_q_gold = sum(r["comparison_question_vs_gold"]["triplet_precision"] for r in valid_question_gold_results) / len(valid_question_gold_results)
            avg_recall_q_gold = sum(r["comparison_question_vs_gold"]["triplet_recall"] for r in valid_question_gold_results) / len(valid_question_gold_results)
            avg_subset_f1_q_gold = sum(r["comparison_question_vs_gold"]["subset_f1"] for r in valid_question_gold_results) / len(valid_question_gold_results)
            logger.info(f"\n=== Question vs Gold Summary ===")
            logger.info(f"Valid samples: {len(valid_question_gold_results)}")
            logger.info(f"Average Precision: {avg_precision_q_gold:.3f}")
            logger.info(f"Average Recall: {avg_recall_q_gold:.3f}")
            logger.info(f"Average F1: {avg_f1_q_gold:.3f}")
            logger.info(f"Average Subset F1: {avg_subset_f1_q_gold:.3f}")

        if graphcheck_em_scores:
            avg_em = sum(graphcheck_em_scores) / len(graphcheck_em_scores)
            avg_f1_gc = sum(graphcheck_f1_scores) / len(graphcheck_f1_scores) if graphcheck_f1_scores else 0.0
            logger.info(f"\n=== GraphCheck Answering Summary ===")
            logger.info(f"GraphCheck attached samples: {len(graphcheck_em_scores)}")
            logger.info(f"Average EM: {avg_em:.3f}")
            logger.info(f"Average F1: {avg_f1_gc:.3f}")


if __name__ == "__main__":
    main()

