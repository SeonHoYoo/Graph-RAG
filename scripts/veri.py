import argparse
import json
import logging
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from direct import Direct
from model_library.base_model import BaseModel
from model_library.llm_clients import Qwen
from model_library.construct_model import ConstructModel
from utils.graph import Graph


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


class QwenVerificationModel:
    def __init__(self, model: Any, tokenizer: Any):
        self.client = Qwen(model, tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self.client.generate(prompt, max_tokens=max_new_tokens)

    def generate_with_confidence(self, prompt: str, max_new_tokens: int = 8) -> Tuple[str, float]:
        answer = self.generate(prompt, max_new_tokens=max_new_tokens)
        return answer, 0.0

    def parse_boolean_answer(self, answer: str) -> bool:
        answer = answer.split("\n")[0].lower().strip(" .")
        boolean_mapping = {
            "true": True, "false": False, "yes": True, "no": False,
            "it is impossible to say": False, "it's impossible to say": False,
            "it is impossible to tell": False, "it's impossible to tell": False,
            "it is not possible to say": False, "it's not possible to say": False,
            "it is not possible to tell": False, "it's not possible to tell": False
        }
        if answer in boolean_mapping:
            return boolean_mapping[answer]
        for sample_text, boolean_value in boolean_mapping.items():
            if answer.startswith(sample_text):
                return boolean_value
        logger.error(f"Unmapped answer detected: '{answer}'")
        return False

    def get_verification_prompt(
        self,
        claim: str,
        evidence: Optional[str] = None,
        definition_triples: Optional[List[str]] = None,
        existing_bindings: Optional[Dict[str, str]] = None,
    ) -> str:
        def_text = "\n".join(definition_triples) if definition_triples else "(none)"
        binding_text = json.dumps(existing_bindings, ensure_ascii=False) if existing_bindings else "{}"
        if not evidence:
            return (
                "Determine whether the claim is supported.\n"
                f"Claim: {claim}\n"
                f"Definition triples:\n{def_text}\n"
                f"Existing bindings: {binding_text}\n"
                "Treat the definition triples as hard type constraints for placeholders.\n"
                "Answer with only 'true' or 'false'."
            )
        return (
            "Determine whether the evidence supports the claim.\n"
            f"Evidence: {evidence}\n"
            f"Claim: {claim}\n"
            f"Definition triples:\n{def_text}\n"
            f"Existing bindings: {binding_text}\n"
            "Treat the definition triples as hard type constraints for placeholders.\n"
            "If the relation appears to match but the candidate entity violates a definition triple, answer 'false'.\n"
            "Answer with only 'true' or 'false'."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--input_filename", type=str, default=None)
    parser.add_argument("--input_file_path", type=str, default=None)
    parser.add_argument("--construct_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--construct_batch_size", type=int, default=1)
    parser.add_argument("--base_model_name", type=str, default="google/flan-t5-xl")
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--use_searchr1", action="store_true")
    parser.add_argument("--nudge_searchr1", action="store_true")
    parser.add_argument("--use_total_search_results", action="store_true")
    parser.add_argument("--searchr1_top_k", type=int, default=3)
    parser.add_argument("--searchr1_max_turns", type=int, default=3)
    parser.add_argument("--evidence_setting", type=str, choices=["open-book", "open-book+gold", "gold"], default="open-book")
    parser.add_argument("--verification_source", type=str, choices=["triplet", "doc"], default="triplet")
    parser.add_argument("--reasoning_mode", type=str, choices=["standard", "searchr1_graph"], default="standard")
    parser.add_argument("--verification_top_k", type=int, default=20)
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
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace('"', " ").replace("'", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def init_base_model_if_needed(direct_model: Direct) -> None:
    if direct_model.base_model is not None:
        return
    model_name = direct_model.base_model_name
    if model_name.lower().startswith("qwen"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )
        direct_model.base_model = QwenVerificationModel(model, tokenizer)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        direct_model.base_model = BaseModel(model, tokenizer)
    logger.info(f"Base model '{direct_model.base_model_name}' initialized for LLM verification.")


def retrieve_raw_question_docs(
    sample: Dict[str, Any],
    direct_model: Direct,
    args: argparse.Namespace,
    thinking: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    evidence_list = []
    doc_id_list = []
    is_gold_list = []
    gold_ids = {unicodedata.normalize("NFC", doc_id).strip() for doc_id in sample.get("gold_id_list", [])}
    search_info: Dict[str, Any] = {}

    if args.evidence_setting in {"open-book", "open-book+gold"}:
        hit_list, search_info = direct_model.retrieve(
            sample["question"],
            top_k=args.bm25_top_k,
            use_searchr1=args.use_searchr1,
            use_total_search_results=args.use_total_search_results,
            nudge_searchr1=args.nudge_searchr1,
            thinking=thinking,
        )
        for hit in hit_list:
            doc_id = unicodedata.normalize("NFC", hit["doc_id"]).strip()
            text = unicodedata.normalize("NFC", hit["text"]).strip()
            evidence = f"(Title: {doc_id}) {text}"
            if evidence not in evidence_list:
                evidence_list.append(evidence)
                doc_id_list.append(doc_id)
                is_gold_list.append(1 if doc_id in gold_ids else 0)

    if args.evidence_setting in {"gold", "open-book+gold"}:
        for doc_id, text in zip(sample.get("gold_id_list", []), sample.get("gold_evidence_list", [])):
            normalized_doc_id = unicodedata.normalize("NFC", doc_id).strip()
            normalized_text = unicodedata.normalize("NFC", text).strip()
            evidence = f"(Title: {normalized_doc_id}) {normalized_text}"
            if evidence not in evidence_list:
                evidence_list.append(evidence)
                doc_id_list.append(normalized_doc_id)
                is_gold_list.append(1)

    retrieval_info = {
        "query": sample["question"],
        "evidence_setting": args.evidence_setting,
        "doc_id_list": doc_id_list,
        "is_gold_list": is_gold_list,
    }
    if args.use_searchr1:
        retrieval_info["searchr1_answer"] = search_info.get("predicted_answer", "")
        retrieval_info["searchr1_reasoning_path"] = search_info.get("reasoning_path", "")
        retrieval_info["searchr1_reasoning_steps"] = search_info.get("reasoning_steps", [])
        retrieval_info["num_turns"] = search_info.get("num_turns", 0)
        retrieval_info["retrieval_turns"] = search_info.get("retrieval_turns", [])
    return evidence_list, retrieval_info


def merge_unique_documents(base_docs: List[str], new_docs: List[str]) -> List[str]:
    merged = list(base_docs)
    seen = set(base_docs)
    for doc in new_docs:
        if doc not in seen:
            merged.append(doc)
            seen.add(doc)
    return merged


def build_searchr1_graph_thinking(
    question: str,
    target_triplet: str,
    known_bindings: Dict[str, str],
    remaining_triplets: List[str],
    definition_triples: List[str],
) -> str:
    def_text = "\n".join(definition_triples) if definition_triples else "(none)"
    binding_text = json.dumps(known_bindings, ensure_ascii=False) if known_bindings else "{}"
    remaining_text = "\n".join(remaining_triplets) if remaining_triplets else "(none)"
    return (
        f"Question: {question}\n"
        "Use the graph state below to guide retrieval and reasoning.\n"
        f"Current target triplet:\n{target_triplet}\n"
        f"Current bindings:\n{binding_text}\n"
        f"Definition triples:\n{def_text}\n"
        f"Remaining unresolved triplets:\n{remaining_text}\n"
        "Reason step by step about the current target triplet and search for evidence that helps resolve it."
    )


def build_question_graph(sample: Dict[str, Any], construct_model: ConstructModel) -> Tuple[List[str], List[str], Graph]:
    question_sample = {"question": sample["question"]}
    question_sample = construct_model.process_sample(question_sample)
    def_triples = question_sample.get("definition_triples", [])
    triples = question_sample.get("triples", [])
    return def_triples, triples, Graph(def_triples, triples)


def build_document_graph(
    retrieved_documents: List[str],
    construct_model: ConstructModel,
) -> Tuple[List[str], List[str], Graph]:
    doc_def_triples: List[str] = []
    doc_triples: List[str] = []
    for doc in retrieved_documents:
        try:
            extracted_def, extracted_triples = construct_model.extract_triplets_from_document(doc)
            doc_def_triples.extend(extracted_def)
            doc_triples.extend(extracted_triples)
        except Exception as exc:
            logger.warning(f"Failed to extract triples from retrieved document: {exc}")
    return doc_def_triples, doc_triples, Graph(doc_def_triples, doc_triples)


def load_precomputed_document_evidence(
    sample: Dict[str, Any],
    evidence_setting: str,
) -> Tuple[List[str], List[str], List[str], Dict[str, Any], Optional[str]]:
    if evidence_setting == "gold":
        graph_key = "gold_graph"
        graph_data = sample.get("gold_graph") or {}
    else:
        graph_key = "doc_graph"
        graph_data = sample.get("doc_graph") or sample.get("document_graph") or {}

    doc_def_triples = list(graph_data.get("definition_triples", []))
    doc_triples = list(graph_data.get("triples", []))
    retrieved_documents: List[str] = []
    for item in graph_data.get("per_document", []):
        document = item.get("document")
        if document:
            retrieved_documents.append(document)
    retrieval_info = dict(graph_data.get("retrieval_info", {}))
    if not (retrieved_documents or doc_triples):
        graph_key = None
    return retrieved_documents, doc_def_triples, doc_triples, retrieval_info, graph_key


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def verify_with_llm(
    question_triplet: str,
    evidence_items: List[str],
    direct_model: Direct,
    question_def_triples: List[str],
    existing_bindings: Dict[str, str],
) -> Dict[str, Any]:
    init_base_model_if_needed(direct_model)
    evidence_text = direct_model.truncate("\n".join(evidence_items))
    def_text = "\n".join(question_def_triples) if question_def_triples else "(none)"
    binding_text = json.dumps(existing_bindings, ensure_ascii=False) if existing_bindings else "{}"
    verify_prompt = direct_model.base_model.get_verification_prompt(
        question_triplet,
        evidence_text,
        definition_triples=question_def_triples,
        existing_bindings=existing_bindings,
    )
    raw_verify_output, conf = direct_model.base_model.generate_with_confidence(verify_prompt, max_new_tokens=8)
    supported = direct_model.base_model.parse_boolean_answer(raw_verify_output)
    binding_prompt = (
        "You are extracting placeholder bindings from evidence for a question triplet.\n"
        f"Question triplet: {question_triplet}\n"
        f"Definition triples:\n{def_text}\n"
        f"Existing bindings: {binding_text}\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Return valid JSON only with this schema:\n"
        "{\"new_bindings\": {}, \"notes\": []}\n"
        "Rules:\n"
        "- Treat the definition triples as hard type constraints for placeholders.\n"
        "- Put only newly inferred placeholder bindings in new_bindings.\n"
        "- If no new binding exists, return an empty object.\n"
        "- If a candidate binding violates the definition triples, do not include it in new_bindings.\n"
        "- notes should be short strings.\n"
        "- Do not include any extra text outside the JSON object.\n"
    )
    raw_binding_output = direct_model.base_model.generate(binding_prompt, max_new_tokens=256)
    parsed = extract_json_block(raw_binding_output) or {}
    new_bindings = parsed.get("new_bindings", {})
    notes = parsed.get("notes", [])
    result = {
        "supported": supported,
        "new_bindings": new_bindings if isinstance(new_bindings, dict) else {},
        "notes": notes if isinstance(notes, list) else [],
        "verification_confidence": conf,
        "raw_verify_output": raw_verify_output,
        "raw_binding_output": raw_binding_output,
    }
    if not result["notes"] and raw_binding_output.strip():
        result["notes"] = ["binding_output_parsed"]
    if parsed == {} and raw_binding_output.strip():
        result["notes"].append("binding_json_parse_failed")
    return result


def verify_question_triplet(
    question_triplet: str,
    source: str,
    docs: List[str],
    doc_triples: List[str],
    direct_model: Direct,
    known_bindings: Dict[str, str],
    top_k: int,
    question_def_triples: List[str],
) -> Dict[str, Any]:
    if source == "triplet":
        matched_evidence = doc_triples if top_k <= 0 else doc_triples[:top_k]
        llm_result = verify_with_llm(
            question_triplet,
            matched_evidence,
            direct_model,
            question_def_triples,
            known_bindings,
        )
        result = {
            "question_triplet": question_triplet,
            "supported": llm_result["supported"],
            "matched_evidence": matched_evidence,
            "verification_confidence": llm_result["verification_confidence"],
        }
        if llm_result["new_bindings"]:
            result["new_bindings"] = llm_result["new_bindings"]
        if llm_result["notes"]:
            result["notes"] = llm_result["notes"]
        return result

    matched_docs = docs if top_k <= 0 else docs[:top_k]
    llm_result = verify_with_llm(
        question_triplet,
        matched_docs,
        direct_model,
        question_def_triples,
        known_bindings,
    )
    result = {
        "question_triplet": question_triplet,
        "supported": llm_result["supported"],
        "matched_evidence": matched_docs,
        "verification_confidence": llm_result["verification_confidence"],
    }
    if llm_result["new_bindings"]:
        result["new_bindings"] = llm_result["new_bindings"]
    if llm_result["notes"]:
        result["notes"] = llm_result["notes"]
    return result


def infer_answer_from_bindings(question: str, bindings: Dict[str, str], ent_types: Dict[str, str]) -> Optional[str]:
    question_norm = normalize_text(question)
    location_values = []
    for ent, value in sorted(bindings.items()):
        ent_type = ent_types.get(ent, "")
        if any(keyword in ent_type for keyword in ["country", "location"]):
            norm_value = normalize_text(value)
            if norm_value and norm_value not in location_values:
                location_values.append(norm_value)
    if "same country" in question_norm or "same location" in question_norm:
        if len(location_values) < 2:
            return None
        return "yes" if len(location_values) == 1 else "no"
    return None


def process_sample(
    sample: Dict[str, Any],
    construct_model: ConstructModel,
    direct_model: Direct,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.reasoning_mode == "searchr1_graph":
        return process_sample_searchr1_graph(sample, construct_model, direct_model, args)

    question_def_triples, question_triples, question_graph = build_question_graph(sample, construct_model)
    precomputed_docs, precomputed_def_triples, precomputed_doc_triples, precomputed_retrieval_info, precomputed_graph_key = load_precomputed_document_evidence(
        sample,
        args.evidence_setting,
    )
    use_precomputed_doc_graph = bool(precomputed_doc_triples or precomputed_docs)

    if use_precomputed_doc_graph:
        retrieved_documents = precomputed_docs
        retrieval_info = {
            "source": "precomputed_doc_graph",
            "precomputed_graph_key": precomputed_graph_key,
            **precomputed_retrieval_info,
        }
    else:
        retrieved_documents, retrieval_info = retrieve_raw_question_docs(sample, direct_model, args)

    result: Dict[str, Any] = {
        "index": sample.get("index"),
        "uid": sample.get("uid"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "answer_aliases": sample.get("answer_aliases", []),
        "num_hops": sample.get("num_hops"),
        "gold_id_list": sample.get("gold_id_list", []),
        "question_graph": {
            "definition_triples": question_def_triples,
            "triples": question_triples,
            "num_triplets": len(question_graph.total_triples),
        },
        "retrieved_documents": retrieved_documents,
        "retrieval_info": retrieval_info,
    }

    doc_def_triples: List[str] = []
    doc_triples: List[str] = []
    if args.verification_source == "triplet":
        if use_precomputed_doc_graph:
            doc_def_triples = precomputed_def_triples
            doc_triples = precomputed_doc_triples
            doc_graph = Graph(doc_def_triples, doc_triples)
        else:
            doc_def_triples, doc_triples, doc_graph = build_document_graph(retrieved_documents, construct_model)
        result["document_graph"] = {
            "definition_triples": doc_def_triples,
            "triples": doc_triples,
            "num_triplets": len(doc_graph.total_triples),
        }
        if use_precomputed_doc_graph:
            result["document_graph"]["source"] = "precomputed_input"
            result["document_graph"]["graph_key"] = precomputed_graph_key

    ent_types = {
        def_triple.latent_entity: normalize_text(def_triple.definition)
        for def_triple in question_graph.def_triples
        if def_triple.latent_entity and def_triple.definition
    }
    known_bindings: Dict[str, str] = {}
    triplet_verification = []
    for question_triplet in question_triples:
        triplet_result = verify_question_triplet(
            question_triplet=question_triplet,
            source=args.verification_source,
            docs=retrieved_documents,
            doc_triples=doc_triples,
            direct_model=direct_model,
            known_bindings=known_bindings,
            top_k=args.verification_top_k,
            question_def_triples=question_def_triples,
        )
        triplet_verification.append(triplet_result)
        if triplet_result.get("supported") and triplet_result.get("new_bindings"):
            known_bindings.update(triplet_result["new_bindings"])

    supported_triplets = sum(1 for item in triplet_verification if item.get("supported"))
    total_triplets = len(question_triples)
    answer_from_verification = infer_answer_from_bindings(sample["question"], known_bindings, ent_types)

    result["triplet_verification"] = triplet_verification
    result["entity_bindings"] = known_bindings
    result["supported_triplets"] = supported_triplets
    result["total_triplets"] = total_triplets
    result["triplet_support_score"] = supported_triplets / total_triplets if total_triplets else 0.0
    result["answer_from_verification"] = answer_from_verification
    result["answer_matches_gold"] = answer_from_verification == normalize_text(sample.get("answer", "")) if answer_from_verification else None
    return result


def process_sample_searchr1_graph(
    sample: Dict[str, Any],
    construct_model: ConstructModel,
    direct_model: Direct,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question_def_triples, question_triples, question_graph = build_question_graph(sample, construct_model)

    result: Dict[str, Any] = {
        "index": sample.get("index"),
        "uid": sample.get("uid"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "answer_aliases": sample.get("answer_aliases", []),
        "num_hops": sample.get("num_hops"),
        "gold_id_list": sample.get("gold_id_list", []),
        "question_graph": {
            "definition_triples": question_def_triples,
            "triples": question_triples,
            "num_triplets": len(question_graph.total_triples),
        },
    }

    ent_types = {
        def_triple.latent_entity: normalize_text(def_triple.definition)
        for def_triple in question_graph.def_triples
        if def_triple.latent_entity and def_triple.definition
    }

    known_bindings: Dict[str, str] = {}
    cumulative_documents: List[str] = []
    triplet_verification: List[Dict[str, Any]] = []
    step_trace: List[Dict[str, Any]] = []
    retrieval_steps: List[Dict[str, Any]] = []

    for step_idx, question_triplet in enumerate(question_triples):
        remaining_triplets = question_triples[step_idx:]
        thinking = build_searchr1_graph_thinking(
            question=sample["question"],
            target_triplet=question_triplet,
            known_bindings=known_bindings,
            remaining_triplets=remaining_triplets,
            definition_triples=question_def_triples,
        )
        step_documents, step_retrieval_info = retrieve_raw_question_docs(
            sample=sample,
            direct_model=direct_model,
            args=args,
            thinking=thinking,
        )
        cumulative_documents = merge_unique_documents(cumulative_documents, step_documents)

        doc_def_triples: List[str] = []
        doc_triples: List[str] = []
        step_doc_graph_payload: Optional[Dict[str, Any]] = None
        if args.verification_source == "triplet":
            doc_def_triples, doc_triples, step_doc_graph = build_document_graph(cumulative_documents, construct_model)
            step_doc_graph_payload = {
                "definition_triples": doc_def_triples,
                "triples": doc_triples,
                "num_triplets": len(step_doc_graph.total_triples),
            }

        bindings_before = dict(known_bindings)
        triplet_result = verify_question_triplet(
            question_triplet=question_triplet,
            source=args.verification_source,
            docs=cumulative_documents,
            doc_triples=doc_triples,
            direct_model=direct_model,
            known_bindings=known_bindings,
            top_k=args.verification_top_k,
            question_def_triples=question_def_triples,
        )
        triplet_verification.append(triplet_result)
        if triplet_result.get("supported") and triplet_result.get("new_bindings"):
            known_bindings.update(triplet_result["new_bindings"])

        step_trace.append({
            "step_index": step_idx,
            "target_triplet": question_triplet,
            "graph_guidance": thinking,
            "bindings_before": bindings_before,
            "bindings_after": dict(known_bindings),
            "step_retrieval_info": step_retrieval_info,
            "step_retrieved_documents": step_documents,
            "cumulative_retrieved_documents": cumulative_documents,
            "verification_result": triplet_result,
            "partial_document_graph": step_doc_graph_payload,
        })
        retrieval_steps.append(step_retrieval_info)

    supported_triplets = sum(1 for item in triplet_verification if item.get("supported"))
    total_triplets = len(question_triples)
    answer_from_verification = infer_answer_from_bindings(sample["question"], known_bindings, ent_types)

    result["reasoning_mode"] = "searchr1_graph"
    result["retrieved_documents"] = cumulative_documents
    result["retrieval_info"] = {
        "reasoning_mode": "searchr1_graph",
        "num_steps": len(retrieval_steps),
        "step_retrievals": retrieval_steps,
    }
    if args.verification_source == "triplet":
        final_doc_def_triples, final_doc_triples, final_doc_graph = build_document_graph(cumulative_documents, construct_model)
        result["document_graph"] = {
            "definition_triples": final_doc_def_triples,
            "triples": final_doc_triples,
            "num_triplets": len(final_doc_graph.total_triples),
        }
    result["step_trace"] = step_trace
    result["triplet_verification"] = triplet_verification
    result["entity_bindings"] = known_bindings
    result["supported_triplets"] = supported_triplets
    result["total_triplets"] = total_triplets
    result["triplet_support_score"] = supported_triplets / total_triplets if total_triplets else 0.0
    result["answer_from_verification"] = answer_from_verification
    result["answer_matches_gold"] = answer_from_verification == normalize_text(sample.get("answer", "")) if answer_from_verification else None
    return result


def main() -> None:
    args = parse_args()
    args.dataset, args.input_filename, input_path = resolve_input_args(args)

    construct_model = ConstructModel(
        construct_model_name=args.construct_model_name,
        dataset_name=args.dataset,
        api_key=args.api_key,
        batch_size=args.construct_batch_size,
    )

    direct_args = argparse.Namespace(
        dataset=args.dataset,
        input_filename=args.input_filename,
        direct_filename=None,
        base_model_name=args.base_model_name,
        setting="open-book",
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
            result_list.append(process_sample(sample, construct_model, direct_model, args))
        except Exception as exc:
            logger.error(f"Failed to process sample {sample.get('index')}: {exc}")
            result_list.append({
                "index": sample.get("index"),
                "uid": sample.get("uid"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "error": str(exc),
            })

    base_output_name = args.output_filename or f"verification_{args.input_filename}"
    output_stem, output_ext = os.path.splitext(base_output_name)
    if not output_ext:
        output_ext = ".json"
    slurm_job_id = os.getenv("SLURM_JOB_ID", os.getenv("JOB_ID", "local"))
    output_filename = (
        f"{output_stem}_{args.reasoning_mode}_{args.evidence_setting}_{args.verification_source}_llm_{slurm_job_id}_{len(result_list)}{output_ext}"
    )
    output_dir = args.output_dir or os.path.join(
        "results", args.dataset, "verification", args.construct_model_name.split("/")[-1]
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as handle:
        json.dump(result_list, handle, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
