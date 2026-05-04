import argparse
import json
import logging
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from direct import Direct
from model_library.base_model import BaseModel
from model_library.llm_clients import GPT, Qwen


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
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "it is impossible to say": False,
            "it's impossible to say": False,
            "it is impossible to tell": False,
            "it's impossible to tell": False,
            "it is not possible to say": False,
            "it's not possible to say": False,
            "it is not possible to tell": False,
            "it's not possible to tell": False,
        }
        if answer in boolean_mapping:
            return boolean_mapping[answer]
        for sample_text, boolean_value in boolean_mapping.items():
            if answer.startswith(sample_text):
                return boolean_value
        logger.error("Unmapped answer detected: '%s'", answer)
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


class GPTVerificationModel:
    def __init__(self, model_name: str, client: Any):
        self.client = GPT(model_name, client)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self.client.generate(prompt, max_tokens=max_new_tokens)

    def generate_with_confidence(self, prompt: str, max_new_tokens: int = 8) -> Tuple[str, float]:
        answer = self.generate(prompt, max_new_tokens=max_new_tokens)
        return answer, 0.0

    def parse_boolean_answer(self, answer: str) -> bool:
        answer = answer.split("\n")[0].lower().strip(" .")
        boolean_mapping = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "it is impossible to say": False,
            "it's impossible to say": False,
            "it is impossible to tell": False,
            "it's impossible to tell": False,
            "it is not possible to say": False,
            "it's not possible to say": False,
            "it is not possible to tell": False,
            "it's not possible to tell": False,
        }
        if answer in boolean_mapping:
            return boolean_mapping[answer]
        for sample_text, boolean_value in boolean_mapping.items():
            if answer.startswith(sample_text):
                return boolean_value
        logger.error("Unmapped answer detected: '%s'", answer)
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


def init_base_model_if_needed(direct_model: Direct) -> None:
    if direct_model.base_model is not None:
        return
    model_name = direct_model.base_model_name
    model_name_lower = model_name.lower()
    if model_name_lower.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT verification models.")
        client = OpenAI(api_key=api_key)
        direct_model.base_model = GPTVerificationModel(model_name, client)
    elif model_name_lower.startswith("qwen"):
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
    logger.info("Base model '%s' initialized for LLM verification.", direct_model.base_model_name)


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
    run_verifier: bool = True,
    run_binding_extraction: bool = True,
) -> Dict[str, Any]:
    init_base_model_if_needed(direct_model)
    evidence_text = direct_model.truncate("\n".join(evidence_items))
    def_text = "\n".join(question_def_triples) if question_def_triples else "(none)"
    binding_text = json.dumps(existing_bindings, ensure_ascii=False) if existing_bindings else "{}"
    raw_verify_output = ""
    conf = 0.0
    supported: Optional[bool] = None
    if run_verifier:
        verify_prompt = direct_model.base_model.get_verification_prompt(
            question_triplet,
            evidence_text,
            definition_triples=question_def_triples,
            existing_bindings=existing_bindings,
        )
        raw_verify_output, conf = direct_model.base_model.generate_with_confidence(verify_prompt, max_new_tokens=8)
        supported = direct_model.base_model.parse_boolean_answer(raw_verify_output)

    raw_binding_output = ""
    parsed = {}
    new_bindings = {}
    notes = []
    if run_binding_extraction:
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
            "- Use placeholder keys exactly as they appear in the question triplet, for example (ENT1).\n"
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
    docs: List[str],
    direct_model: Direct,
    known_bindings: Dict[str, str],
    top_k: int,
    question_def_triples: List[str],
    run_verifier: bool = True,
    run_binding_extraction: bool = True,
) -> Dict[str, Any]:
    matched_docs = docs if top_k <= 0 else docs[:top_k]
    llm_result = verify_with_llm(
        question_triplet,
        matched_docs,
        direct_model,
        question_def_triples,
        known_bindings,
        run_verifier=run_verifier,
        run_binding_extraction=run_binding_extraction,
    )
    result = {
        "question_triplet": question_triplet,
        "supported": llm_result["supported"],
        "matched_evidence": matched_docs,
    }
    if llm_result["new_bindings"]:
        result["new_bindings"] = llm_result["new_bindings"]
    if llm_result.get("raw_verify_output"):
        result["raw_verify_output"] = llm_result["raw_verify_output"]
    if llm_result.get("raw_binding_output"):
        result["raw_binding_output"] = llm_result["raw_binding_output"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--input_filename", type=str, default=None)
    parser.add_argument("--input_file_path", type=str, default=None)
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--use_searchr1", action="store_true", default=True)
    parser.add_argument("--nudge_searchr1", action="store_true", default=True)
    parser.add_argument("--use_verification", action="store_true")
    parser.add_argument("--use_verifier", action="store_true")
    parser.add_argument("--use_binding_extraction", action="store_true")
    parser.add_argument("--use_total_search_results", action="store_true")
    parser.add_argument("--searchr1_top_k", type=int, default=3)
    parser.add_argument("--searchr1_max_turns", type=int, default=5)
    parser.add_argument("--evidence_setting", type=str, choices=["open-book", "open-book+gold", "gold"], default="open-book")
    parser.add_argument("--graph_prompt_mode", type=str, choices=["full_graph", "stepwise"], default="stepwise")
    parser.add_argument("--verification_top_k", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
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
                    raise ValueError(
                        "When using --input_file_path outside datasets/<dataset>/..., --dataset must also be provided."
                    )

        input_filename = os.path.basename(input_path)
        return dataset, input_filename, input_path

    if not args.dataset or not args.input_filename:
        raise ValueError("Provide either --input_file_path or both --dataset and --input_filename.")

    input_path = os.path.join("datasets", args.dataset, "claims", args.input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return args.dataset, args.input_filename, input_path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    return " ".join(text.split())


def get_question_graph(sample: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    question_graph = sample.get("question_graph")
    if isinstance(question_graph, dict):
        definition_triples = list(question_graph.get("definition_triples", []) or [])
        triples = list(question_graph.get("triples", []) or [])
    else:
        definition_triples = list(sample.get("definition_triples", []) or [])
        triples = list(sample.get("triples", []) or [])
    return definition_triples, triples


def merge_unique_documents(base_docs: List[str], new_docs: List[str]) -> List[str]:
    merged = list(base_docs)
    seen = set(base_docs)
    for doc in new_docs:
        if doc not in seen:
            merged.append(doc)
            seen.add(doc)
    return merged


def extract_triplet_entities(triplet: str) -> List[str]:
    return re.findall(r"\(ENT\d+\)", triplet)


def get_current_definition_triples(
    current_triplet: str,
    definition_triples: List[str],
) -> List[str]:
    target_ents = set(extract_triplet_entities(current_triplet))
    if not target_ents:
        return []
    current_defs = []
    for definition in definition_triples:
        ent_matches = extract_triplet_entities(definition)
        if ent_matches and ent_matches[0] in target_ents:
            current_defs.append(definition)
    return current_defs


def format_bindings(bindings: Dict[str, str]) -> str:
    if not bindings:
        return "(none)"
    return "\n".join(f"{ent} = {value}" for ent, value in sorted(bindings.items()))


def build_triplet_query(
    current_triplet: str,
    current_definition_triples: List[str],
    resolved_bindings: Dict[str, str],
) -> str:
    query = current_triplet.replace("[SEP]", " ")
    for ent, value in sorted(resolved_bindings.items(), key=lambda item: len(item[0]), reverse=True):
        query = query.replace(ent, value)
    query = query.replace("  ", " ").strip()
    return query


def build_graph_step_thinking(
    current_triplet: str,
    definition_triples: List[str],
    all_triplets: List[str],
    resolved_bindings: Dict[str, str],
    graph_prompt_mode: str,
) -> str:
    current_definition_triples = get_current_definition_triples(current_triplet, definition_triples)
    def_text = "\n".join(current_definition_triples) if current_definition_triples else "(none)"
    binding_text = format_bindings(resolved_bindings)
    if graph_prompt_mode == "full_graph":
        all_def_text = "\n".join(definition_triples) if definition_triples else "(none)"
        all_triplet_text = "\n".join(all_triplets) if all_triplets else "(none)"
        return (
            f"Current triplet:\n{current_triplet}\n\n"
            f"Known bindings:\n{binding_text}\n\n"
            f"Graph triplets:\n{all_triplet_text}\n\n"
            f"Type constraints:\n{all_def_text}\n\n"
            "Resolve the current triplet first using the graph state.\n"
            "Do not assume new bindings without evidence."
        )

    return (
        f"Current triplet:\n{current_triplet}\n\n"
        f"Known bindings:\n{binding_text}\n\n"
        f"Type constraints:\n{def_text}\n\n"
        "Find evidence only for the current triplet.\n"
        "Do not assume new bindings without evidence."
    )


def retrieve_step_documents(
    sample: Dict[str, Any],
    direct_model: Direct,
    args: argparse.Namespace,
    query: str,
    thinking: str,
) -> Tuple[List[str], Dict[str, Any]]:
    gold_ids = {normalize_text(doc_id) for doc_id in sample.get("gold_id_list", [])}
    retrieved_documents: List[str] = []
    doc_id_list: List[str] = []
    is_gold_list: List[int] = []
    search_info: Dict[str, Any] = {}

    if args.evidence_setting in {"open-book", "open-book+gold"}:
        hit_list, search_info = direct_model.retrieve(
            query,
            top_k=args.bm25_top_k,
            use_searchr1=args.use_searchr1,
            use_total_search_results=args.use_total_search_results,
            nudge_searchr1=args.nudge_searchr1,
            thinking=thinking,
        )
        for hit in hit_list:
            doc_id = normalize_text(hit["doc_id"])
            text = normalize_text(hit["text"])
            evidence = f"(Title: {doc_id}) {text}"
            if evidence not in retrieved_documents:
                retrieved_documents.append(evidence)
                doc_id_list.append(doc_id)
                is_gold_list.append(1 if doc_id in gold_ids else 0)

    if args.evidence_setting in {"gold", "open-book+gold"}:
        for doc_id, text in zip(sample.get("gold_id_list", []), sample.get("gold_evidence_list", [])):
            normalized_doc_id = normalize_text(doc_id)
            normalized_text = normalize_text(text)
            evidence = f"(Title: {normalized_doc_id}) {normalized_text}"
            if evidence not in retrieved_documents:
                retrieved_documents.append(evidence)
                doc_id_list.append(normalized_doc_id)
                is_gold_list.append(1)

    retrieval_info: Dict[str, Any] = {
        "query": query,
        "original_question": sample.get("question"),
        "evidence_setting": args.evidence_setting,
        "doc_id_list": doc_id_list,
        "is_gold_list": is_gold_list,
        "full_response": search_info.get("full_response", ""),
        "searchr1_answer": search_info.get("predicted_answer", ""),
        "searchr1_reasoning_path": search_info.get("reasoning_path", ""),
        "searchr1_reasoning_steps": search_info.get("reasoning_steps", []),
        "num_turns": search_info.get("num_turns", 0),
        "retrieval_turns": search_info.get("retrieval_turns", []),
    }
    return retrieved_documents, retrieval_info


def process_sample(
    sample: Dict[str, Any],
    direct_model: Direct,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    definition_triples, graph_triplets = get_question_graph(sample)
    cumulative_documents: List[str] = []
    step_trace: List[Dict[str, Any]] = []
    known_bindings: Dict[str, str] = {}
    triplet_verification: List[Dict[str, Any]] = []

    if not graph_triplets:
        result = dict(sample)
        result["question_graph"] = {
            "definition_triples": definition_triples,
            "triples": graph_triplets,
        }
        result["error"] = "No graph triplets found in input sample."
        return result

    for step_idx, graph_triplet in enumerate(graph_triplets):
        current_definition_triples = get_current_definition_triples(graph_triplet, definition_triples)
        step_query = build_triplet_query(
            current_triplet=graph_triplet,
            current_definition_triples=current_definition_triples,
            resolved_bindings=known_bindings,
        )
        thinking = build_graph_step_thinking(
            current_triplet=graph_triplet,
            definition_triples=definition_triples,
            all_triplets=graph_triplets,
            resolved_bindings=known_bindings,
            graph_prompt_mode=args.graph_prompt_mode,
        )
        step_documents, step_retrieval_info = retrieve_step_documents(sample, direct_model, args, step_query, thinking)
        cumulative_documents = merge_unique_documents(cumulative_documents, step_documents)
        bindings_before = dict(known_bindings)
        verification_result: Optional[Dict[str, Any]] = None
        verification_enabled = args.use_verification or args.use_verifier or args.use_binding_extraction
        if verification_enabled:
            verification_result = verify_question_triplet(
                question_triplet=graph_triplet,
                docs=cumulative_documents,
                direct_model=direct_model,
                known_bindings=known_bindings,
                top_k=args.verification_top_k,
                question_def_triples=current_definition_triples,
                run_verifier=(args.use_verification or args.use_verifier),
                run_binding_extraction=(args.use_verification or args.use_binding_extraction),
            )
            triplet_verification.append(verification_result)
            if verification_result.get("new_bindings"):
                known_bindings.update(verification_result["new_bindings"])
        step_trace.append({
            "step_index": step_idx,
            "target_triplet": graph_triplet,
            "step_query": step_query,
            "graph_guidance": thinking,
            "current_definition_triples": current_definition_triples,
            "bindings_before": bindings_before,
            "bindings_after": dict(known_bindings),
            "step_retrieved_documents": step_documents,
            "step_retrieval_info": step_retrieval_info,
            "verification_result": verification_result,
            "cumulative_retrieved_documents": list(cumulative_documents),
        })

    final_retrieval = step_trace[-1]["step_retrieval_info"] if step_trace else {}
    predicted_answer = final_retrieval.get("searchr1_answer", "")

    result = dict(sample)
    result["question_graph"] = {
        "definition_triples": definition_triples,
        "triples": graph_triplets,
        "num_triplets": len(graph_triplets),
    }
    result["retrieved_documents"] = cumulative_documents
    result["predicted_answer"] = predicted_answer
    result["reasoning_path"] = final_retrieval.get("searchr1_reasoning_path", "")
    result["reasoning_steps"] = final_retrieval.get("searchr1_reasoning_steps", [])
    result["step_trace"] = step_trace
    if args.use_verification or args.use_verifier or args.use_binding_extraction:
        result["entity_bindings"] = known_bindings
        result["triplet_verification"] = triplet_verification
    return result
    return result


def main() -> None:
    args = parse_args()
    args.dataset, args.input_filename, input_path = resolve_input_args(args)
    if args.use_verification:
        args.use_verifier = True
        args.use_binding_extraction = True
    if not args.use_searchr1:
        logger.warning("This script is intended for SearchR1. Forcing use_searchr1=True.")
        args.use_searchr1 = True
    if not args.nudge_searchr1:
        logger.warning("Graph thinking injection requires infer_with_nudge. Forcing nudge_searchr1=True.")
        args.nudge_searchr1 = True
    if not (args.use_verifier or args.use_binding_extraction):
        logger.info("Running without verifier or binding extraction. base_model_name will not be used.")

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

    with open(input_path, "r", encoding="utf-8") as handle:
        input_list = json.load(handle)
    if args.max_samples is not None:
        input_list = input_list[:args.max_samples]

    results = []
    for sample in tqdm(input_list, desc="Graph reasoning path"):
        try:
            results.append(process_sample(sample, direct_model, args))
        except Exception as exc:
            logger.error("Failed to process sample %s: %s", sample.get("index"), exc)
            results.append({
                "index": sample.get("index"),
                "uid": sample.get("uid"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "error": str(exc),
            })

    base_output_name = args.output_filename or f"graph_reasoning_path_{args.input_filename}"
    output_stem, output_ext = os.path.splitext(base_output_name)
    if not output_ext:
        output_ext = ".json"
    slurm_job_id = os.getenv("SLURM_JOB_ID", os.getenv("JOB_ID", "local"))
    output_filename = (
        f"{output_stem}_searchr1_graph_{args.evidence_setting}_{args.graph_prompt_mode}_"
        f"{slurm_job_id}_{len(results)}{output_ext}"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
