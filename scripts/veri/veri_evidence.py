import argparse
import json
import logging
import os
from typing import Any, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


class TripletEvidenceVerdict(BaseModel):
    document_supported: Literal["yes", "no"]


class TripletBatchItem(BaseModel):
    question_triplet: str
    document_supported: Literal["yes", "no"]


class TripletBatchVerdict(BaseModel):
    triplet_verifications: list[TripletBatchItem]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_source", type=str, choices=["retrieved", "gold"], default="retrieved")
    parser.add_argument("--retrieval_input_file", type=str, default=None)
    parser.add_argument("--gold_input_file_path", type=str, default=None)
    parser.add_argument("--graph_input_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--verification_target", type=str, choices=["triplet", "raw_question"], default="triplet")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_trials", type=int, default=3)
    parser.add_argument("--max_documents_chars", type=int, default=12000)
    return parser.parse_args()


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}")
    return data


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
    return result


def build_gold_documents(sample: dict[str, Any]) -> list[str]:
    gold_ids = sample.get("gold_id_list", [])
    gold_evidence_list = sample.get("gold_evidence_list", [])
    if not isinstance(gold_ids, list) or not isinstance(gold_evidence_list, list):
        return []

    documents = []
    for doc_id, evidence in zip(gold_ids, gold_evidence_list):
        if not isinstance(evidence, str):
            continue
        evidence_text = evidence.strip()
        if not evidence_text:
            continue
        if isinstance(doc_id, str) and doc_id.strip():
            documents.append(f"(Title: {doc_id.strip()}) {evidence_text}")
        else:
            documents.append(evidence_text)
    return documents


def get_question_graph(graph_sample: dict[str, Any]) -> dict[str, Any]:
    nested_graph = graph_sample.get("question_graph")
    if isinstance(nested_graph, dict):
        return nested_graph
    return graph_sample


def get_question_triplets(graph_sample: dict[str, Any]) -> tuple[list[str], list[str]]:
    question_graph = get_question_graph(graph_sample)
    definition_triples = normalize_string_list(question_graph.get("definition_triples", []))
    triples = normalize_string_list(question_graph.get("triples", []))
    return definition_triples, triples


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def build_documents_text(retrieved_documents: list[str], max_chars: int) -> str:
    if not retrieved_documents:
        return "(none)"
    joined = "\n\n".join(
        f"[Document {idx + 1}]\n{document}" for idx, document in enumerate(retrieved_documents)
    )
    return truncate_text(joined, max_chars)


def build_prompt(
    question: str,
    definition_triples: list[str],
    target_text: str,
    retrieved_documents_text: str,
    verification_target: str,
) -> str:
    definition_text = "\n".join(definition_triples) if definition_triples else "(none)"
    if verification_target == "raw_question":
        return (
            "You are verifying whether retrieved documents support answering a question.\n"
            "Read the question and the retrieved documents.\n"
            "Decide whether the retrieved documents contain enough evidence to answer the question correctly.\n\n"
            "Return JSON only with this schema:\n"
            "{\"document_supported\": \"yes\" or \"no\"}\n\n"
            "Rules:\n"
            "- Return \"yes\" only when the retrieved documents contain enough evidence to answer the full question.\n"
            "- Return \"no\" if the support is missing, ambiguous, or contradicted.\n"
            "- Do not include any explanation outside the JSON.\n\n"
            f"Question:\n{question}\n\n"
            f"Definition triples:\n{definition_text}\n\n"
            f"Retrieved documents:\n{retrieved_documents_text}\n"
        )
    return (
        "You are verifying whether retrieved documents support a question triplet.\n"
        "Read the question, definition triples, the target question triplet, and the retrieved documents.\n"
        "Decide whether the retrieved documents contain enough evidence to support the target question triplet.\n\n"
        "Return JSON only with this schema:\n"
        "{\"document_supported\": \"yes\" or \"no\"}\n\n"
        "Rules:\n"
        "- Return \"yes\" only when the retrieved documents explicitly support the target question triplet.\n"
        "- Return \"no\" if the support is missing, ambiguous, or contradicted.\n"
        "- Use the definition triples as constraints when interpreting placeholders such as (ENT1).\n"
        "- Do not include any explanation outside the JSON.\n\n"
        f"Question:\n{question}\n\n"
        f"Definition triples:\n{definition_text}\n\n"
        f"Target question triplet:\n{target_text}\n\n"
        f"Retrieved documents:\n{retrieved_documents_text}\n"
    )


def build_triplet_batch_prompt(
    question: str,
    definition_triples: list[str],
    question_triplets: list[str],
    retrieved_documents_text: str,
) -> str:
    definition_text = "\n".join(definition_triples) if definition_triples else "(none)"
    triplet_text = "\n".join(
        f"{idx + 1}. {triplet}" for idx, triplet in enumerate(question_triplets)
    ) if question_triplets else "(none)"
    return (
        "You are verifying whether retrieved documents support each question triplet.\n"
        "Read the question, definition triples, the list of question triplets, and the retrieved documents.\n"
        "For every question triplet, decide whether the retrieved documents explicitly support it.\n\n"
        "Return JSON only with this schema:\n"
        "{\"triplet_verifications\": [{\"question_triplet\": \"...\", \"document_supported\": \"yes\" or \"no\"}]}\n\n"
        "Rules:\n"
        "- Return exactly one output item for each input question triplet.\n"
        "- Copy each question_triplet string exactly.\n"
        "- Return \"yes\" only when the retrieved documents explicitly support that triplet.\n"
        "- Return \"no\" if the support is missing, ambiguous, or contradicted.\n"
        "- Use the definition triples as constraints when interpreting placeholders such as (ENT1).\n"
        "- Do not include any explanation outside the JSON.\n\n"
        f"Question:\n{question}\n\n"
        f"Definition triples:\n{definition_text}\n\n"
        f"Question triplets:\n{triplet_text}\n\n"
        f"Retrieved documents:\n{retrieved_documents_text}\n"
    )


def parse_verdict_from_text(text: str) -> Optional[TripletEvidenceVerdict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    json_block = text[start:end + 1]
    try:
        payload = json.loads(json_block)
    except json.JSONDecodeError:
        return None

    try:
        if hasattr(TripletEvidenceVerdict, "model_validate"):
            verdict = TripletEvidenceVerdict.model_validate(payload)
        else:
            verdict = TripletEvidenceVerdict.parse_obj(payload)
    except ValidationError:
        return None
    return verdict


def parse_triplet_batch_from_text(text: str) -> Optional[TripletBatchVerdict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    json_block = text[start:end + 1]
    try:
        payload = json.loads(json_block)
    except json.JSONDecodeError:
        return None

    try:
        if hasattr(TripletBatchVerdict, "model_validate"):
            verdict = TripletBatchVerdict.model_validate(payload)
        else:
            verdict = TripletBatchVerdict.parse_obj(payload)
    except ValidationError:
        return None
    return verdict


def request_verdict(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_trials: int,
) -> tuple[TripletEvidenceVerdict, str]:
    raw_output = ""
    for _ in range(max(1, max_trials)):
        raw_output = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON that matches the requested schema.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=32,
        ).choices[0].message.content or ""

        parsed = parse_verdict_from_text(raw_output)
        if parsed is not None:
            return parsed, raw_output

    return TripletEvidenceVerdict(document_supported="no"), raw_output


def request_triplet_batch_verdict(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_trials: int,
) -> tuple[TripletBatchVerdict, str]:
    raw_output = ""
    for _ in range(max(1, max_trials)):
        raw_output = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON that matches the requested schema.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=4096,
        ).choices[0].message.content or ""

        parsed = parse_triplet_batch_from_text(raw_output)
        if parsed is not None:
            return parsed, raw_output

    return TripletBatchVerdict(triplet_verifications=[]), raw_output


def aggregate_triplet_support(
    question_triplets: list[str],
    triplet_verifications: list[TripletBatchItem],
) -> str:
    if not question_triplets:
        return "no"
    returned_map = {item.question_triplet: item.document_supported for item in triplet_verifications}
    for question_triplet in question_triplets:
        if returned_map.get(question_triplet, "no") != "yes":
            return "no"
    return "yes"


def process_sample(
    evidence_sample: dict[str, Any],
    graph_sample: Optional[dict[str, Any]],
    client: OpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.evidence_source == "gold":
        retrieved_documents = build_gold_documents(evidence_sample)
        graph_source = evidence_sample
    else:
        retrieved_documents = normalize_string_list(evidence_sample.get("retrieved_documents", []))
        graph_source = graph_sample if graph_sample is not None else {}
    definition_triples, question_triples = get_question_triplets(graph_source)
    retrieved_documents_text = build_documents_text(retrieved_documents, args.max_documents_chars)

    if args.verification_target == "raw_question":
        targets = [evidence_sample.get("question", "")]
        for target_text in targets:
            prompt = build_prompt(
                question=evidence_sample.get("question", ""),
                definition_triples=definition_triples,
                target_text=target_text,
                retrieved_documents_text=retrieved_documents_text,
                verification_target=args.verification_target,
            )
            verdict, raw_output = request_verdict(
                client=client,
                model_name=args.model_name,
                prompt=prompt,
                max_trials=args.max_trials,
            )
            document_supported = verdict.document_supported
    else:
        prompt = build_triplet_batch_prompt(
            question=evidence_sample.get("question", ""),
            definition_triples=definition_triples,
            question_triplets=question_triples,
            retrieved_documents_text=retrieved_documents_text,
        )
        batch_verdict, raw_output = request_triplet_batch_verdict(
            client=client,
            model_name=args.model_name,
            prompt=prompt,
            max_trials=args.max_trials,
        )
        document_supported = aggregate_triplet_support(
            question_triplets=question_triples,
            triplet_verifications=batch_verdict.triplet_verifications,
        )

    return {
        "index": evidence_sample.get("index", graph_source.get("index")),
        "uid": evidence_sample.get("uid", graph_source.get("uid")),
        "question": evidence_sample.get("question", graph_source.get("question")),
        "answer": evidence_sample.get("answer", graph_source.get("answer")),
        "answer_matches_gold": evidence_sample.get("answer_matches_gold"),
        "retrieved_documents": retrieved_documents,
        "evidence_source": args.evidence_source,
        "verification_target": args.verification_target,
        "document_supported": document_supported,
        "question_graph": {
            "definition_triples": definition_triples,
            "triples": question_triples,
        },
        "prompt_output": raw_output,
    }


def validate_pairing(
    evidence_sample: dict[str, Any],
    graph_sample: dict[str, Any],
    position: int,
) -> None:
    retrieval_index = evidence_sample.get("index")
    graph_index = graph_sample.get("index")
    if retrieval_index is not None and graph_index is not None and retrieval_index != graph_index:
        raise ValueError(
            f"Mismatched sample index at position {position}: "
            f"retrieval index={retrieval_index}, graph index={graph_index}"
        )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    client = OpenAI(api_key=api_key)

    if args.evidence_source == "gold":
        if not args.gold_input_file_path:
            raise ValueError("--gold_input_file_path is required when --evidence_source gold")
        evidence_samples = load_json(args.gold_input_file_path)
        evidence_input_path = args.gold_input_file_path
        graph_samples = None
    else:
        if not args.retrieval_input_file:
            raise ValueError("--retrieval_input_file is required when --evidence_source retrieved")
        if not args.graph_input_file:
            raise ValueError("--graph_input_file is required when --evidence_source retrieved")
        evidence_samples = load_json(args.retrieval_input_file)
        evidence_input_path = args.retrieval_input_file
        graph_samples = load_json(args.graph_input_file)

        if len(evidence_samples) != len(graph_samples):
            raise ValueError(
                f"Input length mismatch: evidence={len(evidence_samples)}, graph={len(graph_samples)}"
            )

    if args.max_samples is not None:
        evidence_samples = evidence_samples[:args.max_samples]
        if graph_samples is not None:
            graph_samples = graph_samples[:args.max_samples]

    logger.info(
        "Loaded retrieval samples=%d, graph samples=%d, model=%s",
        len(evidence_samples),
        len(graph_samples) if graph_samples is not None else len(evidence_samples),
        args.model_name,
    )

    output_samples = []
    if graph_samples is None:
        iterator = ((evidence_sample, None) for evidence_sample in evidence_samples)
    else:
        iterator = zip(evidence_samples, graph_samples)

    for position, (evidence_sample, graph_sample) in enumerate(
        tqdm(iterator, total=len(evidence_samples), desc="Verify evidence")
    ):
        if graph_sample is not None:
            validate_pairing(evidence_sample, graph_sample, position)
        output_samples.append(process_sample(evidence_sample, graph_sample, client, args))

    retrieval_stem = os.path.splitext(os.path.basename(evidence_input_path))[0]
    graph_stem = os.path.splitext(os.path.basename(args.graph_input_file))[0]
    output_name = args.output_filename or f"veri_evidence_{retrieval_stem}_{graph_stem}.json"
    output_path = os.path.join(args.output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_samples, handle, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
