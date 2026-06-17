#!/usr/bin/env python3
"""Question-triple-level graph/document/think verification.

This is a narrower variant of compare_all.py.  For each sample, it verifies
each non-definition question graph triple against the sample's full document
evidence and full SearchR1 think trace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare_all import (  # noqa: E402
    attach_question_graph,
    build_graph_lookup,
    extract_think,
    get_gold_answers,
    get_openai_client,
    get_retrieval_info,
    get_searchr1_answer,
    get_searchr1_trace,
    get_turns,
    has_trace_output,
    judge_temperature,
    load_dotenv,
    load_existing_output,
    load_records,
    maybe_sleep,
    pydantic_to_dict,
    sample_id,
    save_output,
    truncate_text,
)

LOG_SAMPLE_LIMIT = 5
JUDGE_MAX_TOKENS = 2048


class DocumentJudgeOutput(BaseModel):
    document_supported: bool = Field(
        description="Whether the document text supports or helps fill the selected question graph triple."
    )


class ThinkJudgeOutput(BaseModel):
    think_aligned: bool = Field(
        description="Whether the think trace is aligned with the selected question graph triple."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify each question graph triple against documents and think traces."
    )
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file with SearchR1 traces")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--question-graph-input",
        default=None,
        help="Optional JSON/JSONL file containing question_graph fields matched by uid or index",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI judge model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-doc-chars", type=int, default=12000)
    parser.add_argument("--max-think-chars", type=int, default=6000)
    parser.add_argument("--max-triple-chars", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Skip sample ids already in output")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Include the original sample under source_sample in the output",
    )
    return parser.parse_args()


def get_question_graph(sample: Dict[str, Any]) -> Dict[str, Any]:
    graph = sample.get("question_graph")
    if isinstance(graph, dict):
        return graph
    graph = sample.get("cot_graph")
    if isinstance(graph, dict):
        return graph
    return {
        "definition_triples": sample.get("cot_def_triples", []),
        "triples": sample.get("cot_triples", []),
    }


def get_question_triples(sample: Dict[str, Any]) -> List[str]:
    graph = get_question_graph(sample)
    triples = graph.get("triples", [])
    if not isinstance(triples, list):
        return []
    return [str(triple).strip() for triple in triples if str(triple).strip()]


def get_documents_and_thinks(sample: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[str], List[str], str, str]:
    retrieval_info = get_retrieval_info(sample)
    trace = get_searchr1_trace(retrieval_info)
    turns = get_turns(trace, retrieval_info)

    documents: List[str] = []
    thinks: List[str] = []

    initial_thinking = trace.get("initial_thinking") or retrieval_info.get("searchr1_initial_thinking")
    if initial_thinking:
        thinks.append(str(initial_thinking))

    for turn in turns:
        model_output = str(turn.get("model_output", ""))
        think = extract_think(model_output)
        if think:
            thinks.append(think)

        search_results = turn.get("search_results", [])
        if isinstance(search_results, list):
            documents.extend(str(doc) for doc in search_results if str(doc).strip())
        elif search_results:
            documents.append(str(search_results))

    total_search_results = (
        trace.get("total_search_results")
        or retrieval_info.get("searchr1_total_search_results")
        or sample.get("retrieved_documents")
        or []
    )
    if isinstance(total_search_results, dict):
        total_search_results = total_search_results.get("document_text", [])
    if isinstance(total_search_results, list):
        documents.extend(str(doc) for doc in total_search_results if str(doc).strip())
    elif total_search_results:
        documents.append(str(total_search_results))

    documents = list(dict.fromkeys(documents))
    thinks = list(dict.fromkeys(thinks))

    document_text = truncate_text(documents, args.max_doc_chars)
    think_text = truncate_text(thinks, args.max_think_chars)
    output_documents = document_text.splitlines() if document_text else []
    output_thinks = think_text.splitlines() if think_text else []
    return output_documents, output_thinks, document_text, think_text


def build_document_prompt(
    sample: Dict[str, Any],
    question_triple: str,
    document_text: str,
    args: argparse.Namespace,
) -> str:
    question_triple = truncate_text(question_triple, args.max_triple_chars)
    return f"""
You are verifying whether retrieved evidence supports one question-graph triple.

Rules:
- Judge only from the provided fields.
- Do not use outside knowledge.
- The question-graph triple may contain placeholders such as (ENT1), (ENT2), etc.
- Treat document_supported as a slot-filling evidence check for the selected triple.

Question-graph triple:
{question_triple}

Task: document_supported
Return true only when the document text explicitly supports the selected triple:
- it identifies a concrete value for an unresolved ENT placeholder, or
- it verifies the selected triple's entity/relation/fact.

Return false when:
- support is missing, ambiguous, or contradicted,
- the document only overlaps in surface words,
- the document supports a different triple but not this selected triple,
- the document does not help fill or verify this selected triple.

If unsure, return false.

Document text:
{document_text}
""".strip()


def build_think_prompt(
    sample: Dict[str, Any],
    question_triple: str,
    think_text: str,
    args: argparse.Namespace,
) -> str:
    question_triple = truncate_text(question_triple, args.max_triple_chars)
    return f"""
You are verifying whether a model's reasoning trace follows one question-graph triple.

Rules:
- Judge only from the provided fields.
- Do not use outside knowledge.
- The question-graph triple may contain placeholders such as (ENT1), (ENT2), etc.
- Treat think_aligned as a reasoning-path coverage check for the selected triple.

Question-graph triple:
{question_triple}

Task: think_aligned
Return true only when the think trace covers the selected triple's required reasoning content:
- it explicitly seeks, infers, or uses the same missing entity/fact/relation represented by the selected triple, and
- it stays consistent with the entity/relation/fact required by the selected triple.

Return false when:
- the think trace is empty,
- it skips this selected triple's hop,
- it follows a different entity/relation/hop,
- it uses a fact incompatible with this selected triple,
- it guesses an intermediate fact instead of resolving it,
- it drifts to the wrong entity, even a similar-name entity,
- it only states the final answer without reasoning for this selected triple.

Do not require the same wording or step order. Focus on whether the necessary reasoning content is present.
If unsure, return false.

SearchR1 think trace:
{think_text}
""".strip()


def judge_structured(
    client: Any,
    prompt: str,
    args: argparse.Namespace,
    output_model: Any,
    check_name: str,
    retries: int = 2,
) -> Dict[str, Any]:
    last_error = ""
    temperature = judge_temperature(args.model, args.temperature)
    for attempt in range(retries + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=args.model,
                temperature=temperature,
                max_tokens=JUDGE_MAX_TOKENS,
                response_format=output_model,
                messages=[
                    {"role": "system", "content": "Answer using the required structured output."},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                usage = getattr(response, "usage", None)
                raise ValueError(f"Empty parsed judge response; finish_reason={finish_reason!r}; usage={usage!r}")
            return pydantic_to_dict(parsed)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            print(f"[judge error] {check_name}: {last_error}", file=sys.stderr, flush=True)
            if attempt < retries:
                time.sleep(1.0 + attempt)

    fallback = {name: False for name in output_model.model_fields}
    fallback["error"] = last_error
    return fallback


def bool_label(value: Any) -> int:
    return int(bool(value))


def evaluate_sample(
    client: Any,
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    retrieval_info = get_retrieval_info(sample)
    trace = get_searchr1_trace(retrieval_info)
    question_triples = get_question_triples(sample)
    documents, thinks, document_text, think_text = get_documents_and_thinks(sample, args)
    gold_answers = get_gold_answers(sample)
    searchr1_answer = get_searchr1_answer(retrieval_info, trace)

    result: Dict[str, Any] = {
        "index": sample.get("index"),
        "uid": sample.get("uid"),
        "question": sample.get("question", ""),
        "gold_answers": gold_answers,
        "searchr1_answer": searchr1_answer,
        "num_question_triples": len(question_triples),
        "question_verifications": [],
        "doc_think_ok": {
            "sample_pair": "0/0",
            "question_pairs": "",
            "doc_ok_all": False,
            "think_ok_all": False,
        },
        "document_text": documents,
        "think_text": thinks,
    }

    if has_trace_output(trace, retrieval_info, sample):
        result["searchr1_trace_summary"] = {
            "num_turns": len(get_turns(trace, retrieval_info)),
            "has_documents": bool(document_text),
            "has_think": bool(think_text),
        }

    pairs: List[str] = []
    doc_oks: List[bool] = []
    think_oks: List[bool] = []

    for idx, question_triple in enumerate(question_triples):
        document_prompt = build_document_prompt(
            sample,
            question_triple,
            document_text,
            args,
        )
        document_output = judge_structured(
            client,
            document_prompt,
            args,
            DocumentJudgeOutput,
            "document_supported",
        )
        maybe_sleep(args)

        think_prompt = build_think_prompt(
            sample,
            question_triple,
            think_text,
            args,
        )
        think_output = judge_structured(
            client,
            think_prompt,
            args,
            ThinkJudgeOutput,
            "think_aligned",
        )
        maybe_sleep(args)

        doc_ok = bool(document_output.get("document_supported"))
        think_ok = bool(think_output.get("think_aligned"))
        pair = f"{bool_label(doc_ok)}/{bool_label(think_ok)}"
        pairs.append(pair)
        doc_oks.append(doc_ok)
        think_oks.append(think_ok)

        result["question_verifications"].append({
            "question_index": idx,
            "question_triple": question_triple,
            "document_supported": doc_ok,
            "think_aligned": think_ok,
            "pair": pair,
        })

    doc_ok_all = bool(doc_oks) and all(doc_oks)
    think_ok_all = bool(think_oks) and all(think_oks)
    result["doc_think_ok"] = {
        "sample_pair": f"{bool_label(doc_ok_all)}/{bool_label(think_ok_all)}",
        "question_pairs": "|".join(pairs),
        "doc_ok_all": doc_ok_all,
        "think_ok_all": think_ok_all,
    }

    if args.keep_source:
        result["source_sample"] = sample

    return result


def log_sample_result(result: Dict[str, Any], count: int) -> None:
    message = (
        f"\n[sample output {count}/{LOG_SAMPLE_LIMIT}]\n"
        f"{json.dumps(result, ensure_ascii=False, indent=2)}"
    )
    if tqdm is not None:
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr, flush=True)


def main() -> int:
    args = parse_args()
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    if not os.getenv("SKIML_API_KEY"):
        print("SKIML_API_KEY is not set.", file=sys.stderr)
        return 2

    graph_lookup = {}
    if args.question_graph_input:
        graph_lookup = build_graph_lookup(load_records(Path(args.question_graph_input)))

    output_path = Path(args.output)
    records = load_records(Path(args.input))
    existing_results, done_ids = load_existing_output(output_path) if args.resume else ([], set())
    selected = records[args.start :]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]

    client = get_openai_client()
    results = list(existing_results)
    logged_sample_count = 0

    progress = tqdm(
        selected,
        total=len(selected),
        desc="compare_one",
        dynamic_ncols=True,
    ) if tqdm is not None else selected

    for offset, sample in enumerate(progress, start=1):
        sid = sample_id(sample)
        if sid in done_ids:
            continue
        if tqdm is not None:
            progress.set_postfix_str(f"sample={sid}")
        else:
            print(f"[{offset}/{len(selected)}] evaluating sample {sid}", file=sys.stderr, flush=True)

        sample = attach_question_graph(sample, graph_lookup)
        result = evaluate_sample(client, sample, args)
        results.append(result)
        done_ids.add(sid)
        save_output(output_path, results, args)

        if logged_sample_count < LOG_SAMPLE_LIMIT:
            logged_sample_count += 1
            log_sample_result(result, logged_sample_count)

    save_output(output_path, results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
