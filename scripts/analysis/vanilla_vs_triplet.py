#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model_library.llm_clients import GPT, Qwen


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


ALIGN_SYSTEM_PROMPT = (
    "You evaluate whether a vanilla reasoning trace covers the required logical reasoning steps in a question-triplet plan. "
    "Be strict: if the trace shows retrieval drift to the wrong entity, answer no. "
    "Return only JSON with one key: {\"align\": \"yes\"} or {\"align\": \"no\"}."
)


class AlignDecision(BaseModel):
    align: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_1", type=str, required=True, help="Vanilla/SearchR1 result JSON")
    parser.add_argument("--input_file_2", type=str, required=True, help="Question-graph JSON")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--reasoning_source",
        type=str,
        choices=["think", "subquery", "think+subquery"],
        default="think",
        help="Alignment source: think uses reasoning trace, subquery uses retrieval_turns queries, think+subquery uses both.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default="vanilla_vs_triplet.json")
    return parser.parse_args()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return data


def init_model(model_name: str) -> Any:
    model_name_lower = model_name.lower()
    if model_name_lower.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT models.")
        client = OpenAI(api_key=api_key)
        return GPT(model_name, client)

    if model_name_lower.startswith("qwen") or model_name_lower.startswith("meta-llama") or model_name_lower.startswith("llama"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        return Qwen(model, tokenizer)

    raise ValueError(f"Unsupported model: {model_name}")


def extract_json_block(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start:end + 1]


def parse_align_output(raw_output: str) -> str:
    normalized = raw_output.strip().lower()
    if normalized in {"yes", "no"}:
        return normalized

    try:
        parsed = json.loads(extract_json_block(raw_output))
        try:
            validated = AlignDecision.model_validate(parsed)
        except AttributeError:
            validated = AlignDecision.parse_obj(parsed)
        align = str(validated.align).strip().lower()
        if align in {"yes", "no"}:
            return align
    except json.JSONDecodeError:
        pass

    return "no"


def collect_think_trace(sample: Dict[str, Any]) -> List[str]:
    steps = sample.get("reasoning_steps", [])
    if isinstance(steps, list) and steps:
        return [str(step).strip() for step in steps if str(step).strip()]
    retrieval_info = sample.get("retrieval_info", {}) if isinstance(sample.get("retrieval_info"), dict) else {}
    steps = retrieval_info.get("searchr1_reasoning_steps", [])
    if isinstance(steps, list) and steps:
        return [str(step).strip() for step in steps if str(step).strip()]
    retrieval_turns = retrieval_info.get("retrieval_turns", [])
    outputs: List[str] = []
    if isinstance(retrieval_turns, list):
        for turn in retrieval_turns:
            if not isinstance(turn, dict):
                continue
            model_output = str(turn.get("model_output", "")).strip()
            if model_output:
                outputs.append(model_output)
    return outputs


def collect_subqueries(sample: Dict[str, Any]) -> List[str]:
    retrieval_info = sample.get("retrieval_info", {}) if isinstance(sample.get("retrieval_info"), dict) else {}
    retrieval_turns = retrieval_info.get("retrieval_turns", [])
    queries: List[str] = []
    if isinstance(retrieval_turns, list):
        for turn in retrieval_turns:
            if not isinstance(turn, dict):
                continue
            query = str(turn.get("query", "")).strip()
            if query:
                queries.append(query)
    return queries


def build_question_graph_trace(sample: Dict[str, Any]) -> str:
    question_graph = sample.get("question_graph", {})
    definition_triples = question_graph.get("definition_triples", []) or []
    triples = question_graph.get("triples", []) or []

    parts: List[str] = []
    parts.append("Definition triples:")
    parts.extend(definition_triples if definition_triples else ["(none)"])
    parts.append("")
    parts.append("Triples:")
    parts.extend(triples if triples else ["(none)"])
    return "\n".join(parts)


def build_align_prompt(
    question: str,
    vanilla_trace: str,
    question_graph_trace: str,
    reasoning_source: str,
) -> str:
    if reasoning_source == "think":
        source_label = "Vanilla reasoning trace"
    elif reasoning_source == "subquery":
        source_label = "Search subqueries"
    else:
        source_label = "Vanilla reasoning trace and search subqueries"

    if reasoning_source == "subquery":
        return (
            f"Question:\n{question}\n\n"
            f"{source_label}:\n{vanilla_trace or '(none)'}\n\n"
            f"Required question-triplet plan (Blueprint):\n{question_graph_trace}\n\n"
            "Task: Decide whether the subquery sequence covers the required relation steps in the Blueprint.\n"
            "You are judging only the retrieval plan structure. Do NOT judge entity correctness or drift.\n\n"
            "Rules for Align \"yes\":\n"
            "1. The subqueries cover each required relation type in the Blueprint (each relation in the triplets).\n"
            "2. For relations that appear multiple times, the subqueries include at least the same number of relation-focused queries.\n"
            "3. Minor wording differences are allowed, but required steps must be present.\n\n"
            "Rules for Align \"no\":\n"
            "1. Any required relation step from the Blueprint is missing in the subqueries.\n"
            "2. The subqueries skip required slots or collapse multiple required steps into one unrelated query.\n"
            "3. The subquery sequence follows an unrelated structure.\n\n"
            "If unsure, answer \"no\".\n\n"
            'Output format: {"align": "yes"} or {"align": "no"}'
        )
    if reasoning_source == "think+subquery":
        return (
            f"Question:\n{question}\n\n"
            f"{source_label}:\n{vanilla_trace or '(none)'}\n\n"
            f"Required question-triplet plan (Blueprint):\n{question_graph_trace}\n\n"
            "Task: Decide whether the combined vanilla reasoning trace and subquery sequence cover the Blueprint.\n\n"
            "Judge both inputs together:\n"
            "1. The reasoning trace should show the needed logical steps, entities, relations, and intermediate facts.\n"
            "2. The subqueries should cover each required relation step in the Blueprint.\n"
            "3. Minor wording or ordering differences are allowed.\n\n"
            "Rules for Align \"yes\":\n"
            "1. The reasoning trace covers the required reasoning path without missing key intermediate steps.\n"
            "2. The subquery sequence covers each required relation-focused retrieval step in the Blueprint.\n"
            "3. Neither input shows a clearly unrelated structure.\n\n"
            "Rules for Align \"no\":\n"
            "1. The reasoning trace misses key logical steps, guesses unsupported intermediate facts, or drifts to the wrong entity.\n"
            "2. The subquery sequence misses required relation steps, skips required slots, or follows an unrelated retrieval structure.\n"
            "3. Only one of the two inputs aligns but the other does not.\n\n"
            "Focus on reasoning and retrieval-plan coverage, not final answer correctness.\n"
            "If the trace switches to a different entity, even a similar-name entity, treat that as retrieval drift and answer no.\n"
            "If unsure, answer \"no\".\n\n"
            'Output format: {"align": "yes"} or {"align": "no"}'
        )
    return (
        f"Question:\n{question}\n\n"
        f"{source_label}:\n{vanilla_trace or '(none)'}\n\n"
        f"Required question-triplet plan (Blueprint):\n{question_graph_trace}\n\n"
        "Task: Decide whether the Vanilla reasoning trace covers the logical reasoning steps required by the Blueprint.\n\n"
        '- Align "yes": The trace shows the required logical reasoning steps, including the needed entities, relations, and intermediate facts, even if the order or wording differs.\n'
        '- Align "no": The trace misses required logical steps, skips key intermediate facts, jumps to an answer without the needed reasoning, follows a different reasoning path, or shows retrieval drift to the wrong entity.\n\n'
        "Focus only on the reasoning process. Ignore final answer correctness.\n"
        "Do not require the same wording or the same step order. Judge whether the necessary logical reasoning content is present.\n"
        "If the trace switches to a different entity, even a similar-name entity, treat that as retrieval drift and answer no.\n"
        "If the trace only guesses an intermediate fact instead of resolving it, answer no.\n\n"
        'Output format: {"align": "yes"} or {"align": "no"}'
    )


def compare_pair(
    vanilla_sample: Dict[str, Any],
    graph_sample: Dict[str, Any],
    model_client: Any,
    reasoning_source: str,
) -> Dict[str, Any]:
    think_outputs = collect_think_trace(vanilla_sample)
    subquery_outputs = collect_subqueries(vanilla_sample)
    if reasoning_source == "subquery":
        model_output: Any = subquery_outputs
        vanilla_trace = "\n\n".join(subquery_outputs)
    elif reasoning_source == "think+subquery":
        model_output = {
            "think": think_outputs,
            "subquery": subquery_outputs,
        }
        think_trace = "\n\n".join(think_outputs) or "(none)"
        subquery_trace = "\n\n".join(subquery_outputs) or "(none)"
        vanilla_trace = f"[Thinking]\n{think_trace}\n\n[Subqueries]\n{subquery_trace}"
    else:
        model_output = think_outputs
        vanilla_trace = "\n\n".join(think_outputs)
    question_graph = graph_sample.get("question_graph", {})
    question_graph_trace = build_question_graph_trace(graph_sample)
    prompt = build_align_prompt(
        question=graph_sample.get("question", ""),
        vanilla_trace=vanilla_trace,
        question_graph_trace=question_graph_trace,
        reasoning_source=reasoning_source,
    )
    raw_output = model_client.generate(
        user_message=prompt,
        system_message=ALIGN_SYSTEM_PROMPT,
        max_tokens=256,
        temperature=0.0,
    )
    align = parse_align_output(raw_output)

    return {
        "index": graph_sample.get("index"),
        "uid": graph_sample.get("uid"),
        "num_hops": graph_sample.get("num_hops"),
        "question": graph_sample.get("question"),
        "answer": graph_sample.get("answer"),
        "question_graph": question_graph,
        "model_output": model_output,
        "predicted_answer": vanilla_sample.get("predicted_answer"),
        "answer_matches_gold": vanilla_sample.get("answer_matches_gold"),
        "reasoning_source": reasoning_source,
        "align": align,
    }


def main() -> None:
    args = parse_args()
    vanilla_list = load_json(args.input_file_1)
    graph_list = load_json(args.input_file_2)

    vanilla_by_index = {sample.get("index"): sample for sample in vanilla_list}
    graph_by_index = {sample.get("index"): sample for sample in graph_list}
    shared_indices = sorted(idx for idx in graph_by_index if idx in vanilla_by_index)

    if args.max_samples is not None:
        shared_indices = shared_indices[: args.max_samples]

    model_client = init_model(args.base_model_name)
    results: List[Dict[str, Any]] = []

    for index in shared_indices:
        vanilla_sample = vanilla_by_index[index]
        graph_sample = graph_by_index[index]
        try:
            if vanilla_sample.get("uid") != graph_sample.get("uid"):
                logger.warning(
                    "UID mismatch at index=%s: vanilla=%s graph=%s",
                    index,
                    vanilla_sample.get("uid"),
                    graph_sample.get("uid"),
                )
            results.append(compare_pair(vanilla_sample, graph_sample, model_client, args.reasoning_source))
        except Exception as exc:
            logger.error("Failed to compare index=%s: %s", index, exc)
            results.append(
                {
                    "index": graph_sample.get("index"),
                    "uid": graph_sample.get("uid"),
                    "num_hops": graph_sample.get("num_hops"),
                    "question": graph_sample.get("question"),
                    "answer": graph_sample.get("answer"),
                    "question_graph": graph_sample.get("question_graph", {}),
                    "model_output": (
                        collect_subqueries(vanilla_sample)
                        if args.reasoning_source == "subquery"
                        else {
                            "think": collect_think_trace(vanilla_sample),
                            "subquery": collect_subqueries(vanilla_sample),
                        }
                        if args.reasoning_source == "think+subquery"
                        else collect_think_trace(vanilla_sample)
                    ),
                    "predicted_answer": vanilla_sample.get("predicted_answer"),
                    "answer_matches_gold": vanilla_sample.get("answer_matches_gold"),
                    "reasoning_source": args.reasoning_source,
                    "align": "no",
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_filename
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    logger.info("Saved %d results to %s", len(results), output_path)


if __name__ == "__main__":
    main()
