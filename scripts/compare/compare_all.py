#!/usr/bin/env python3
"""Evidence-grounded verification for SearchR1 traces.

This script evaluates SearchR1 retrieval turns with an OpenAI judge.  The
evaluation is turn-local by default: each turn is judged only against its own
query, model output, and retrieved documents.  The only cross-turn check is
document_vs_next_think, which compares turn i documents with turn i+1 think.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from prompt import build_prompt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ALLOWED_LABELS = {
    "question_graph_vs_document": {"true", "false"},
    "question_graph_vs_think": {"true", "false"},
    "think_vs_query": {"true", "false"},
    "document_vs_next_think": {"true", "false", "hallu", "fail"},
    "searchr1_final_answer_correct": {"true", "false"},
}
LOG_SAMPLE_LIMIT = 5
JUDGE_MAX_TOKENS = 2048


class JudgeOutput(BaseModel):
    label: str = Field(description="One allowed label for the requested verification check.")


def judge_temperature(model: str, requested_temperature: float) -> float:
    """gpt-5 chat models reject temperature=0 in this LiteLLM setup."""
    if model.startswith("gpt-5"):
        return 1.0
    return requested_temperature


def pydantic_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def pydantic_schema(model_cls: Any) -> Dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def validate_pydantic(model_cls: Any, data: Dict[str, Any]) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify SearchR1 traces with evidence-grounded GPT judging."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input .json or .jsonl file with existing SearchR1 traces. Required when --trace-source=file.",
    )
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
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between API calls")
    parser.add_argument("--max-doc-chars", type=int, default=12000)
    parser.add_argument("--max-graph-chars", type=int, default=6000)
    parser.add_argument("--max-think-chars", type=int, default=4000)
    parser.add_argument("--max-query-chars", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Skip indices already in output")
    parser.add_argument(
        "--trace-source",
        choices=["file", "searchr1"],
        default="file",
        help="file: use SearchR1 traces from input file. searchr1: run SearchR1 for every sample, then compare.",
    )
    parser.add_argument(
        "--rerun-empty-searchr1",
        action="store_true",
        help="When --trace-source=file, rerun SearchR1 for samples with empty retrieval turns or search results.",
    )
    parser.add_argument(
        "--searchr1-model-id",
        default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        help="HuggingFace model id for SearchR1 reruns",
    )
    parser.add_argument("--searchr1-retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--searchr1-max-turns", type=int, default=4)
    parser.add_argument("--searchr1-top-k", type=int, default=3)
    parser.add_argument("--searchr1-max-new-tokens", type=int, default=500)
    parser.add_argument("--searchr1-temperature", type=float, default=1.0)
    parser.add_argument("--searchr1-device", default=None)
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Include the original sample under source_sample in the output",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path must be a JSON/JSONL file, not a directory: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "results"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported input JSON structure: {path}")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_existing_output(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        return [], set()

    done = set()
    for item in results:
        sample_id = item.get("uid", item.get("index"))
        if sample_id is not None:
            done.add(str(sample_id))
    return results, done


def build_graph_lookup(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in ("uid", "index"):
            value = record.get(key)
            if value is not None:
                lookup[f"{key}:{value}"] = record
    return lookup


def attach_question_graph(sample: Dict[str, Any], graph_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not graph_lookup or sample.get("question_graph") or sample.get("cot_graph") or sample.get("cot_triples"):
        return sample

    match = None
    if sample.get("uid") is not None:
        match = graph_lookup.get(f"uid:{sample['uid']}")
    if match is None and sample.get("index") is not None:
        match = graph_lookup.get(f"index:{sample['index']}")
    if match is None:
        return sample

    merged = dict(sample)
    for key in ("question_graph", "cot_graph", "cot_triples", "cot_def_triples"):
        if match.get(key) and not merged.get(key):
            merged[key] = match[key]
    return merged


def save_output(path: Path, results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def log_sample_result(result: Dict[str, Any], count: int) -> None:
    message = (
        f"\n[sample output {count}/{LOG_SAMPLE_LIMIT}]\n"
        f"{json.dumps(result, ensure_ascii=False, indent=2)}"
    )
    if tqdm is not None:
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr, flush=True)


def truncate_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = "\n".join(str(x) for x in value)
    else:
        text = str(value)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.I)
    return matches[-1].strip() if matches else ""


def extract_think(model_output: str) -> str:
    return extract_tag(model_output, "think")


def normalize_answer(value: Any) -> str:
    return str(value or "").strip()


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


def graph_to_text(graph: Any, max_chars: int) -> str:
    if isinstance(graph, dict):
        parts = []
        definition_triples = graph.get("definition_triples", graph.get("def_triples", []))
        triples = graph.get("triples", [])
        if definition_triples:
            parts.append("Definition triples:\n" + truncate_text(definition_triples, max_chars))
        if triples:
            parts.append("Triples:\n" + truncate_text(triples, max_chars))
        if parts:
            return truncate_text("\n\n".join(parts), max_chars)
    return truncate_text(graph, max_chars)


def get_retrieval_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        sample.get("retrieval_info"),
        sample.get("searchr1_retrieval_info"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate

    for process in sample.get("verification_process", []) or []:
        if isinstance(process, dict) and isinstance(process.get("searchr1_retrieval_info"), dict):
            return process["searchr1_retrieval_info"]

    return {}


def get_searchr1_trace(retrieval_info: Dict[str, Any]) -> Dict[str, Any]:
    trace = retrieval_info.get("searchr1_trace")
    if isinstance(trace, dict):
        return trace
    return retrieval_info


def get_searchr1_answer(retrieval_info: Dict[str, Any], trace: Dict[str, Any]) -> str:
    for key in ("searchr1_answer", "predicted_answer", "answer"):
        if retrieval_info.get(key) not in (None, ""):
            return normalize_answer(retrieval_info.get(key))
        if trace.get(key) not in (None, ""):
            return normalize_answer(trace.get(key))
    return ""


def get_gold_answers(sample: Dict[str, Any]) -> List[str]:
    answers = []
    for key in ("answer", "gold_answer"):
        if sample.get(key) not in (None, ""):
            answers.append(normalize_answer(sample[key]))
    aliases = sample.get("answer_aliases", sample.get("aliases", []))
    if isinstance(aliases, list):
        answers.extend(normalize_answer(x) for x in aliases if normalize_answer(x))
    elif aliases:
        answers.append(normalize_answer(aliases))
    return list(dict.fromkeys(x for x in answers if x))


def get_turns(trace: Dict[str, Any], retrieval_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = trace.get("retrieval_turns")
    if not isinstance(turns, list):
        turns = retrieval_info.get("retrieval_turns")
    return turns if isinstance(turns, list) else []


def get_searchr1_runner(args: argparse.Namespace) -> Any:
    from search_r1 import SearchR1Inference

    return SearchR1Inference(
        model_id=args.searchr1_model_id,
        retriever_url=args.searchr1_retriever_url,
        max_turns=args.searchr1_max_turns,
        max_new_tokens=args.searchr1_max_new_tokens,
        temperature=args.searchr1_temperature,
        topk=args.searchr1_top_k,
        device=args.searchr1_device,
    )


def has_empty_trace(sample: Dict[str, Any]) -> bool:
    retrieval_info = get_retrieval_info(sample)
    trace = get_searchr1_trace(retrieval_info)
    turns = get_turns(trace, retrieval_info)
    total_search_results = (
        trace.get("total_search_results")
        or retrieval_info.get("searchr1_total_search_results")
        or sample.get("retrieved_documents")
        or []
    )
    return not turns or not total_search_results


def normalize_doc_title(title: Any) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip().lower())


def extract_doc_title(document_text: str) -> str:
    match = re.match(r"\(Title:\s*(.*?)\)\s*", str(document_text or ""), flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def annotate_search_results(
    search_results: Any,
    gold_id_list: Any,
    is_gold_list: Any = None,
) -> Dict[str, List[Any]]:
    if not isinstance(search_results, list):
        return {"document_text": [], "is_gold": []}
    gold_titles = {normalize_doc_title(title) for title in (gold_id_list or [])}
    document_texts = []
    is_gold = []
    use_given_gold = isinstance(is_gold_list, list) and len(is_gold_list) == len(search_results)
    for idx, document_text in enumerate(search_results):
        title = extract_doc_title(str(document_text))
        document_texts.append(str(document_text))
        if use_given_gold:
            is_gold.append(int(bool(is_gold_list[idx])))
        else:
            is_gold.append(int(normalize_doc_title(title) in gold_titles))
    return {
        "document_text": document_texts,
        "is_gold": is_gold,
    }


def sanitize_searchr1_trace(
    trace: Dict[str, Any],
    retrieval_info: Dict[str, Any],
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    total_search_results = (
        trace.get("total_search_results")
        or retrieval_info.get("searchr1_total_search_results")
        or sample.get("retrieved_documents")
        or []
    )
    reasoning_steps = (
        trace.get("reasoning_steps")
        or retrieval_info.get("searchr1_reasoning_steps")
        or sample.get("reasoning_steps")
        or []
    )
    return {
        "prompt": trace.get("prompt", retrieval_info.get("searchr1_prompt", "")),
        "reasoning_steps": reasoning_steps,
        "total_search_results": annotate_search_results(
            total_search_results,
            sample.get("gold_id_list", []),
            retrieval_info.get("is_gold_list"),
        ),
    }


def has_trace_output(
    trace: Dict[str, Any],
    retrieval_info: Dict[str, Any],
    sample: Dict[str, Any],
) -> bool:
    return any((
        trace.get("prompt"),
        retrieval_info.get("searchr1_prompt"),
        trace.get("reasoning_steps"),
        retrieval_info.get("searchr1_reasoning_steps"),
        sample.get("reasoning_steps"),
        trace.get("total_search_results"),
        retrieval_info.get("searchr1_total_search_results"),
        sample.get("retrieved_documents"),
    ))


def attach_searchr1_result(sample: Dict[str, Any], search_info: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(sample)
    old_retrieval_info = merged.get("retrieval_info")
    retrieval_info = dict(old_retrieval_info) if isinstance(old_retrieval_info, dict) else {}

    search_info = dict(search_info)
    search_info.pop("initial_thinking", None)
    search_info.pop("last_search_results_list", None)

    retrieval_info.update({
        "query": search_info.get("question", merged.get("question", "")),
        "full_response": search_info.get("full_response", ""),
        "searchr1_answer": search_info.get("predicted_answer", ""),
        "searchr1_reasoning_path": search_info.get("reasoning_path", ""),
        "searchr1_reasoning_steps": search_info.get("reasoning_steps", []),
        "searchr1_prompt": search_info.get("prompt", ""),
        "searchr1_question": search_info.get("question", merged.get("question", "")),
        "num_turns": search_info.get("num_turns", 0),
        "retrieval_turns": search_info.get("retrieval_turns", []),
        "searchr1_total_search_results": search_info.get("total_search_results", []),
        "searchr1_trace": search_info,
        "rerun_searchr1": True,
    })

    merged["retrieval_info"] = retrieval_info
    merged["predicted_answer"] = search_info.get("predicted_answer", merged.get("predicted_answer", ""))
    merged["reasoning_path"] = search_info.get("reasoning_path", merged.get("reasoning_path", ""))
    merged["reasoning_steps"] = search_info.get("reasoning_steps", merged.get("reasoning_steps", []))
    return merged


def make_turn_context(
    turn: Dict[str, Any],
    next_turn: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    initial_thinking: str = "",
) -> Dict[str, Any]:
    model_output = truncate_text(turn.get("model_output", ""), args.max_think_chars * 2)
    think = extract_think(model_output)
    if not think and initial_thinking:
        think = initial_thinking
    next_model_output = truncate_text(next_turn.get("model_output", ""), args.max_think_chars * 2) if next_turn else ""
    next_think = extract_think(next_model_output)

    documents = turn.get("search_results", [])
    if not isinstance(documents, list):
        documents = [documents] if documents else []
    documents = [truncate_text(doc, args.max_doc_chars) for doc in documents]
    document_text = truncate_text(documents, args.max_doc_chars)

    return {
        "subquery": truncate_text(turn.get("query", ""), args.max_query_chars),
        "think": truncate_text(think, args.max_think_chars),
        "documents": documents,
        "document_text": document_text,
        "next_think": truncate_text(next_think, args.max_think_chars),
    }


def get_openai_client() -> Any:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai and httpx packages are required. Install them or activate the right environment."
        ) from exc

    return OpenAI(
        api_key=os.environ["SKIML_API_KEY"],
        base_url="https://147.47.200.198:7861",
        http_client=httpx.Client(verify=False),
    )


def parse_judge_json(content: str) -> Dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        content = match.group(0)
    return json.loads(content)


def judge(
    client: Any,
    check_name: str,
    prompt: str,
    args: argparse.Namespace,
    retries: int = 2,
) -> Dict[str, Any]:
    allowed = ALLOWED_LABELS[check_name]
    last_error = ""
    temperature = judge_temperature(args.model, args.temperature)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                temperature=temperature,
                max_tokens=JUDGE_MAX_TOKENS,
                messages=[
                    {
                        "role": "system",
                        "content": "Return strict JSON only. Do not include markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                usage = getattr(response, "usage", None)
                raise ValueError(
                    f"Empty judge response; finish_reason={finish_reason!r}; usage={usage!r}"
                )
            parsed = parse_judge_json(content)
            judge_output = validate_pydantic(JudgeOutput, parsed)
            label = judge_output.label.strip().lower()
            if label not in allowed:
                raise ValueError(f"Invalid label {label!r}; allowed={sorted(allowed)}")
            judge_output.label = label
            return pydantic_to_dict(judge_output)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            print(f"[judge error] {check_name}: {last_error}", file=sys.stderr, flush=True)
            if attempt < retries:
                time.sleep(1.0 + attempt)

    return pydantic_to_dict(JudgeOutput(
        label="error",
    ))


def maybe_sleep(args: argparse.Namespace) -> None:
    if args.sleep > 0:
        time.sleep(args.sleep)


def evaluate_sample(
    client: Any,
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    retrieval_info = get_retrieval_info(sample)
    trace = get_searchr1_trace(retrieval_info)
    turns = get_turns(trace, retrieval_info)
    initial_thinking = trace.get("initial_thinking") or retrieval_info.get("searchr1_initial_thinking") or ""
    question_graph = get_question_graph(sample)
    question_graph_text = graph_to_text(question_graph, args.max_graph_chars)
    searchr1_answer = get_searchr1_answer(retrieval_info, trace)
    gold_answers = get_gold_answers(sample)
    output_schema = pydantic_schema(JudgeOutput)

    result = {
        "index": sample.get("index"),
        "uid": sample.get("uid"),
        "question": sample.get("question", ""),
        "gold_answers": gold_answers,
        "searchr1_answer": searchr1_answer,
        "num_turns": len(turns),
        "answer_correct": None,
        "turn_verifications": [],
    }

    if has_trace_output(trace, retrieval_info, sample):
        result["searchr1_trace"] = sanitize_searchr1_trace(trace, retrieval_info, sample)

    answer_prompt = build_prompt(
        "searchr1_final_answer_correct",
        sample,
        question_graph_text,
        {},
        output_schema,
        searchr1_answer=searchr1_answer,
        gold_answers=gold_answers,
    )
    result["answer_correct"] = judge(
        client, "searchr1_final_answer_correct", answer_prompt, args
    )
    maybe_sleep(args)

    for i, turn in enumerate(turns):
        next_turn = turns[i + 1] if i + 1 < len(turns) else None
        turn_initial_thinking = initial_thinking if i == 0 else ""
        context = make_turn_context(turn, next_turn, args, turn_initial_thinking)
        turn_result = {
            "turn": turn.get("turn", i),
            "subquery": context["subquery"],
            "think": context["think"],
            "document_text": context["documents"],
            "next_think": context["next_think"],
            "question_graph_vs_document": None,
            "question_graph_vs_think": None,
            "think_vs_query": None,
            "document_vs_next_think": None,
        }

        for check_name in (
            "question_graph_vs_document",
            "question_graph_vs_think",
            "think_vs_query",
        ):
            prompt = build_prompt(check_name, sample, question_graph_text, context, output_schema)
            turn_result[check_name] = judge(client, check_name, prompt, args)
            maybe_sleep(args)

        if next_turn is not None and context["next_think"]:
            prompt = build_prompt(
                "document_vs_next_think",
                sample,
                question_graph_text,
                context,
                output_schema,
            )
            turn_result["document_vs_next_think"] = judge(
                client, "document_vs_next_think", prompt, args
            )
            maybe_sleep(args)

        result["turn_verifications"].append(turn_result)

    if args.keep_source:
        result["source_sample"] = sample

    return result


def sample_id(sample: Dict[str, Any]) -> str:
    value = sample.get("uid", sample.get("index"))
    return str(value) if value is not None else str(id(sample))


def main() -> int:
    args = parse_args()
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    if not os.getenv("SKIML_API_KEY"):
        print("SKIML_API_KEY is not set.", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    graph_lookup = {}
    if args.question_graph_input:
        graph_records = load_records(Path(args.question_graph_input))
        graph_lookup = build_graph_lookup(graph_records)
    else:
        graph_records = []

    if args.trace_source == "file":
        if not args.input:
            print("--input is required when --trace-source=file.", file=sys.stderr)
            return 2
        records = load_records(Path(args.input))
    else:
        if args.input:
            records = load_records(Path(args.input))
        elif graph_records:
            records = graph_records
        else:
            print("--question-graph-input is required when --trace-source=searchr1 and --input is omitted.", file=sys.stderr)
            return 2

    existing_results, done_ids = load_existing_output(output_path) if args.resume else ([], set())
    client = get_openai_client()
    searchr1_runner = None

    selected = records[args.start :]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]

    results = list(existing_results)
    total = len(selected)
    logged_sample_count = 0
    progress = tqdm(
        selected,
        total=total,
        desc="compare_all",
        dynamic_ncols=True,
    ) if tqdm is not None else selected

    for offset, sample in enumerate(progress, start=1):
        sid = sample_id(sample)
        if sid in done_ids:
            continue
        if tqdm is not None:
            progress.set_postfix_str(f"sample={sid}")
        else:
            print(f"[{offset}/{total}] evaluating sample {sid}", file=sys.stderr, flush=True)
        sample = attach_question_graph(sample, graph_lookup)
        should_run_searchr1 = (
            args.trace_source == "searchr1"
            or (args.trace_source == "file" and args.rerun_empty_searchr1 and has_empty_trace(sample))
        )
        if should_run_searchr1:
            if searchr1_runner is None:
                searchr1_runner = get_searchr1_runner(args)
            if tqdm is None:
                print(f"[{offset}/{total}] running SearchR1 for sample {sid}", file=sys.stderr, flush=True)
            elif args.trace_source == "file":
                tqdm.write(f"[{offset}/{total}] empty trace; running SearchR1 for sample {sid}", file=sys.stderr)
            search_info = searchr1_runner.infer(str(sample.get("question", "")), verbose=False)
            sample = attach_searchr1_result(sample, search_info)
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
