"""
Latency benchmark for:
  1. SearchR1 vanilla inference  (question-level + per-turn)
  2. Document-triplet model       (Llama-3.2-1B-Instruct-document)
  3. Think+search-triplet model   (Llama-3.2-1B-Instruct-question+think+search)

Samples 10 examples per num_hop per dataset, then runs each pipeline phase
sequentially (models are unloaded between phases to avoid OOM).

Inference uses the same system prompts and input format as fine-tuning
(see scripts/finetune/small-llm.py :: SYSTEM_PROMPTS + load_records).
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from search_r1 import SearchR1Inference

# System prompts from small-llm.py :: SYSTEM_PROMPTS (must match fine-tuning)
SYSTEM_PROMPTS = {
    "document": (
        "You are a knowledge graph extraction expert. "
        "Extract all factual triples from the given document. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "question": (
        "You are a knowledge graph extraction expert. "
        "Given a multi-hop question, extract the reasoning triples that represent the relationships needed to answer it. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "think+search": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step and its search query, extract the factual triples it asserts. "
        "For facts already stated, use concrete values. For facts still being searched, use placeholders like (ENT1), (ENT2). "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark for vanilla + triplet models")
    parser.add_argument("--datasets_root", type=str,
                        default="/home/hyeseojeon/data/graph/datasets",
                        help="Root directory containing <dataset>/claims/train_sampled.json")
    parser.add_argument("--datasets", nargs="+",
                        default=["hotpotqa", "2wikimultihopqa", "musique"],
                        help="Dataset names to benchmark")
    parser.add_argument("--samples_per_dataset", type=int, default=30,
                        help="Number of samples to draw per dataset (num_hop-agnostic)")
    parser.add_argument("--seed", type=int, default=42)

    # SearchR1
    parser.add_argument("--searchr1_model_id", type=str,
                        default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--searchr1_max_turns", type=int, default=5)
    parser.add_argument("--searchr1_top_k", type=int, default=3)

    # Triplet models (local fine-tuned)
    parser.add_argument("--document_model_path", type=str,
                        default="/home/hyeseojeon/data/graph/outputs/finetune/Llama-3.2-1B-Instruct-document")
    parser.add_argument("--think_search_model_path", type=str,
                        default="/home/hyeseojeon/data/graph/outputs/finetune/Llama-3.2-1B-Instruct-question+think+search")
    parser.add_argument("--document_max_new_tokens", type=int, default=384,
                        help="Max new tokens for document triplet extraction (covers 98% of training data)")
    parser.add_argument("--think_search_max_new_tokens", type=int, default=128,
                        help="Max new tokens for think+search triplet extraction (fewer triples)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="/home/hyeseojeon/data/graph/results/latency")
    parser.add_argument("--output_filename", type=str, default="latency_benchmark.json")

    # Control which phases to run
    parser.add_argument("--skip_vanilla", action="store_true")
    parser.add_argument("--skip_document", action="store_true")
    parser.add_argument("--skip_think_search", action="store_true")

    # Resume from existing vanilla results
    parser.add_argument("--vanilla_results_path", type=str, default=None,
                        help="Path to previously saved vanilla results JSON to skip vanilla phase")

    # Two-stage vLLM workflow support
    parser.add_argument("--intermediate_results_path", type=str, default=None,
                        help="Path to intermediate results JSON saved after document phase (skips vanilla + document phases)")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate results (with retrieval_turns) after document phase for two-stage vLLM workflows")

    # vLLM backend (OpenAI-compatible API)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM server instead of local transformers inference")
    parser.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8001/v1",
                        help="Base URL for vLLM OpenAI-compatible API (default: http://127.0.0.1:8001/v1)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_and_sample(datasets_root: str, datasets: List[str],
                    samples_per_dataset: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    all_samples: List[Dict[str, Any]] = []

    for ds in datasets:
        path = os.path.join(datasets_root, ds, "claims", "train_sampled.json")
        with open(path) as f:
            data = json.load(f)

        chosen = rng.sample(data, min(samples_per_dataset, len(data)))
        for item in chosen:
            item = dict(item)
            item["_dataset"] = ds
            all_samples.append(item)

        logger.info(f"[{ds}] total={len(data)}  sampled={len(chosen)}")

    logger.info(f"Total samples: {len(all_samples)}")
    return all_samples


# ---------------------------------------------------------------------------
# Phase 1: Vanilla SearchR1
# ---------------------------------------------------------------------------

def run_vanilla_phase(
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    logger.info("=== Phase 1: Vanilla SearchR1 inference ===")
    model = SearchR1Inference(
        model_id=args.searchr1_model_id,
        retriever_url=args.retriever_url,
        max_turns=args.searchr1_max_turns,
        topk=args.searchr1_top_k,
    )

    results = []
    for sample in tqdm(samples, desc="vanilla", file=sys.stdout):
        question = sample["question"]
        t_start = time.perf_counter()
        info = model.infer(question, verbose=False)
        total_elapsed = time.perf_counter() - t_start

        turns = info["retrieval_turns"]

        # question: first turn generate latency (question read + first think+search)
        first_turn_latency = turns[0]["generate_latency_sec"] if turns else 0.0

        # document: search/retrieval latency per turn
        search_latencies = [
            t["search_latency_sec"] for t in turns if "search_latency_sec" in t
        ]

        # think: model generate latency per turn (all turns)
        generate_latencies = [
            t["generate_latency_sec"] for t in turns if "generate_latency_sec" in t
        ]

        # turn: generate + search per turn
        turn_totals = [
            t.get("generate_latency_sec", 0.0) + t.get("search_latency_sec", 0.0)
            for t in turns
        ]

        results.append({
            "dataset": sample["_dataset"],
            "num_hops": sample.get("num_hops", sample.get("num_hop")),
            "question": question,
            "answer": sample.get("answer"),
            "vanilla": {
                "question_latency_sec":      round(first_turn_latency, 4),
                "mean_search_latency_sec":   round(_mean(search_latencies), 4),
                "mean_generate_latency_sec": round(_mean(generate_latencies), 4),
                "mean_turn_latency_sec":     round(_mean(turn_totals), 4),
                "sample_latency_sec":        round(info["latency"]["question_latency_sec"], 4),
                "num_turns":                 info["num_turns"],
            },
            "retrieval_turns": turns,
        })

    # Unload SearchR1
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SearchR1 model unloaded.")
    return results
# ---------------------------------------------------------------------------
# Phase 2: Document-triplet model
# ---------------------------------------------------------------------------

def _vllm_generate(client: "OpenAI", model_name: str, messages: List[Dict],
                   max_new_tokens: int) -> Tuple[str, float]:
    """Call vLLM OpenAI-compatible API (single item)."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    latency = time.perf_counter() - t0
    content = response.choices[0].message.content
    return content, round(latency, 4)


def _vllm_generate_batch(client: "OpenAI", model_name: str, messages_list: List[List[Dict]],
                         max_new_tokens: int) -> Tuple[List[str], float]:
    """Call vLLM OpenAI-compatible API (batch via sequential calls, timed as one batch)."""
    t0 = time.perf_counter()
    responses = []
    for messages in messages_list:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        responses.append(response.choices[0].message.content)
    latency = time.perf_counter() - t0
    return responses, round(latency, 4)


def _load_local_model(model_path: str, compile_model: bool = True) -> Tuple[Any, Any]:
    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    if compile_model:
        logger.info("Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Compilation done.")

    # Warmup: CUDA 첫 실행 비용을 측정에서 제외
    logger.info("Running warmup pass...")
    dummy = tokenizer(["warmup"], return_tensors="pt").to(model.device if not compile_model else next(model.parameters()).device)
    with torch.no_grad():
        model.generate(**dummy, max_new_tokens=8, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
    logger.info("Warmup done.")

    return model, tokenizer


def _apply_chat_template(tokenizer, messages: List[Dict], add_generation_prompt: bool) -> str:
    """Apply chat template matching the fine-tuning setup in small-llm.py."""
    try:
        # Qwen3 supports enable_thinking; Llama and others raise TypeError
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _local_generate(
    model, tokenizer, user_content: str, system_prompt: str, max_new_tokens: int
) -> Tuple[str, float]:
    """Single-item generate (used for question and think+search)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response, round(latency, 4)


def _local_generate_batch(
    model, tokenizer, user_contents: List[str], system_prompt: str, max_new_tokens: int
) -> Tuple[List[str], float]:
    """Batch generate — processes all docs in a turn in one forward pass.
    Uses left padding (required for correct decoder-only batch generation).
    """
    texts = [
        _apply_chat_template(
            tokenizer,
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": u}],
            add_generation_prompt=True,
        )
        for u in user_contents
    ]

    # Left padding for generation (decoder-only models require this)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    tokenizer.padding_side = "right"  # restore for training compatibility

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0

    prompt_len = inputs["input_ids"].shape[1]
    responses = [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in outputs
    ]
    return responses, round(latency, 4)


def run_document_phase(
    vanilla_results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    logger.info("=== Phase 2: Document-triplet model ===")

    if args.use_vllm:
        if not OpenAI:
            raise ImportError("openai library required for vLLM. Install: pip install openai")
        logger.info(f"Using vLLM at: {args.vllm_base_url}")
        client = OpenAI(api_key="dummy", base_url=args.vllm_base_url)
        # Extract model name from path for API (e.g., "Llama-3.2-3B-Instruct-document")
        doc_model_name = os.path.basename(args.document_model_path)
    else:
        model, tokenizer = _load_local_model(args.document_model_path)

    system_prompt = SYSTEM_PROMPTS["document"]
    for result in tqdm(vanilla_results, desc="document model", file=sys.stdout):
        turn_records = []
        for turn_info in result["retrieval_turns"]:
            docs = turn_info.get("search_results", [])
            if not docs:
                continue

            if args.use_vllm:
                messages_list = [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": doc},
                    ]
                    for doc in docs
                ]
                _, turn_latency = _vllm_generate_batch(
                    client, doc_model_name, messages_list, args.document_max_new_tokens
                )
            else:
                _, turn_latency = _local_generate_batch(
                    model, tokenizer, docs, system_prompt, args.document_max_new_tokens
                )
            turn_records.append({
                "turn": turn_info["turn"],
                "num_docs": len(docs),
                "turn_latency_sec": turn_latency,
            })

        all_turn_latencies = [r["turn_latency_sec"] for r in turn_records]
        result["document_model"] = {
            "turns": turn_records,
            "mean_turn_latency_sec": round(_mean(all_turn_latencies), 4),
            "sample_overhead_sec":   round(sum(all_turn_latencies), 4),
        }

    if not args.use_vllm:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Document model unloaded.")
    else:
        logger.info("vLLM document model complete.")
    return vanilla_results


# ---------------------------------------------------------------------------
# Phase 3: Think+Search-triplet model
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_think(text: str) -> str:
    matches = _THINK_RE.findall(text)
    return matches[-1].strip() if matches else ""


def run_think_search_phase(
    vanilla_results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Mirrors the question+think+search mixed-mode fine-tuning (small-llm.py):
      - source_type="question"    → SYSTEM_PROMPTS["question"]    + question text   (once per sample)
      - source_type="think+search"→ SYSTEM_PROMPTS["think+search"] + "{think}\n[Search] {query}"  (once per turn)
    """
    logger.info("=== Phase 3: Think+Search-triplet model ===")

    if args.use_vllm:
        if not OpenAI:
            raise ImportError("openai library required for vLLM. Install: pip install openai")
        logger.info(f"Using vLLM at: {args.vllm_base_url}")
        client = OpenAI(api_key="dummy", base_url=args.vllm_base_url)
        ts_model_name = os.path.basename(args.think_search_model_path)
    else:
        model, tokenizer = _load_local_model(args.think_search_model_path)

    question_sys = SYSTEM_PROMPTS["question"]
    think_search_sys = SYSTEM_PROMPTS["think+search"]

    for result in tqdm(vanilla_results, desc="think+search model", file=sys.stdout):
        # -- question inference (source_type="question") --
        if args.use_vllm:
            messages = [
                {"role": "system", "content": question_sys},
                {"role": "user", "content": result["question"]},
            ]
            _, q_latency = _vllm_generate(client, ts_model_name, messages, args.think_search_max_new_tokens)
        else:
            _, q_latency = _local_generate(
                model, tokenizer,
                result["question"],
                question_sys,
                args.think_search_max_new_tokens,
            )

        # -- per-turn think+search inference (source_type="think+search") --
        turn_records = []
        for turn_info in result["retrieval_turns"]:
            query = turn_info.get("query")
            if not query:
                continue  # final answer turn — no search query

            model_output = turn_info.get("model_output", "")
            think_text = _extract_think(model_output)

            # Input format from finetuning: f"{think_text}\n[Search] {search_query}"
            user_content = f"{think_text}\n[Search] {query}"

            if args.use_vllm:
                messages = [
                    {"role": "system", "content": think_search_sys},
                    {"role": "user", "content": user_content},
                ]
                _, latency = _vllm_generate(client, ts_model_name, messages, args.think_search_max_new_tokens)
            else:
                _, latency = _local_generate(
                    model, tokenizer, user_content, think_search_sys, args.think_search_max_new_tokens
                )
            turn_records.append({
                "turn": turn_info["turn"],
                "latency_sec": latency,
            })

        all_turn_latencies = [r["latency_sec"] for r in turn_records]
        result["think_search_model"] = {
            "question_latency_sec": round(q_latency, 4),
            "turns": turn_records,
            "mean_turn_latency_sec": round(_mean(all_turn_latencies), 4),
            "sample_overhead_sec":   round(q_latency + sum(all_turn_latencies), 4),
        }

    if not args.use_vllm:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Think+search model unloaded.")
    else:
        logger.info("vLLM think+search model complete.")
    return vanilla_results


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float], unit: str = "s") -> str:
    return f"{val:.3f}{unit}" if val is not None else "  N/A  "


def _col(v: Optional[float], w: int = 10) -> str:
    return _fmt(v).center(w)


def print_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print one table per dataset, then a global-average table.

    Columns
    -------
    question : vanilla = turn-0 generate latency (question read + first think+search gen)
               graph   = question → triplets latency (question+think+search model)
    document : vanilla = mean search/retrieval latency per turn
               graph   = mean latency per single document (document model)
    think    : vanilla = mean generate latency per turn (all turns)
               graph   = mean think+search → triplets latency per turn
    turn     : vanilla = mean (generate + search) latency per turn
               graph   = mean (doc-model turn + think-model turn) overhead per turn
    sample   : vanilla = total question_latency_sec (wall-clock per question)
               graph   = total model overhead per sample
                         (question-graph + all doc-graphs + all think-graphs)
    """
    has_vanilla = any("vanilla" in r for r in results)
    has_doc     = any("document_model" in r for r in results)
    has_ts      = any("think_search_model" in r for r in results)
    has_graph   = has_doc or has_ts

    COL_W = 12
    HDR = ["question", "document", "think", "turn", "sample"]
    hdr_line = "  {:<10}".format("") + "".join(h.center(COL_W) for h in HDR)
    div = "  " + "-" * (10 + COL_W * len(HDR))

    def table_row(label: str, vals: List[Optional[float]]) -> str:
        return "  {:<10}".format(label) + "".join(_col(v, COL_W) for v in vals)

    def ds_table(rows: List[Dict]) -> None:
        # vanilla row
        if has_vanilla:
            v_q  = _mean([r["vanilla"]["question_latency_sec"]      for r in rows])
            v_d  = _mean([r["vanilla"]["mean_search_latency_sec"]   for r in rows])
            v_th = _mean([r["vanilla"]["mean_generate_latency_sec"] for r in rows])
            v_t  = _mean([r["vanilla"]["mean_turn_latency_sec"]     for r in rows])
            v_s  = _mean([r["vanilla"]["sample_latency_sec"]        for r in rows])
            print(table_row("vanilla", [v_q, v_d, v_th, v_t, v_s]))
        # graph row: only the model (triplet extraction) latencies
        if has_graph:
            g_q  = _mean([r["think_search_model"]["question_latency_sec"]  for r in rows]) if has_ts  else None
            g_d  = _mean([r["document_model"]["mean_turn_latency_sec"]      for r in rows]) if has_doc else None
            g_th = _mean([r["think_search_model"]["mean_turn_latency_sec"] for r in rows]) if has_ts  else None
            g_t_vals = []
            for r in rows:
                doc_t = r["document_model"]["mean_turn_latency_sec"]      if has_doc else 0.0
                ts_t  = r["think_search_model"]["mean_turn_latency_sec"]  if has_ts  else 0.0
                g_t_vals.append(doc_t + ts_t)
            g_t = _mean(g_t_vals)
            g_s_vals = []
            for r in rows:
                doc_s = r["document_model"]["sample_overhead_sec"]        if has_doc else 0.0
                ts_s  = r["think_search_model"]["sample_overhead_sec"]    if has_ts  else 0.0
                g_s_vals.append(doc_s + ts_s)
            g_s = _mean(g_s_vals)
            print(table_row("graph", [g_q, g_d, g_th, g_t, g_s]))

    datasets = sorted({r["dataset"] for r in results})
    for ds in datasets:
        dr = [r for r in results if r["dataset"] == ds]
        n_turns_avg = _mean([r["vanilla"]["num_turns"] for r in dr]) if has_vanilla else 0.0
        print(f"\n  ── {ds}  (n={len(dr)}, avg_turns={n_turns_avg:.1f}) ──")
        print(hdr_line)
        print(div)
        ds_table(dr)

    # Global averages
    print(f"\n  ── ALL DATASETS  (n={len(results)}) ──")
    print(hdr_line)
    print(div)
    ds_table(results)
    print()


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    samples = load_and_sample(
        args.datasets_root, args.datasets, args.samples_per_dataset, args.seed
    )

    # --- Load starting results ---
    if args.intermediate_results_path:
        logger.info(f"Loading intermediate (post-document) results from: {args.intermediate_results_path}")
        with open(args.intermediate_results_path) as f:
            results = json.load(f)
        args.skip_document = True
    # --- Phase 1: Vanilla ---
    elif args.vanilla_results_path:
        logger.info(f"Loading vanilla results from: {args.vanilla_results_path}")
        with open(args.vanilla_results_path) as f:
            results = json.load(f)
    elif not args.skip_vanilla:
        results = run_vanilla_phase(samples, args)
        # Save intermediate vanilla results
        os.makedirs(args.output_dir, exist_ok=True)
        vanilla_path = os.path.join(
            args.output_dir, args.output_filename.replace(".json", "_vanilla_only.json")
        )
        with open(vanilla_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Vanilla results saved to: {vanilla_path}")
    else:
        logger.warning("Skipping vanilla phase — no vanilla_results_path provided")
        results = [
            {
                "dataset": s["_dataset"],
                "num_hops": s.get("num_hops", s.get("num_hop")),
                "question": s["question"],
                "answer": s.get("answer"),
                "retrieval_turns": [],
            }
            for s in samples
        ]

    # Print vanilla-only summary immediately so it's visible even if later phases fail
    logger.info("=== Vanilla phase complete. Intermediate summary: ===")
    print_summary(results)

    # --- Phase 2: Document model ---
    if not args.skip_document:
        try:
            results = run_document_phase(results, args)
        except Exception as e:
            logger.error(f"Document phase failed: {e}", exc_info=True)
            logger.warning("Skipping document phase — proceeding to next phase.")

    # Save intermediate results (with retrieval_turns) for two-stage vLLM workflows
    if args.save_intermediate and not args.skip_document:
        os.makedirs(args.output_dir, exist_ok=True)
        intermediate_path = os.path.join(
            args.output_dir, args.output_filename.replace(".json", "_after_doc.json")
        )
        with open(intermediate_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Intermediate results saved to: {intermediate_path}")

    # --- Phase 3: Think+search model ---
    if not args.skip_think_search:
        try:
            results = run_think_search_phase(results, args)
        except Exception as e:
            logger.error(f"Think+search phase failed: {e}", exc_info=True)
            logger.warning("Skipping think+search phase — proceeding to save.")

    # Save full results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    output_data = [
        {k: v for k, v in r.items() if k != "retrieval_turns"}
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_path}")

    logger.info("=== Final summary: ===")
    print_summary(results)


if __name__ == "__main__":
    main()
