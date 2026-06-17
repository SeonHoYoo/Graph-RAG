"""
Online latency measurement: vanilla SearchR1 + parallel graph extraction via vLLM.

Per-turn pipeline (identical generate flow to vanilla SearchR1):
  1. generate until </search> (ONE call, same as vanilla)
  2. extract think text → fire think_graph async (vLLM)
  3. retrieval
  4. fire doc_graph async (vLLM)
  5. wait for think_graph if not done   → think_overhead
  6. append <information>...</information>
  7. wait for doc_graph if not done     → doc_overhead
  8. next turn

Question graph fires at sample start (non-blocking, no SearchR1 wait).

Measured latencies
------------------
vanilla:
  question  = turn-0 generate latency (until </search>)
  document  = mean retrieval latency per turn
  think     = mean generate latency per turn (until </search>)
  turn      = mean (generate + retrieval) per turn
  sample    = sum of (generate + retrieval) per turn

graph overhead:
  question  = question graph generation time (parallel, no blocking)
  think     = mean wait time after retrieval for think_graph per turn
  document  = mean wait time after </information> for doc_graph per turn
  turn      = mean (think_overhead + doc_overhead) per turn
  sample    = sum of all turn overheads
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("pip install openai")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from search_r1 import SearchR1Inference

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

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


class _LimitFilter(logging.Filter):
    def __init__(self, limit: int):
        super().__init__()
        self._count = 0
        self._limit = limit

    def filter(self, record: logging.LogRecord) -> bool:
        if self._count < self._limit:
            self._count += 1
            return True
        return False


# Show only first 20 HTTP request logs from httpx/openai
for _name in ("httpx", "openai", "openai._base_client"):
    _log = logging.getLogger(_name)
    _log.addFilter(_LimitFilter(20))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online latency benchmark")
    p.add_argument("--datasets_root", default="/home/hyeseojeon/data/graph/datasets")
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    p.add_argument("--samples_per_dataset", type=int, default=None,
                   help="Samples per dataset. Omit or leave empty = use all.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--searchr1_model_id",
                   default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    p.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--searchr1_max_turns", type=int, default=5)
    p.add_argument("--searchr1_top_k", type=int, default=3)

    p.add_argument("--vllm_base_url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--document_model_path",
                   default="outputs/finetune/Qwen2.5-0.5B-Instruct-document")
    p.add_argument("--think_search_model_path",
                   default="outputs/finetune/Qwen2.5-0.5B-Instruct-question+think+search")
    p.add_argument("--document_max_new_tokens", type=int, default=384)
    p.add_argument("--think_search_max_new_tokens", type=int, default=128)

    p.add_argument("--output_dir", default="results/latency/online")
    p.add_argument("--output_filename", default="online_latency.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_sample(datasets_root: str, datasets: List[str],
                    n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    for ds in datasets:
        path = os.path.join(datasets_root, ds, "claims", "train_sampled.json")
        with open(path) as f:
            data = json.load(f)
        chosen = rng.sample(data, len(data) if n is None else min(n, len(data)))
        for item in chosen:
            item = dict(item)
            item["_dataset"] = ds
            samples.append(item)
        logger.info(f"[{ds}] sampled {len(chosen)}/{len(data)}")
    logger.info(f"Total samples: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# Online runner
# ---------------------------------------------------------------------------

class OnlineRunner:
    def __init__(
        self,
        r1: SearchR1Inference,
        client: OpenAI,
        doc_model: str,
        ts_model: str,
        doc_max_tokens: int,
        ts_max_tokens: int,
    ):
        self.r1 = r1
        self.client = client
        self.doc_model = doc_model
        self.ts_model = ts_model
        self.doc_max_tokens = doc_max_tokens
        self.ts_max_tokens = ts_max_tokens

    # ── vLLM helpers (called from worker threads) ────────────────────────────

    def _call_single(self, model: str, system: str, user: str, max_tokens: int) -> Tuple[str, float]:
        """Returns (generated_text, latency_sec)."""
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content, round(time.perf_counter() - t0, 4)

    def _call_batch(self, model: str, system: str, users: List[str], max_tokens: int) -> Tuple[List[str], float]:
        """Sequential batch calls; returns (list_of_generated_texts, total_latency_sec)."""
        t0 = time.perf_counter()
        outputs = []
        for user in users:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            outputs.append(resp.choices[0].message.content)
        return outputs, round(time.perf_counter() - t0, 4)

    # ── SearchR1 generate helper ─────────────────────────────────────────────

    def _generate(self, prompt: str, stop_criteria) -> Tuple[str, float, bool]:
        """Single generate call. Returns (decoded_new_tokens, latency_sec, hit_eos)."""
        r1 = self.r1
        ids = r1.tokenizer.encode(prompt, return_tensors="pt").to(r1.device)
        t0 = time.perf_counter()
        out = r1.model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=r1.max_new_tokens,
            stopping_criteria=stop_criteria,
            pad_token_id=r1.tokenizer.eos_token_id,
            do_sample=True,
            temperature=r1.temperature,
        )
        latency = round(time.perf_counter() - t0, 4)
        hit_eos = out[0][-1].item() in r1.curr_eos
        text = r1.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return text, latency, hit_eos

    # ── main per-sample inference ─────────────────────────────────────────────

    def run(self, question: str, executor: ThreadPoolExecutor) -> Dict[str, Any]:
        r1 = self.r1
        q = question.strip()
        if not q.endswith("?"):
            q += "?"

        base = (
            f"Answer the given question. "
            f"You must conduct reasoning inside <think> and </think> first every time you get new information. "
            f"After reasoning, if you find you lack some knowledge, you can call a search engine by "
            f"<search> query </search> and it will return the top searched results between <information> "
            f"and </information>. You can search as many times as your want. "
            f"If you find no further external knowledge needed, you can directly provide the answer inside "
            f"<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
            f"Question: {q}\n"
        )
        if r1.tokenizer.chat_template:
            prompt = r1.tokenizer.apply_chat_template(
                [{"role": "user", "content": base}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt = base

        # ── question graph: fire immediately, non-blocking ───────────────────
        q_future: Future = executor.submit(
            self._call_single, self.ts_model, SYSTEM_PROMPTS["question"],
            q, self.ts_max_tokens,
        )

        cnt = 0
        turns: List[Dict[str, Any]] = []

        while cnt < r1.max_turns:

            # ── Step 1: generate until </search> (identical to vanilla SearchR1) ──
            turn_output, t_gen, hit_eos = self._generate(prompt, r1.stopping_criteria)

            if hit_eos:
                # final answer turn (no search query)
                turns.append({
                    "turn": cnt, "query": None,
                    "model_output": turn_output, "search_results": [],
                    "think_graph_triples": None, "doc_graph_triples": [],
                    "generate_latency_sec": t_gen,
                    "search_latency_sec": 0.0,
                    "think_graph_overhead_sec": 0.0,
                    "doc_graph_overhead_sec": 0.0,
                })
                break

            # ── Step 2: extract think text → fire think_graph async ──────────
            m = _THINK_RE.search(turn_output)
            think_text = m.group(1).strip() if m else ""
            if think_text:
                think_future: Future = executor.submit(
                    self._call_single, self.ts_model, SYSTEM_PROMPTS["think+search"],
                    think_text, self.ts_max_tokens,
                )
            else:
                think_future = None

            # ── Step 3: retrieval ─────────────────────────────────────────────
            query = r1._get_query(turn_output)
            t0 = time.perf_counter()
            docs = r1._search(query) if query else []
            search_lat = round(time.perf_counter() - t0, 4)

            # ── Step 4: fire doc graph async ──────────────────────────────────
            if docs:
                doc_future: Future = executor.submit(
                    self._call_batch, self.doc_model, SYSTEM_PROMPTS["document"],
                    docs, self.doc_max_tokens,
                )
            else:
                doc_future = None

            # ── Step 5: wait for think graph (deadline: before <information>) ─
            if think_future is not None:
                t0 = time.perf_counter()
                think_graph_triples, _ = think_future.result()
                think_overhead = round(max(0.0, time.perf_counter() - t0), 4)
            else:
                think_graph_triples = None
                think_overhead = 0.0

            # ── Step 6: append <information>...</information> ─────────────────
            results_str = "\n".join(f"Doc {i+1}{d}" for i, d in enumerate(docs))
            info_block = r1.curr_search_template.format(
                output_text=turn_output, search_results=results_str
            )
            prompt += info_block

            # ── Step 7: wait for doc graph (deadline: before next generation) ─
            if doc_future is not None:
                t0 = time.perf_counter()
                doc_graph_triples, _ = doc_future.result()
                doc_overhead = round(max(0.0, time.perf_counter() - t0), 4)
            else:
                doc_graph_triples = []
                doc_overhead = 0.0

            turns.append({
                "turn": cnt,
                "query": query,
                "model_output": turn_output,
                "search_results": docs,
                "think_graph_triples": think_graph_triples,
                "doc_graph_triples": doc_graph_triples,
                "generate_latency_sec": t_gen,
                "search_latency_sec": search_lat,
                "think_graph_overhead_sec": think_overhead,
                "doc_graph_overhead_sec": doc_overhead,
            })
            cnt += 1

        q_graph_triples, q_graph_latency = q_future.result()

        vanilla = {
            "question_latency_sec":      turns[0]["generate_latency_sec"] if turns else 0.0,
            "mean_generate_latency_sec": _mean([t["generate_latency_sec"] for t in turns]),
            "mean_search_latency_sec":   _mean([t["search_latency_sec"] for t in turns]),
            "mean_turn_latency_sec":     _mean([
                t["generate_latency_sec"] + t["search_latency_sec"] for t in turns
            ]),
            "sample_latency_sec": round(sum(
                t["generate_latency_sec"] + t["search_latency_sec"] for t in turns
            ), 4),
        }
        graph = {
            "question_graph_latency_sec": q_graph_latency,
            "mean_think_overhead_sec":    _mean([t["think_graph_overhead_sec"] for t in turns]),
            "mean_doc_overhead_sec":      _mean([t["doc_graph_overhead_sec"] for t in turns]),
            "mean_turn_overhead_sec":     _mean([
                t["think_graph_overhead_sec"] + t["doc_graph_overhead_sec"] for t in turns
            ]),
            "sample_overhead_sec": round(sum(
                t["think_graph_overhead_sec"] + t["doc_graph_overhead_sec"] for t in turns
            ), 4),
        }
        total = {
            "mean_turn_latency_sec": round(
                vanilla["mean_turn_latency_sec"] + graph["mean_turn_overhead_sec"], 4
            ),
            "sample_latency_sec": round(
                vanilla["sample_latency_sec"] + graph["sample_overhead_sec"], 4
            ),
        }
        return {
            "question": q,
            "dataset": None,   # filled by caller
            "num_hops": None,  # filled by caller
            "num_turns": cnt,
            "question_graph_triples": q_graph_triples,
            "vanilla": vanilla,
            "graph": graph,
            "total": total,
            "retrieval_turns": turns,
        }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}s" if v is not None else "  N/A  "


def _col(v: Optional[float], w: int = 12) -> str:
    return _fmt(v).center(w)


def print_summary(results: List[Dict[str, Any]]) -> None:
    """
    Columns
    -------
    question : vanilla = turn-0 generate latency
               graph   = question graph generation time (non-blocking)
    document : vanilla = mean retrieval latency per turn
               graph   = mean doc_graph wait after </information> per turn
    think    : vanilla = mean generate latency per turn (until </search>)
               graph   = mean think_graph wait after </search> per turn
    turn     : vanilla = mean (generate + retrieval) per turn
               graph   = mean (think_overhead + doc_overhead) per turn
    sample   : vanilla = total wall-clock per sample
               graph   = sum of all turn overheads per sample
    """
    COL_W = 12
    HDR = ["question", "document", "think", "turn", "sample"]
    hdr_line = "  {:<10}".format("") + "".join(h.center(COL_W) for h in HDR)
    div = "  " + "-" * (10 + COL_W * len(HDR))

    def row(label: str, vals: List[Optional[float]]) -> str:
        return "  {:<10}".format(label) + "".join(_col(v, COL_W) for v in vals)

    def table(rows: List[Dict]) -> None:
        v_q  = _mean([r["vanilla"]["question_latency_sec"]      for r in rows])
        v_d  = _mean([r["vanilla"]["mean_search_latency_sec"]   for r in rows])
        v_th = _mean([r["vanilla"]["mean_generate_latency_sec"] for r in rows])
        v_t  = _mean([r["vanilla"]["mean_turn_latency_sec"]     for r in rows])
        v_s  = _mean([r["vanilla"]["sample_latency_sec"]        for r in rows])

        g_q  = _mean([r["graph"]["question_graph_latency_sec"] for r in rows])
        g_d  = _mean([r["graph"]["mean_doc_overhead_sec"]      for r in rows])
        g_th = _mean([r["graph"]["mean_think_overhead_sec"]    for r in rows])
        g_t  = _mean([r["graph"]["mean_turn_overhead_sec"]     for r in rows])
        g_s  = _mean([r["graph"]["sample_overhead_sec"]        for r in rows])

        t_t  = _mean([r["total"]["mean_turn_latency_sec"] for r in rows])
        t_s  = _mean([r["total"]["sample_latency_sec"]    for r in rows])

        print(row("vanilla", [v_q,  v_d,  v_th, v_t,  v_s]))
        print(row("graph",   [g_q,  g_d,  g_th, g_t,  g_s]))
        print(row("total",   [None, None, None,  t_t,  t_s]))

    datasets = sorted({r["dataset"] for r in results})
    for ds in datasets:
        dr = [r for r in results if r["dataset"] == ds]
        avg_turns = _mean([r["num_turns"] for r in dr])
        print(f"\n  ── {ds}  (n={len(dr)}, avg_turns={avg_turns:.1f}) ──")
        print(hdr_line)
        print(div)
        table(dr)

    print(f"\n  ── ALL DATASETS  (n={len(results)}) ──")
    print(hdr_line)
    print(div)
    table(results)
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    logger.info(f"Loading SearchR1: {args.searchr1_model_id}")
    r1 = SearchR1Inference(
        model_id=args.searchr1_model_id,
        retriever_url=args.retriever_url,
        max_turns=args.searchr1_max_turns,
        topk=args.searchr1_top_k,
    )

    logger.info(f"Connecting to vLLM at: {args.vllm_base_url}")
    client = OpenAI(api_key="dummy", base_url=args.vllm_base_url)
    doc_model  = os.path.basename(args.document_model_path)
    ts_model   = os.path.basename(args.think_search_model_path)

    runner = OnlineRunner(
        r1=r1,
        client=client,
        doc_model=doc_model,
        ts_model=ts_model,
        doc_max_tokens=args.document_max_new_tokens,
        ts_max_tokens=args.think_search_max_new_tokens,
    )

    results: List[Dict[str, Any]] = []
    # max_workers=3: question, think, doc graphs can all be in-flight simultaneously
    with ThreadPoolExecutor(max_workers=3) as executor:
        for i, sample in enumerate(tqdm(samples, desc="online inference", file=sys.stdout)):
            result = runner.run(sample["question"], executor)
            result["dataset"]  = sample["_dataset"]
            result["num_hops"] = sample.get("num_hops", sample.get("num_hop"))
            results.append(result)

            if i < 5:
                v, g, t = result["vanilla"], result["graph"], result["total"]
                logger.info(
                    f"[sample {i}] {result['dataset']} | turns={result['num_turns']} | "
                    f"vanilla(turn={v['mean_turn_latency_sec']:.3f}s sample={v['sample_latency_sec']:.3f}s) | "
                    f"graph(q={g['question_graph_latency_sec']:.3f}s think={g['mean_think_overhead_sec']:.3f}s doc={g['mean_doc_overhead_sec']:.3f}s overhead={g['sample_overhead_sec']:.3f}s) | "
                    f"total(turn={t['mean_turn_latency_sec']:.3f}s sample={t['sample_latency_sec']:.3f}s)"
                )

    logger.info("=== Summary ===")
    print_summary(results)

    os.makedirs(args.output_dir, exist_ok=True)
    stem, ext = os.path.splitext(args.output_filename)

    # File 1: latency summary only (no per-turn detail)
    latency_path = os.path.join(args.output_dir, f"{stem}_latency{ext}")
    latency_data = [{k: v for k, v in r.items() if k != "retrieval_turns"} for r in results]
    with open(latency_path, "w") as f:
        json.dump(latency_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Latency summary saved to: {latency_path}")

    # File 2: per-turn graph breakdown (retrieval_turns included)
    detail_path = os.path.join(args.output_dir, f"{stem}_detail{ext}")
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Detail saved to: {detail_path}")


if __name__ == "__main__":
    main()
