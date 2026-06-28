#!/usr/bin/env python
"""Run SearchR1 with VeriGraph feedback injected after every search turn.

This is the online-feedback experiment, separate from the baseline fallback
runner. It does not run a vanilla SearchR1 prepass and does not use a trigger:
each sample starts with SearchR1 + observer feedback enabled from turn 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from graphqa.qa import _is_yesno_question, score_answer
from graphqa.scripts.run_online_corrector import _load_raw_dataset
from graphqa.scripts.run_online_eval import (
    ExtractorPool,
    GraphGenResult,
    SYSTEM_PROMPTS,
    parse_graph_output,
    _split_question_triples,
    make_graph_input,
    triples_to_raw,
)
from graphqa.tasi.embedding import get_default_encoder
from search_r1 import SearchR1Inference

from online_feedback_core import run_hs_online_feedback

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--input-filename", default="train_sampled.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dataset-limits", type=int, nargs="*", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--uid-file", default="", help="Optional file containing one uid per line to run a fixed subset.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-suffix", default="")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8003/retrieve")

    p.add_argument("--searchr1-model", default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    p.add_argument("--searchr1-top-k", type=int, default=3)
    p.add_argument("--searchr1-max-turns", type=int, default=5)
    p.add_argument("--searchr1-max-new-tokens", type=int, default=500)
    p.add_argument("--searchr1-temperature", type=float, default=1.0)
    p.add_argument("--searchr1-verbose", action="store_true")

    p.add_argument("--online-feedback-max-turns", type=int, default=8)
    p.add_argument("--online-feedback-threshold", type=float, default=0.50)
    p.add_argument(
        "--feedback-style",
        choices=[
            "natural_mismatch",
            "natural_mismatch_query",
            "natural_revision",
            "revision_brief",
            "correction_brief",
            "repair_brief",
            "natural",
            "natural_brief",
            "natural_check",
            "natural_guard",
            "soft",
            "score",
            "strict",
            "checklist",
            "next_query",
            "explicit_requirements",
            "requirement_brief",
            "natural_mismatch_query_norid",
            "natural_mismatch_query_short",
            "explicit_requirements_short",
            "explicit_requirements_soft",
        ],
        default="natural_mismatch",
    )
    p.add_argument("--feedback-position", choices=["after_info", "before_info", "inside_info", "next_think_prefix"], default="after_info")
    p.add_argument("--feedback-engine", choices=["legacy", "labels"], default="legacy")
    p.add_argument(
        "--feedback-source",
        choices=["full", "question_only"],
        default="full",
        help="full = Q/D/T verifier; question_only = inject only decomposed question requirements as vg_hint.",
    )
    p.add_argument(
        "--feedback-trigger",
        choices=[
            "all",
            "non_support",
            "actionable",
            "strict",
            "strict_query",
            "strict_or_missing_query",
            "strict_query_or_missing_query",
            "unsupported_only",
            "conflict_only",
            "claim_only",
            "missing_query",
            "off",
        ],
        default="all",
        help="Which verifier labels are strong enough to inject as vg_hint.",
    )
    p.add_argument("--abstain-on-unresolved", action="store_true")
    p.add_argument("--abstain-min-turn", type=int, default=3)

    p.add_argument("--max-docs-per-sample", type=int, default=10)
    p.add_argument("--doc-max-words", type=int, default=500)
    p.add_argument("--question-model", default="outputs/finetune/Qwen2.5-0.5B-Instruct-question")
    p.add_argument("--document-model", default="outputs/finetune/Qwen2.5-0.5B-Instruct-document")
    p.add_argument("--think-model", default="outputs/finetune/Qwen2.5-0.5B-Instruct-think")
    p.add_argument("--question-task", default="question")
    p.add_argument("--think-task", default="think")
    p.add_argument("--graph-backend", choices=["local", "vllm"], default="local")
    p.add_argument("--vllm-base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--vllm-question-base-url", default="")
    p.add_argument("--vllm-document-base-url", default="")
    p.add_argument("--vllm-think-base-url", default="")
    p.add_argument("--vllm-question-model-name", default="")
    p.add_argument("--vllm-document-model-name", default="")
    p.add_argument("--vllm-think-model-name", default="")
    p.add_argument("--vllm-batch-concurrency", type=int, default=8)
    p.add_argument("--graph-overlap-workers", type=int, default=4)
    p.add_argument("--graph-dtype", default="bfloat16")
    p.add_argument("--graph-device-map", default="auto")
    p.add_argument("--graph-base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--graph-max-new-tokens", type=int, default=512)
    p.add_argument("--graph-temperature", type=float, default=0.0)
    p.add_argument("--cosine-doc-top-k", type=int, default=3)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _served_model_name(model_id: str) -> str:
    return str(model_id or "").rstrip("/").split("/")[-1]


class VLLMGraphExtractor:
    def __init__(self, *, client: Any, model_name: str, max_new_tokens: int, temperature: float, batch_concurrency: int = 8) -> None:
        self.client = client
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.batch_concurrency = max(1, int(batch_concurrency))

    def generate(self, task: str, user_content: str) -> GraphGenResult:
        if task not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown graph task: {task}")
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        text = str(resp.choices[0].message.content or "").strip()
        return GraphGenResult(text=text, triples=parse_graph_output(text))

    def generate_many(self, task: str, user_contents: List[str]) -> List[GraphGenResult]:
        """Generate several graph outputs concurrently so vLLM can batch them."""
        if not user_contents:
            return []
        if len(user_contents) == 1:
            return [self.generate(task, user_contents[0])]
        workers = min(self.batch_concurrency, len(user_contents))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self.generate, task, user) for user in user_contents]
            return [f.result() for f in futures]


class VLLMExtractorPool:
    def __init__(self, args: argparse.Namespace) -> None:
        if OpenAI is None:
            raise RuntimeError("The openai package is required for --graph-backend vllm")
        self.args = args
        self._pool: Dict[str, VLLMGraphExtractor] = {}
        self._clients: Dict[str, Any] = {}
        self._name_by_model = {
            args.question_model: args.vllm_question_model_name or _served_model_name(args.question_model),
            args.document_model: args.vllm_document_model_name or _served_model_name(args.document_model),
            args.think_model: args.vllm_think_model_name or _served_model_name(args.think_model),
        }
        self._url_by_model = {
            args.question_model: args.vllm_question_base_url or args.vllm_base_url,
            args.document_model: args.vllm_document_base_url or args.vllm_base_url,
            args.think_model: args.vllm_think_base_url or args.vllm_base_url,
        }

    def _client_for(self, base_url: str) -> Any:
        if base_url not in self._clients:
            self._clients[base_url] = OpenAI(api_key="dummy", base_url=base_url)
        return self._clients[base_url]

    def get(self, model_id: str) -> VLLMGraphExtractor:
        if model_id not in self._pool:
            model_name = self._name_by_model.get(model_id) or _served_model_name(model_id)
            base_url = self._url_by_model.get(model_id) or self.args.vllm_base_url
            logger.info("[graph-model:vllm] %s -> served model %s at %s", model_id, model_name, base_url)
            self._pool[model_id] = VLLMGraphExtractor(
                client=self._client_for(base_url),
                model_name=model_name,
                max_new_tokens=int(self.args.graph_max_new_tokens),
                temperature=float(self.args.graph_temperature),
                batch_concurrency=int(getattr(self.args, "vllm_batch_concurrency", 8) or 8),
            )
        return self._pool[model_id]


def _resolve_limits(args: argparse.Namespace) -> Optional[List[int]]:
    dataset_limits = args.dataset_limits if args.dataset_limits else None
    if dataset_limits and len(dataset_limits) != len(args.datasets):
        if len(dataset_limits) == 1:
            dataset_limits = dataset_limits * len(args.datasets)
        else:
            raise ValueError(
                f"--dataset-limits length ({len(dataset_limits)}) must match "
                f"--datasets length ({len(args.datasets)})"
            )
    return dataset_limits


def _load_uid_filter(path: str) -> Optional[set]:
    if not path:
        return None
    p = pathlib.Path(path)
    uids = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            uids.add(line.split()[0])
    return uids


def _avg(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return round(mean(xs), 6) if xs else None


def _sum(vals: Iterable[Optional[float]]) -> float:
    return round(sum(float(v) for v in vals if v is not None), 6)


def _turn_score_summary(turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_scores: List[Optional[float]] = []
    think_scores: List[Optional[float]] = []
    injected = 0
    sanitized = 0
    label_unsupported = 0
    label_conflict = 0
    label_missing = 0
    label_query_miss = 0
    doc_graph_latencies: List[Optional[float]] = []
    think_graph_latencies: List[Optional[float]] = []
    alignment_latencies: List[Optional[float]] = []
    label_latencies: List[Optional[float]] = []
    feedback_latencies: List[Optional[float]] = []
    observer_latencies: List[Optional[float]] = []
    for tr in turn_records:
        if tr.get("injected_feedback"):
            injected += 1
        if tr.get("query_was_sanitized"):
            sanitized += 1
        for lab in tr.get("verification_labels", []) or []:
            if lab.get("d_t") == "unsupported_think_claim":
                label_unsupported += 1
            elif lab.get("d_t") == "conflict":
                label_conflict += 1
            if lab.get("q_d") in {"no_support", "weak_or_irrelevant_doc"}:
                label_missing += 1
            if lab.get("query_q") == "query_may_miss_requirement":
                label_query_miss += 1
        for row in tr.get("alignment", []) or []:
            d = row.get("doc_match", {}) or {}
            t = row.get("think_match", {}) or {}
            doc_scores.append(d.get("min_field_cosine"))
            think_scores.append(t.get("min_field_cosine"))
        lat = tr.get("latency", {}) or {}
        doc_graph_latencies.append(lat.get("document_graph_sec"))
        think_graph_latencies.append(lat.get("think_graph_sec"))
        alignment_latencies.append(lat.get("alignment_sec"))
        label_latencies.append(lat.get("verification_label_sec"))
        feedback_latencies.append(lat.get("feedback_format_sec"))
        observer_latencies.append(lat.get("observer_total_sec"))
    return {
        "feedback_turns": injected,
        "sanitized_query_turns": sanitized,
        "label_unsupported_think_claim": label_unsupported,
        "label_conflict": label_conflict,
        "label_missing_or_weak_doc": label_missing,
        "label_query_may_miss_requirement": label_query_miss,
        "avg_doc_min_field_cosine": _avg(doc_scores),
        "avg_think_min_field_cosine": _avg(think_scores),
        "mean_document_graph_sec": _avg(doc_graph_latencies),
        "mean_think_graph_sec": _avg(think_graph_latencies),
        "mean_alignment_sec": _avg(alignment_latencies),
        "mean_verification_label_sec": _avg(label_latencies),
        "mean_feedback_format_sec": _avg(feedback_latencies),
        "mean_observer_total_sec": _avg(observer_latencies),
        "sum_document_graph_sec": _sum(doc_graph_latencies),
        "sum_think_graph_sec": _sum(think_graph_latencies),
        "sum_graph_sec": round(_sum(doc_graph_latencies) + _sum(think_graph_latencies), 6),
        "sum_alignment_sec": _sum(alignment_latencies),
        "sum_verification_label_sec": _sum(label_latencies),
        "sum_feedback_format_sec": _sum(feedback_latencies),
        "sum_observer_total_sec": _sum(observer_latencies),
    }


def _summarize(df: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(df))
    if n == 0:
        return {"n_samples": 0}
    return {
        "n_samples": n,
        "em_mean": float(df["em"].mean()),
        "f1_mean": float(df["f1"].mean()),
        "turns_mean": float(df["num_turns"].mean()),
        "feedback_turns_mean": float(df["feedback_turns"].mean()),
        "sanitized_query_turns_mean": float(df["sanitized_query_turns"].mean()),
        "sample_wall_sec_mean": float(df["sample_wall_sec"].mean()) if "sample_wall_sec" in df else None,
        "observer_total_sec_mean": float(df["sum_observer_total_sec"].mean()) if "sum_observer_total_sec" in df else None,
        "graph_sec_mean": float(df["sum_graph_sec"].mean()) if "sum_graph_sec" in df else None,
    }


def _suffix_part(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._-")
    return f"_{s}" if s else ""


def _write_dataset_outputs(
    *,
    out_dir: pathlib.Path,
    dataset: str,
    rows: List[Dict[str, Any]],
    suffix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    sp = _suffix_part(suffix)
    csv_path = out_dir / f"online_feedback_{dataset}{sp}.csv"
    summary_path = out_dir / f"online_feedback_{dataset}_summary{sp}.json"
    df.to_csv(csv_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_summarize(df), f, indent=2, ensure_ascii=False)


def _load_existing_cases_rows_uids(
    cases_path: pathlib.Path,
    legacy_jsonl_path: pathlib.Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set]:
    cases: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    done: set = set()

    if cases_path.exists():
        try:
            with cases_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                cases = [c for c in loaded if isinstance(c, dict)]
        except Exception:
            cases = []
    elif legacy_jsonl_path.exists():
        with legacy_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    cases.append(obj)

    for obj in cases:
        uid = obj.get("uid") or obj.get("question") or ""
        if uid:
            done.add(uid)
        csv_row = obj.get("csv_row")
        if isinstance(csv_row, dict):
            rows.append(csv_row)
    return cases, rows, done


def _write_cases_json(cases_path: pathlib.Path, cases: List[Dict[str, Any]]) -> None:
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)


def _fmt_sec(value: Optional[float]) -> str:
    if value is None:
        return "  N/A  "
    return f"{float(value):.3f}s"


def _print_latency_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    def _mean_col(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [float(r[key]) for r in items if r.get(key) not in ("", None)]
        return round(mean(vals), 6) if vals else None

    def _print_block(name: str, items: List[Dict[str, Any]]) -> None:
        avg_turns = _mean_col(items, "num_turns") or 0.0
        cols = [
            ("q_graph", "question_graph_sec"),
            ("doc_graph", "sum_document_graph_sec"),
            ("think_graph", "sum_think_graph_sec"),
            ("align", "sum_alignment_sec"),
            ("feedback", "sum_feedback_format_sec"),
            ("observer", "sum_observer_total_sec"),
            ("wall", "sample_wall_sec"),
        ]
        print(f"\n  -- {name}  (n={len(items)}, avg_turns={avg_turns:.1f})")
        print("  " + "".join(label.center(14) for label, _ in cols))
        print("  " + "-" * (14 * len(cols)))
        print("  " + "".join(_fmt_sec(_mean_col(items, key)).center(14) for _, key in cols))

    datasets = sorted({str(r.get("dataset") or "") for r in rows})
    print("\n=== HS online feedback latency summary ===")
    for ds in datasets:
        _print_block(ds, [r for r in rows if str(r.get("dataset") or "") == ds])
    _print_block("ALL", rows)
    print()


def run_one_sample(
    *,
    raw: Dict[str, Any],
    dataset: str,
    pool: ExtractorPool,
    searchr1: SearchR1Inference,
    encoder,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sample_t0 = time.perf_counter()
    question = str(raw.get("question", "") or "").strip()
    gold_answer = str(raw.get("answer", "") or "")
    gold_aliases = list(raw.get("answer_aliases", []) or [])

    q_extractor = pool.get(args.question_model)
    overlap_executor = getattr(args, "_graph_overlap_executor", None)

    def _run_question_graph() -> Tuple[GraphGenResult, float]:
        q_t0 = time.perf_counter()
        result = q_extractor.generate(
            args.question_task,
            make_graph_input(args.question_task, question=question),
        )
        return result, time.perf_counter() - q_t0

    if overlap_executor is not None:
        q_future = overlap_executor.submit(_run_question_graph)
    else:
        q_future = None
        q_result, question_graph_elapsed = _run_question_graph()

    run_args = argparse.Namespace(**vars(args))
    run_args.corrector_v2_max_turns = int(args.online_feedback_max_turns)
    run_args.corrector_v2_threshold = float(args.online_feedback_threshold)

    out = run_hs_online_feedback(
        question=question,
        q_triples=None if q_future is not None else _split_question_triples(q_result.triples)[0],
        q_future=q_future,
        pool=pool,
        searchr1=searchr1,
        encoder=encoder,
        args=run_args,
    )

    if q_future is not None:
        q_result, question_graph_elapsed = q_future.result()
    question_graph_sec = round(question_graph_elapsed, 6)
    q_triples, q_def_triples = _split_question_triples(q_result.triples)

    pred = str(out.predicted_answer or "")
    sample_wall_sec = round(time.perf_counter() - sample_t0, 6)
    em, f1 = score_answer(pred, gold_answer, gold_aliases)
    score_summary = _turn_score_summary(out.turn_records)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "uid": raw.get("uid") or raw.get("id") or raw.get("question") or "",
        "index": raw.get("index"),
        "question": question,
        "answer": gold_answer,
        "is_yesno": bool(_is_yesno_question(question)),
        "n_hops": int(raw.get("num_hops", 0) or 0),
        "mode": "online_feedback",
        "predicted_answer": pred,
        "num_turns": int(out.num_turns),
        "em": float(em),
        "f1": float(f1),
        "n_question_triples": len(q_triples),
        "n_question_definition_triples": len(q_def_triples),
        "output_suffix": str(args.output_suffix or ""),
        "graph_backend": str(args.graph_backend or ""),
        "vllm_base_url": str(args.vllm_base_url or "") if args.graph_backend == "vllm" else "",
        "vllm_question_base_url": str(args.vllm_question_base_url or args.vllm_base_url or "") if args.graph_backend == "vllm" else "",
        "vllm_document_base_url": str(args.vllm_document_base_url or args.vllm_base_url or "") if args.graph_backend == "vllm" else "",
        "vllm_think_base_url": str(args.vllm_think_base_url or args.vllm_base_url or "") if args.graph_backend == "vllm" else "",
        "feedback_engine": str(args.feedback_engine or ""),
        "feedback_source": str(args.feedback_source or "full"),
        "feedback_style": str(args.feedback_style or ""),
        "feedback_position": str(args.feedback_position or ""),
        "question_graph_sec": question_graph_sec,
        "sample_wall_sec": sample_wall_sec,
        **score_summary,
    }
    case = {
        "dataset": dataset,
        "uid": row["uid"],
        "question": question,
        "answer": gold_answer,
        "question_graph_raw": q_result.text,
        "question_graph_latency_sec": question_graph_sec,
        "question_triples": triples_to_raw(q_triples),
        "question_definition_triples": triples_to_raw(q_def_triples),
        "full_response": out.full_response,
        "predicted_answer": pred,
        "num_turns": int(out.num_turns),
        "retrieval_turns": out.retrieval_turns,
        "turn_records": out.turn_records,
        "observer_events": out.observer_events,
        "csv_row": row,
    }
    logger.info(
        "[latency] dataset=%s uid=%s turns=%s wall=%.3fs q_graph=%.3fs "
        "graph_sum=%.3fs observer_sum=%.3fs mean_observer_turn=%.3fs",
        dataset,
        row["uid"],
        row["num_turns"],
        row["sample_wall_sec"],
        row["question_graph_sec"],
        row.get("sum_graph_sec") or 0.0,
        row.get("sum_observer_total_sec") or 0.0,
        row.get("mean_observer_total_sec") or 0.0,
    )
    return row, case


def main() -> int:
    args = parse_args()
    dataset_limits = _resolve_limits(args)
    uid_filter = _load_uid_filter(args.uid_file)
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("[online_feedback] init SearchR1 model=%s", args.searchr1_model)
    searchr1 = SearchR1Inference(
        model_id=args.searchr1_model,
        retriever_url=args.retriever_url,
        max_turns=int(args.searchr1_max_turns),
        max_new_tokens=int(args.searchr1_max_new_tokens),
        temperature=float(args.searchr1_temperature),
        topk=int(args.searchr1_top_k),
    )
    encoder = get_default_encoder(model_name=args.encoder, device=args.device)
    if args.graph_backend == "vllm":
        logger.info(
            "[online_feedback] graph backend=vllm q=%s d=%s t=%s",
            args.vllm_question_base_url or args.vllm_base_url,
            args.vllm_document_base_url or args.vllm_base_url,
            args.vllm_think_base_url or args.vllm_base_url,
        )
        pool = VLLMExtractorPool(args)
    else:
        logger.info("[online_feedback] graph backend=local")
        pool = ExtractorPool(args)

    combined_rows: List[Dict[str, Any]] = []
    use_overlap = args.graph_backend == "vllm" and int(args.graph_overlap_workers) > 1
    overlap_workers = max(1, int(args.graph_overlap_workers)) if use_overlap else 1
    with ThreadPoolExecutor(max_workers=overlap_workers) as overlap_executor:
        setattr(args, "_graph_overlap_executor", overlap_executor)
        for d_idx, dataset in enumerate(args.datasets):
            per_limit = dataset_limits[d_idx] if dataset_limits else None
            if args.limit and args.limit > 0:
                per_limit = int(args.limit)
            raw_rows = _load_raw_dataset(
                dataset,
                args.input_filename,
                limit=per_limit,
                start=int(args.start),
            )
            if uid_filter is not None:
                raw_rows = [
                    raw for raw in raw_rows
                    if str(raw.get("uid") or raw.get("id") or raw.get("question") or "") in uid_filter
                ]

            out_dir = output_root / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = str(args.output_suffix or "")
            sp = _suffix_part(suffix)
            cases_path = out_dir / f"online_feedback_{dataset}_cases{sp}.json"
            legacy_jsonl_path = out_dir / f"online_feedback_{dataset}_cases{sp}.jsonl"
            cases, rows, done = _load_existing_cases_rows_uids(cases_path, legacy_jsonl_path)

            for raw in tqdm(raw_rows, desc=f"online_feedback:{dataset}"):
                uid = raw.get("uid") or raw.get("id") or raw.get("question") or ""
                if uid in done:
                    continue
                row, case = run_one_sample(
                    raw=raw,
                    dataset=dataset,
                    pool=pool,
                    searchr1=searchr1,
                    encoder=encoder,
                    args=args,
                )
                rows.append(row)
                cases.append(case)
                done.add(uid)
                _write_cases_json(cases_path, cases)
                _write_dataset_outputs(out_dir=out_dir, dataset=dataset, rows=rows, suffix=suffix)

            _write_dataset_outputs(out_dir=out_dir, dataset=dataset, rows=rows, suffix=suffix)
            combined_rows.extend(rows)
        delattr(args, "_graph_overlap_executor")

    if combined_rows:
        all_df = pd.DataFrame(combined_rows)
        sp = _suffix_part(str(args.output_suffix or ""))
        all_csv = output_root / f"online_feedback_all{sp}.csv"
        all_summary = output_root / f"online_feedback_all_summary{sp}.json"
        all_df.to_csv(all_csv, index=False)
        with all_summary.open("w", encoding="utf-8") as f:
            json.dump(_summarize(all_df), f, indent=2, ensure_ascii=False)
        _print_latency_summary(combined_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
