#!/usr/bin/env python3
"""
Single-process pipeline:
Load one model once, then run triplet extraction -> infill(11 settings) -> answer.
"""

import argparse
import json
import logging
import os
import sys
import time
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_INFILL_SCRIPTS_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
for _p in (_ROOT, _INFILL_SCRIPTS_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from direct import Direct
from extract_triplets import process_sample as process_triplet_sample
from infill_graphs import (
    TRIPLET_SELECTION_CHOICES,
    build_output_path,
    contains_ent_placeholder,
    infill_prompt,
)
from answer import build_answer_prompt, build_graph_evidence, normalize_predicted_answer
from model_library.construct_model import ConstructModel
from model_library.llm_clients import Qwen
from utils.metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--input_filename", type=str, default="train_sampled.json")
    p.add_argument("--bm25_top_k", type=int, default=10)
    p.add_argument("--setting", type=str, default="open-book", choices=["open-book", "open-book+gold", "gold"])
    p.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--max_trials", type=int, default=3)
    p.add_argument("--ent_exist_flag", type=str, choices=["all", "false"], default="all")
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--triplet_top_k_list", type=str, default="3,5,10")
    p.add_argument("--baseline_top_k_list", type=str, default="3,5,10")
    p.add_argument("--output_tag", type=str, default="")
    return p.parse_args()


def select_datasets(dataset_arg: str):
    return DATASETS if dataset_arg == "all" else (dataset_arg,)


def parse_top_k_list(top_k_text: str) -> list[int]:
    ks = []
    for tok in top_k_text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
        except ValueError:
            continue
        if k > 0:
            ks.append(k)
    return sorted(set(ks))


def get_model_output_dir_name(args) -> str:
    model_short = args.model_name.split("/")[-1]
    tag = (args.output_tag or "").strip()
    return f"{model_short}__{tag}" if tag else model_short


def build_question_only_prompt(question: str) -> str:
    return (
        "Answer the question. "
        "Return only a short answer phrase.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def extract_triplets_for_dataset(args, dataset, qwen_client):
    input_path = os.path.join(_ROOT, "datasets", dataset, "claims", args.input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model_short = args.model_name.split("/")[-1]
    setting_tag = args.setting.replace("+", "_")
    input_stem = os.path.splitext(args.input_filename)[0]
    triplet_filename = f"triplets_{input_stem}_{setting_tag}_top{args.bm25_top_k}.json"
    output_path = os.path.join(
        _ROOT, "results", dataset, "triplets", model_short, triplet_filename
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        input_list = json.load(f)
    if args.max_samples is not None:
        input_list = input_list[: args.max_samples]

    result_list = []
    done_indices = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            result_list = json.load(f)
        done_indices = {r["index"] for r in result_list}
        logger.info("[%s] triplet checkpoint: %d samples", dataset, len(done_indices))

    construct_model = ConstructModel(
        construct_model_name=args.model_name,
        dataset_name=dataset,
        construct_model_client=qwen_client,
    )
    direct_args = Namespace(
        dataset=dataset,
        input_filename=args.input_filename,
        direct_filename=None,
        base_model_name="google/flan-t5-xl",
        setting=args.setting,
        bm25_top_k=args.bm25_top_k,
        use_searchr1=False,
        searchr1_top_k=3,
        searchr1_max_turns=3,
        use_total_search_results=False,
        retriever_url=args.retriever_url,
    )
    direct_model = Direct(direct_args)

    new_count = 0
    for sample in tqdm(input_list, desc=f"Triplets:{dataset}"):
        if sample.get("index") in done_indices:
            continue
        try:
            result = process_triplet_sample(sample, construct_model, direct_model, args)
            if result is not None:
                result_list.append(result)
                new_count += 1
        except Exception as e:
            logger.error("[%s] sample index=%s failed: %s", dataset, sample.get("index"), e)
            continue

        if new_count > 0 and new_count % args.checkpoint_every == 0:
            result_list.sort(key=lambda x: x["index"])
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_list, f, indent=2, ensure_ascii=False)
            logger.info("[%s] triplet checkpoint saved: %s", dataset, output_path)

    result_list.sort(key=lambda x: x["index"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)
    logger.info("[%s] triplets saved: %s", dataset, output_path)
    return output_path


def run_infill_for_dataset(args, dataset, triplet_path, qwen_client):
    model_output_dir = get_model_output_dir_name(args)
    infill_output_dir = os.path.join(_ROOT, "infilling", "output", "infill", model_output_dir, dataset)
    answer_output_dir = os.path.join(_ROOT, "infilling", "output", "answer", model_output_dir, dataset)
    os.makedirs(infill_output_dir, exist_ok=True)
    os.makedirs(answer_output_dir, exist_ok=True)

    with open(triplet_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    triplet_top_ks = parse_top_k_list(args.triplet_top_k_list)
    if not triplet_top_ks:
        raise ValueError("triplet_top_k_list must contain at least one positive integer.")
    settings = [(0, f"top{k}", 0) for k in triplet_top_ks]

    for use_full_doc, triplet_selection, force_doc_only in settings:
        setting_name = f"use_full_doc={use_full_doc}, triplet_selection={triplet_selection}, force_doc_only={force_doc_only}"
        logger.info("Starting infill+answer (per-sample): %s", setting_name)
        infill_args = SimpleNamespace(
            model_name=args.model_name,
            data_file=triplet_path,
            output_dir=infill_output_dir,
            question_strategy="triplet_only",
            infill_strategy="triplet_only",
            use_gold_only=0,
            max_trials=args.max_trials,
            use_full_doc=use_full_doc,
            triplet_selection=triplet_selection,
            force_doc_only=force_doc_only,
        )
        infill_results = []
        answer_results = []

        for sample in tqdm(samples, desc="Infill+Answer"):
            sample_out = deepcopy(sample)
            prompt = None
            try:
                prompt = infill_prompt(infill_args, sample_out)
                response = None
                used_trials = 0
                for trial in range(1, args.max_trials + 1):
                    used_trials = trial
                    response = qwen_client.generate(
                        user_message=prompt,
                        max_tokens=512,
                        temperature=0.0,
                    )
                    if not contains_ent_placeholder(response):
                        break
                    if trial < args.max_trials:
                        logger.info(
                            "Sample index=%s still contains (ENT). Retrying (%d/%d).",
                            sample_out.get("index"),
                            trial + 1,
                            args.max_trials,
                        )

                ent_flag = contains_ent_placeholder(response)
                sample_out["infill_result"] = {
                    "question_strategy": infill_args.question_strategy,
                    "infill_strategy": infill_args.infill_strategy,
                    "use_full_doc": use_full_doc,
                    "force_doc_only": force_doc_only,
                    "triplet_selection": triplet_selection,
                    "trials": used_trials,
                    "ent_exist_flag": ent_flag,
                    "response": response,
                    "prompt": prompt,
                }
            except Exception as e:
                logger.exception("Failed infill for sample index=%s", sample_out.get("index"))
                sample_out["infill_result"] = {
                    "question_strategy": infill_args.question_strategy,
                    "infill_strategy": infill_args.infill_strategy,
                    "use_full_doc": use_full_doc,
                    "force_doc_only": force_doc_only,
                    "triplet_selection": triplet_selection,
                    "response": None,
                    "prompt": prompt,
                    "error": str(e),
                    "ent_exist_flag": True,
                }

            # infill 직후 같은 qwen_client로 즉시 answer
            ans_sample = deepcopy(sample_out)
            ent_flag = ans_sample.get("infill_result", {}).get("ent_exist_flag", None)
            if args.ent_exist_flag == "false" and ent_flag is not False:
                ans_sample["answer_skipped"] = True
                ans_sample["answer_skip_reason"] = "ent_exist_flag_is_not_false"
            else:
                try:
                    graph_evidence = build_graph_evidence(ans_sample)
                    answer_prompt = build_answer_prompt(ans_sample.get("question", ""), graph_evidence)
                    predicted = ""
                    for _ in range(max(1, args.max_trials)):
                        predicted = qwen_client.generate(
                            user_message=answer_prompt,
                            max_tokens=256,
                            temperature=0.0,
                        )
                        predicted = normalize_predicted_answer(predicted)
                        if predicted:
                            break

                    gt_answers = [ans_sample.get("answer", "")] + ans_sample.get("answer_aliases", [])
                    gt_answers = [x for x in gt_answers if isinstance(x, str)]
                    em_score = metric_max_over_ground_truths(compute_exact, predicted, gt_answers) if gt_answers else 0.0
                    f1_score = metric_max_over_ground_truths(compute_f1, predicted, gt_answers) if gt_answers else 0.0

                    # Keep gold aliases untouched; do not inject model prediction into gold set.
                    ans_sample["predicted_answer"] = predicted
                    ans_sample["graph_evidence"] = graph_evidence
                    ans_sample["answer_prompt"] = answer_prompt
                    ans_sample["em_score"] = em_score
                    ans_sample["f1_score"] = f1_score
                    ans_sample["answer_skipped"] = False
                    ans_sample["answer_skip_reason"] = None
                except Exception as e:
                    ans_sample["answer_skipped"] = True
                    ans_sample["answer_skip_reason"] = str(e)

            infill_results.append(sample_out)
            answer_results.append(ans_sample)

        infill_output_path = build_output_path(infill_args)
        with open(infill_output_path, "w", encoding="utf-8") as f:
            json.dump(infill_results, f, ensure_ascii=False, indent=2)
        logger.info("Infill result 결과 저장: %s", infill_output_path)

        input_stem = os.path.splitext(os.path.basename(infill_output_path))[0]
        if input_stem.startswith("infill_"):
            input_stem = input_stem[len("infill_"):]
        answer_output_path = os.path.join(
            answer_output_dir,
            f"answer_{input_stem}_{args.ent_exist_flag}.json",
        )
        with open(answer_output_path, "w", encoding="utf-8") as f:
            json.dump(answer_results, f, ensure_ascii=False, indent=2)
        logger.info("Answer result 저장: %s", answer_output_path)

    # Baseline: retrieval is still performed (open-book, non-gold),
    # but prompt uses only the question text.
    baseline_top_ks = parse_top_k_list(args.baseline_top_k_list)
    if not baseline_top_ks:
        raise ValueError("baseline_top_k_list must contain at least one positive integer.")

    direct_args = Namespace(
        dataset=dataset,
        input_filename=args.input_filename,
        direct_filename=None,
        base_model_name="google/flan-t5-xl",
        setting=args.setting,
        bm25_top_k=args.bm25_top_k,
        use_searchr1=False,
        searchr1_top_k=3,
        searchr1_max_turns=3,
        use_total_search_results=False,
        retriever_url=args.retriever_url,
    )
    direct_model = Direct(direct_args)

    for k in baseline_top_ks:
        baseline_results = []
        logger.info("Starting baseline(question-only) with retrieval top_k=%d", k)
        for sample in tqdm(samples, desc=f"Baseline(top{k})"):
            out_sample = deepcopy(sample)
            try:
                _, retrieval_info = direct_model.retrieve_evidence(
                    query=out_sample.get("question", ""),
                    gold_id_list=out_sample.get("gold_id_list", []),
                    gold_evidence_list=[],
                    top_k=k,
                )
            except Exception:
                retrieval_info = {"doc_id_list": [], "is_gold_list": []}

            question = out_sample.get("question", "")
            prompt = build_question_only_prompt(question)
            predicted = ""
            for _ in range(max(1, args.max_trials)):
                predicted = qwen_client.generate(
                    user_message=prompt,
                    max_tokens=256,
                    temperature=0.0,
                )
                predicted = normalize_predicted_answer(predicted)
                if predicted:
                    break

            gt_answers = [out_sample.get("answer", "")] + out_sample.get("answer_aliases", [])
            gt_answers = [x for x in gt_answers if isinstance(x, str)]
            em_score = metric_max_over_ground_truths(compute_exact, predicted, gt_answers) if gt_answers else 0.0
            f1_score = metric_max_over_ground_truths(compute_f1, predicted, gt_answers) if gt_answers else 0.0

            out_sample["baseline_mode"] = "question_only_openbook"
            out_sample["baseline_retrieval_top_k"] = k
            out_sample["retrieval_info"] = retrieval_info
            out_sample["predicted_answer"] = predicted
            out_sample["answer_prompt"] = prompt
            out_sample["em_score"] = em_score
            out_sample["f1_score"] = f1_score
            baseline_results.append(out_sample)

        baseline_output_path = os.path.join(
            answer_output_dir,
            f"answer_baseline_qonly_openbook_top{k}_{args.ent_exist_flag}.json",
        )
        with open(baseline_output_path, "w", encoding="utf-8") as f:
            json.dump(baseline_results, f, ensure_ascii=False, indent=2)
        logger.info("Baseline answer result 저장: %s", baseline_output_path)


def main():
    args = parse_args()
    datasets = select_datasets(args.dataset)
    logger.info("Single-process pipeline start: datasets=%s model=%s", ",".join(datasets), args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    qwen_client = Qwen(model, tokenizer)
    logger.info("Model loaded once: %s", args.model_name)

    t0 = time.time()
    for ds in datasets:
        logger.info("========== Dataset: %s ==========", ds)
        triplet_path = extract_triplets_for_dataset(args, ds, qwen_client)
        run_infill_for_dataset(args, ds, triplet_path, qwen_client)

    logger.info("Pipeline done in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
