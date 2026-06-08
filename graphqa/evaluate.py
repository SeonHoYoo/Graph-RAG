"""Module 4: 평가 + 풍부한 메트릭 (AUC / baseline / LLM)."""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from graphqa.alignment import compute_sample_alignment
from graphqa.data.schema import GraphSample
from graphqa.llm_qa import (
    LLMBackend,
    answer_question_iterative,
    answer_question_triplet_fill,
    answer_question_with_llm,
    answer_question_with_verifier,
    answer_question_tasi_gated,
    DummyBackend,
    TasiGatedQAResult,
    VerifiedQAResult,
)
from graphqa.pipeline import PipelineScores, TASIPipeline
from graphqa.qa import QAResult, answer_question, score_answer
from graphqa.tasi.embedding import SentenceEncoder, get_default_encoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 단일 질문 평가
# ---------------------------------------------------------------------------
def evaluate_single(
    sample: GraphSample,
    pipeline: TASIPipeline,
    *,
    sample_index: Optional[int] = None,
    llm_backend: Optional[LLMBackend] = None,
    prompt_mode: str = "extract",
    use_verifier: bool = False,
    verifier_weights: Optional[Dict[str, float]] = None,
    abstain_threshold: float = 0.0,
    verifier_use_extract: bool = True,
    verifier_use_reason: bool = True,
    verifier_use_tasi: bool = True,
    qa_mode: str = "auto",                      # "auto" | "verifier" | "tasi_gated" | ...
    gated_k_per_slot: int = 5,
    gated_pre_threshold: float = 0.0,
    gated_post_uplift_min: float = -0.05,
    gated_enable_pre: bool = True,
    gated_enable_post: bool = True,
    inject_alignment_signal: bool = False,
    iter_abstain: bool = False,
    triplet_fill_threshold: float = 0.50,
    triplet_fill_max_steps: int = 16,
    triplet_fill_answer_on_fail: bool = False,
    triplet_fill_doc_top_k: int = 1,
    triplet_fill_think_rescue: bool = False,
    triplet_fill_evidence_scope: str = "legacy_full",
) -> Dict[str, object]:
    """한 질문에 대해 5개 TASI score + QA prediction (TASI / LLM / verifier / tasi-gated)."""
    scores: PipelineScores = pipeline.score_sample(sample)
    sample_align = compute_sample_alignment(sample, pipeline.encoder)

    extract_ans = ""
    reason_ans = ""
    verify_score = float("nan")
    verify_grounding = float("nan")
    verify_chain = float("nan")
    verify_type = float("nan")
    verify_source = ""
    abstained = False

    # tasi-gated 전용 컬럼 placeholder
    gated_pre_score = float("nan")
    gated_uplift = float("nan")
    gated_abstain_reason = ""
    gated_filled_slots = ""
    gated_raw = ""
    iter_n_llm = float("nan")
    iter_n_slot = float("nan")
    iter_planned_slots = float("nan")
    iter_mid_abstain_step_1based = float("nan")
    iter_mid_abstain_slot_id = ""
    triplet_doc_ok = float("nan")
    triplet_think_ok = float("nan")
    triplet_ok_pair = ""
    triplet_step_ok_pairs = ""
    triplet_min_doc_score = float("nan")
    triplet_min_think_score = float("nan")
    triplet_step_trace = ""
    triplet_gate_failed = False
    triplet_fail_step_1based = float("nan")
    triplet_fail_reason = ""
    triplet_answer_on_failure = False
    triplet_doc_top_k = int(max(1, int(triplet_fill_doc_top_k)))
    triplet_think_rescue_enabled = False
    triplet_n_think_rescues = 0.0
    triplet_remaining_slots = ""
    triplet_evidence_scope = triplet_fill_evidence_scope
    triplet_route_summary: Dict[str, Any] = {}
    triplet_debug: Optional[Dict[str, Any]] = None

    # mode 결정
    mode = (qa_mode or "auto").lower()
    if mode == "auto":
        mode = "verifier" if use_verifier else "single"

    if mode == "evidence":
        from graphqa.llm_qa import answer_question_evidence
        qa, ev = answer_question_evidence(
            sample, pipeline.encoder, llm_backend,
            inject_alignment_signal=inject_alignment_signal,
            sample_alignment=sample_align,
        )
        tasi_pred = qa.predicted_answer
        llm_pred = ev.final_answer
        final_pred = ev.final_answer
        em, f1 = ev.em, ev.f1
        is_correct = ev.is_correct
        is_yesno = ev.is_yesno
        n_slots = len(qa.slot_fillings)
        baseline_pred = qa.yesno_baseline_pred
        abstained = ev.abstained
        gated_pre_score = ev.cosine_min_pair        # reuse column for cosine_min
        gated_uplift = float("nan")
        gated_abstain_reason = ev.abstain_reason
        gated_filled_slots = "|".join(f"{k}={v}" for k, v in ev.llm_filled_slots.items())
        gated_raw = (ev.llm_raw_response or "")[:600]
    elif mode == "iterative":
        be = llm_backend if llm_backend is not None else DummyBackend()
        qa, it = answer_question_iterative(
            sample, pipeline.encoder, be,
            inject_step_alignment=iter_abstain,
            allow_mid_abstain=iter_abstain,
        )
        tasi_pred = qa.predicted_answer
        llm_pred = it.final_answer
        final_pred = it.final_answer
        em, f1 = it.em, it.f1
        is_correct = it.is_correct
        is_yesno = it.is_yesno
        n_slots = len(qa.slot_fillings)
        baseline_pred = qa.yesno_baseline_pred
        abstained = it.abstained
        gated_pre_score = float(it.n_llm_calls)
        gated_uplift = float(it.n_slot_steps)
        gated_abstain_reason = it.abstain_reason
        gated_filled_slots = "|".join(f"{k}={v}" for k, v in it.filled_slots.items())
        gated_raw = (it.llm_raw_trace or "")[:600]
        iter_n_llm = float(it.n_llm_calls)
        iter_n_slot = float(it.n_slot_steps)
        iter_planned_slots = float(it.iter_planned_slot_count)
        if it.abstained and it.iter_mid_abstain_step_1based > 0:
            iter_mid_abstain_step_1based = float(it.iter_mid_abstain_step_1based)
            iter_mid_abstain_slot_id = it.iter_mid_abstain_slot_id or ""
        else:
            iter_mid_abstain_step_1based = float("nan")
            iter_mid_abstain_slot_id = ""
        sa_align = copy.copy(sample)
        sa_align.Q = it.final_Q
        sample_align = compute_sample_alignment(sa_align, pipeline.encoder)
    elif mode == "triplet_fill":
        be = llm_backend if llm_backend is not None else DummyBackend()
        qa, tf = answer_question_triplet_fill(
            sample, pipeline.encoder, be,
            field_threshold=triplet_fill_threshold,
            max_steps=triplet_fill_max_steps,
            answer_on_failure=triplet_fill_answer_on_fail,
            doc_top_k=triplet_doc_top_k,
            think_rescue=triplet_fill_think_rescue,
            evidence_scope=triplet_fill_evidence_scope,
        )
        tasi_pred = qa.predicted_answer
        llm_pred = tf.final_answer
        final_pred = tf.final_answer
        em, f1 = tf.em, tf.f1
        is_correct = tf.is_correct
        is_yesno = tf.is_yesno
        n_slots = len(tf.filled_slots)
        baseline_pred = qa.yesno_baseline_pred
        abstained = tf.abstained
        gated_pre_score = tf.min_doc_score
        gated_uplift = tf.min_think_score
        gated_abstain_reason = tf.abstain_reason
        gated_filled_slots = "|".join(f"{k}={v}" for k, v in tf.filled_slots.items())
        gated_raw = (tf.llm_raw_trace or "")[:600]
        iter_n_llm = float(tf.n_llm_calls)
        iter_n_slot = float(tf.n_slot_steps)
        iter_planned_slots = float(len(tf.filled_slots))
        triplet_doc_ok = float(int(tf.doc_ok))
        triplet_think_ok = float(int(tf.think_ok))
        triplet_ok_pair = tf.ok_pair
        triplet_step_ok_pairs = tf.step_ok_pairs
        triplet_min_doc_score = tf.min_doc_score
        triplet_min_think_score = tf.min_think_score
        triplet_step_trace = tf.step_trace[:1000]
        triplet_gate_failed = bool(tf.gate_failed)
        triplet_fail_step_1based = (
            float(tf.gate_fail_step_1based)
            if tf.gate_fail_step_1based > 0 else float("nan")
        )
        triplet_fail_reason = tf.gate_fail_reason
        triplet_answer_on_failure = bool(tf.answer_on_failure)
        triplet_think_rescue_enabled = bool(tf.think_rescue_enabled)
        triplet_n_think_rescues = float(tf.n_think_rescue)
        triplet_remaining_slots = "|".join(tf.remaining_slots)
        triplet_evidence_scope = tf.evidence_scope
        triplet_route_summary = dict(tf.route_summary or {})
        triplet_debug = dict(tf.debug or {})
        triplet_debug.update({
            "problem_index": int(sample_index) if sample_index is not None else None,
            "dataset": sample.dataset,
            "uid": sample.uid,
            "question_id": f"{sample.dataset}/{sample.uid}",
            "question": sample.question,
            "ground_truth_answer": sample.answer,
            "ground_truth_aliases": list(sample.answer_aliases),
            "tasi_answer": tasi_pred,
            "dataset_llm_predicted_answer": sample.predicted_answer,
        })
        sa_align = copy.copy(sample)
        sa_align.Q = tf.final_Q
        sample_align = compute_sample_alignment(sa_align, pipeline.encoder)
    elif mode == "tasi_gated":
        qa, gated = answer_question_tasi_gated(
            sample, pipeline.encoder, llm_backend,
            pipeline=pipeline,
            k_per_slot=gated_k_per_slot,
            pre_gate_threshold=gated_pre_threshold,
            post_gate_uplift_min=gated_post_uplift_min,
            enable_pre_gate=gated_enable_pre,
            enable_post_gate=gated_enable_post,
            inject_alignment_signal=inject_alignment_signal,
            sample_alignment=sample_align,
        )
        tasi_pred = qa.predicted_answer
        llm_pred = gated.final_answer
        final_pred = gated.final_answer
        em, f1 = gated.em, gated.f1
        is_correct = gated.is_correct
        is_yesno = gated.is_yesno
        n_slots = len(qa.slot_fillings)
        baseline_pred = qa.yesno_baseline_pred
        abstained = gated.abstained
        gated_pre_score = gated.pre_gate_score
        gated_uplift = gated.post_gate_uplift
        gated_abstain_reason = gated.abstain_reason
        gated_filled_slots = "|".join(f"{k}={v}" for k, v in gated.llm_filled_slots.items())
        gated_raw = (gated.llm_raw_response or "")[:600]
    elif use_verifier:
        qa, vres = answer_question_with_verifier(
            sample, pipeline.encoder, llm_backend,
            verifier_weights=verifier_weights,
            abstain_threshold=abstain_threshold,
            use_extract=verifier_use_extract,
            use_reason=verifier_use_reason,
            use_tasi=verifier_use_tasi,
        )
        tasi_pred = vres.tasi_answer
        extract_ans = vres.extract_answer
        reason_ans = vres.reason_answer
        llm_pred = vres.final_answer
        final_pred = vres.final_answer
        em, f1 = vres.em, vres.f1
        is_correct = vres.is_correct
        is_yesno = qa.is_yesno
        n_slots = len(qa.slot_fillings)
        baseline_pred = qa.yesno_baseline_pred
        verify_score = vres.verify_score
        verify_grounding = vres.verify_grounding
        verify_chain = vres.verify_chain
        verify_type = vres.verify_type
        verify_source = vres.verify_source
        abstained = vres.abstained
    elif llm_backend is None or isinstance(llm_backend, DummyBackend):
        qa = answer_question(sample, pipeline.encoder)
        tasi_pred = qa.predicted_answer
        llm_pred = ""
        final_pred = tasi_pred
        em, f1 = qa.em, qa.f1
        is_correct = qa.is_correct
        is_yesno = qa.is_yesno
        n_slots = len(qa.slot_fillings)
        baseline_pred = qa.yesno_baseline_pred
    else:
        qa, llm_res = answer_question_with_llm(sample, pipeline.encoder, llm_backend,
                                               prompt_mode=prompt_mode)
        tasi_pred = qa.predicted_answer
        llm_pred = llm_res.llm_answer
        final_pred = llm_res.final_answer
        em, f1 = llm_res.em, llm_res.f1
        is_correct = llm_res.is_correct
        is_yesno = qa.is_yesno
        n_slots = len(qa.slot_fillings)
        baseline_pred = qa.yesno_baseline_pred

    # 추가: 'tasi-only' EM/F1, 'always-yes' baseline EM
    tasi_em, tasi_f1 = score_answer(tasi_pred, sample.answer, sample.answer_aliases)
    extract_em = float("nan")
    reason_em = float("nan")
    if use_verifier:
        if extract_ans:
            extract_em, _ = score_answer(extract_ans, sample.answer, sample.answer_aliases)
        else:
            extract_em = 0.0
        if reason_ans:
            reason_em, _ = score_answer(reason_ans, sample.answer, sample.answer_aliases)
        else:
            reason_em = 0.0
    if is_yesno:
        always_yes_em, _ = score_answer("yes", sample.answer, sample.answer_aliases)
    else:
        always_yes_em = 0.0
    # LLM (원래 데이터셋에 들어있는 predicted_answer) 비교
    if sample.predicted_answer is not None:
        llmpred_em, _ = score_answer(str(sample.predicted_answer), sample.answer, sample.answer_aliases)
    else:
        llmpred_em = 0.0

    row: Dict[str, object] = {
        "dataset": sample.dataset,
        "problem_index": int(sample_index) if sample_index is not None else "",
        "uid": sample.uid,
        "question_id": f"{sample.dataset}/{sample.uid}",
        "question_text": sample.question,
        "ground_truth_answer": sample.answer,
        "ground_truth_aliases": "|".join(sample.answer_aliases),
        "tasi_answer": tasi_pred,
        "llm_answer": llm_pred,
        "extract_answer": extract_ans,
        "reason_answer": reason_ans,
        "predicted_answer": final_pred,
        "llm_predicted_answer": sample.predicted_answer,
        "is_correct": is_correct,
        "em": em,
        "f1": f1,
        "tasi_em": tasi_em,
        "tasi_f1": tasi_f1,
        "extract_em": extract_em,
        "reason_em": reason_em,
        "always_yes_em": always_yes_em,
        "llmpred_em": llmpred_em,
        "is_yesno": is_yesno,
        "verify_score": verify_score,
        "verify_grounding": verify_grounding,
        "verify_chain": verify_chain,
        "verify_type": verify_type,
        "verify_source": verify_source,
        "abstained": abstained,
        "qa_mode": mode,
        "gated_pre_score": gated_pre_score,
        "gated_uplift": gated_uplift,
        "gated_abstain_reason": gated_abstain_reason,
        "gated_llm_slots": gated_filled_slots,
        "gated_llm_raw": gated_raw,
        "n_hops": sample.num_hops,
        "n_Q": len(sample.Q),
        "n_T": len(sample.T),
        "n_Sr": len(sample.Sr),
        "n_D": len(sample.D),
        "n_steps": len(sample.T_steps),
        "n_slot_fillings": n_slots,
        "yesno_baseline_pred": baseline_pred,
        "iter_n_llm_calls": iter_n_llm,
        "iter_n_slot_steps": iter_n_slot,
        "iter_planned_slots": iter_planned_slots,
        "iter_mid_abstain_step_1based": iter_mid_abstain_step_1based,
        "iter_mid_abstain_slot_id": iter_mid_abstain_slot_id,
        "triplet_doc_ok": triplet_doc_ok,
        "triplet_think_ok": triplet_think_ok,
        "triplet_ok_pair": triplet_ok_pair,
        "triplet_step_ok_pairs": triplet_step_ok_pairs,
        "triplet_min_doc_score": triplet_min_doc_score,
        "triplet_min_think_score": triplet_min_think_score,
        "triplet_step_trace": triplet_step_trace,
        "triplet_gate_failed": triplet_gate_failed,
        "triplet_fail_step_1based": triplet_fail_step_1based,
        "triplet_fail_reason": triplet_fail_reason,
        "triplet_answer_on_failure": triplet_answer_on_failure,
        "triplet_doc_top_k": triplet_doc_top_k,
        "triplet_think_rescue_enabled": triplet_think_rescue_enabled,
        "triplet_n_think_rescues": triplet_n_think_rescues,
        "triplet_remaining_slots": triplet_remaining_slots,
        "triplet_evidence_scope": triplet_evidence_scope,
        "triplet_route_doc_strict_available_steps": int((triplet_route_summary.get("doc") or {}).get("strict_available", 0)),
        "triplet_route_doc_prefix_available_steps": int((triplet_route_summary.get("doc") or {}).get("prefix_available", 0)),
        "triplet_route_doc_future_only_steps": int((triplet_route_summary.get("doc") or {}).get("future_only", 0)),
        "triplet_route_doc_unavailable_steps": int((triplet_route_summary.get("doc") or {}).get("unavailable", 0)),
        "triplet_route_doc_anywhere_available_steps": int((triplet_route_summary.get("doc") or {}).get("anywhere_available", 0)),
        "triplet_route_doc_order_mismatch_rate": float((triplet_route_summary.get("doc") or {}).get("order_mismatch_rate", 0.0)),
        "triplet_route_doc_longest_exact_prefix": int((triplet_route_summary.get("doc") or {}).get("longest_exact_route_prefix", 0)),
        "triplet_route_think_strict_available_steps": int((triplet_route_summary.get("think") or {}).get("strict_available", 0)),
        "triplet_route_think_prefix_available_steps": int((triplet_route_summary.get("think") or {}).get("prefix_available", 0)),
        "triplet_route_think_future_only_steps": int((triplet_route_summary.get("think") or {}).get("future_only", 0)),
        "triplet_route_think_unavailable_steps": int((triplet_route_summary.get("think") or {}).get("unavailable", 0)),
        "triplet_route_think_anywhere_available_steps": int((triplet_route_summary.get("think") or {}).get("anywhere_available", 0)),
        "triplet_route_think_order_mismatch_rate": float((triplet_route_summary.get("think") or {}).get("order_mismatch_rate", 0.0)),
        "triplet_route_think_longest_exact_prefix": int((triplet_route_summary.get("think") or {}).get("longest_exact_route_prefix", 0)),
        "triplet_route_both_longest_exact_prefix": int(triplet_route_summary.get("both_longest_exact_route_prefix", 0)),
        # 5 TASI scores
        "relevance_score": scores.relevance.tasi,
        "consistency_score": scores.consistency.tasi,
        "alignment_score": scores.alignment.tasi,
        "search_quality_score": scores.search_quality.tasi,
        "retrieval_score": scores.retrieval.tasi,
        "total_tasi_score": scores.total_sum,
    }
    row.update(scores.to_flat_dict())
    row.update(sample_align.to_row())
    if mode == "iterative":
        row["alignment_signal_injected"] = bool(iter_abstain)
    elif mode == "triplet_fill":
        row["alignment_signal_injected"] = False
        if triplet_debug is not None:
            triplet_debug.setdefault("final", {})
            if isinstance(triplet_debug["final"], dict):
                triplet_debug["final"].update({
                    "predicted_answer": final_pred,
                    "is_correct": bool(is_correct),
                    "em": float(em),
                    "f1": float(f1),
                    "abstained": bool(abstained),
                })
            row["_triplet_debug"] = triplet_debug
    else:
        row["alignment_signal_injected"] = bool(inject_alignment_signal)
    return row


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
SCORE_COLS = [
    "relevance_score",
    "consistency_score",
    "alignment_score",
    "search_quality_score",
    "retrieval_score",
    "total_tasi_score",
    "total_product",
    "total_sum",
]


@dataclass
class EvalSummary:
    n_samples: int = 0
    accuracy: float = 0.0           # final (LLM-aware)
    em_mean: float = 0.0
    f1_mean: float = 0.0
    yesno_acc: float = 0.0
    open_acc: float = 0.0

    tasi_only_acc: float = 0.0
    tasi_only_f1: float = 0.0
    always_yes_acc_among_yesno: float = 0.0   # baseline
    always_yes_acc_overall: float = 0.0
    llmpred_acc: float = 0.0                  # 데이터셋 LLM-pred 정확도

    # verifier-mode 전용 (use_verifier=True 일 때만 의미 있음)
    extract_only_acc: float = 0.0
    reason_only_acc: float = 0.0
    abstain_rate: float = 0.0
    accuracy_when_answered: float = 0.0
    abstain_reason_distribution: Dict[str, float] = field(default_factory=dict)
    # iterative + 중간 abstain: 몇 번째 슬롯(1-based)·슬롯 id 별 건수 (전체 샘플 대비 아님, 건수)
    iter_mid_abstain_step_counts: Dict[str, int] = field(default_factory=dict)
    iter_mid_abstain_slot_counts: Dict[str, int] = field(default_factory=dict)
    triplet_gate_fail_rate: float = 0.0
    triplet_answered_after_fail_rate: float = 0.0
    triplet_fail_reason_counts: Dict[str, int] = field(default_factory=dict)
    triplet_fail_step_counts: Dict[str, int] = field(default_factory=dict)
    triplet_think_rescue_sample_count: int = 0
    triplet_think_rescue_step_count: int = 0
    source_distribution: Dict[str, float] = field(default_factory=dict)
    source_accuracy: Dict[str, float] = field(default_factory=dict)
    oracle_best_acc: float = 0.0          # 후보들 중 하나라도 맞으면 1
    verify_auc: float = 0.0               # verify_score 의 정답분리 AUC

    correct_score_means: Dict[str, float] = field(default_factory=dict)
    incorrect_score_means: Dict[str, float] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)
    aucs: Dict[str, float] = field(default_factory=dict)
    hop_breakdown: Dict[int, Dict[str, float]] = field(default_factory=dict)


def _safe_auc(y_true, y_score) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _summarize(df: pd.DataFrame) -> EvalSummary:
    if df.empty:
        return EvalSummary()
    correct_mask = df["is_correct"].astype(bool)
    yesno_mask = df["is_yesno"].astype(bool)

    correct_means = {c: float(df.loc[correct_mask, c].mean())
                     for c in SCORE_COLS if c in df.columns}
    incorrect_means = {c: float(df.loc[~correct_mask, c].mean())
                       for c in SCORE_COLS if c in df.columns}

    correlations: Dict[str, float] = {}
    aucs: Dict[str, float] = {}
    y_true = df["is_correct"].astype(int).values
    for c in SCORE_COLS:
        if c in df.columns and df[c].std() > 0 and df["is_correct"].std() > 0:
            correlations[c] = float(df[c].corr(df["is_correct"].astype(float)))
            aucs[c] = _safe_auc(y_true, df[c].values)

    hop_break: Dict[int, Dict[str, float]] = {}
    for hop, sub in df.groupby("n_hops"):
        hop_break[int(hop)] = {
            "n": int(len(sub)),
            "acc": float(sub["is_correct"].mean()),
            "f1": float(sub["f1"].mean()),
            "tasi_only_acc": float(sub["tasi_em"].mean()) if "tasi_em" in sub.columns else float("nan"),
        }

    extract_acc = float(df["extract_em"].mean()) if "extract_em" in df.columns else 0.0
    reason_acc = float(df["reason_em"].mean()) if "reason_em" in df.columns else 0.0

    abstain_rate = 0.0
    acc_when_answered = float(df["is_correct"].mean())
    abstain_reason_dist: Dict[str, float] = {}
    if "abstained" in df.columns:
        abst = df["abstained"].astype(bool)
        if len(df) > 0:
            abstain_rate = float(abst.mean())
        non_abst = df.loc[~abst]
        if len(non_abst) > 0:
            acc_when_answered = float(non_abst["is_correct"].mean())
        if "gated_abstain_reason" in df.columns and abst.any():
            sub = df.loc[abst, "gated_abstain_reason"].fillna("").astype(str)
            for reason, group in sub.groupby(sub):
                if reason:
                    abstain_reason_dist[str(reason)] = float(len(group) / len(df))

    iter_mid_step_counts: Dict[str, int] = {}
    iter_mid_slot_counts: Dict[str, int] = {}
    if "abstained" in df.columns and "iter_mid_abstain_step_1based" in df.columns:
        abst_m = df["abstained"].astype(bool)
        if abst_m.any():
            sub_a = df.loc[abst_m]
            step_s = sub_a["iter_mid_abstain_step_1based"]
            valid = step_s.notna() & (step_s > 0)
            sub_m = sub_a.loc[valid]
            if len(sub_m) > 0:
                for step_val, grp in sub_m.groupby(
                    sub_m["iter_mid_abstain_step_1based"].astype(int)
                ):
                    iter_mid_step_counts[str(int(step_val))] = int(len(grp))
                if "iter_mid_abstain_slot_id" in sub_m.columns:
                    slots = sub_m["iter_mid_abstain_slot_id"].fillna("").astype(str)
                    for sid, grp in slots[slots != ""].groupby(slots):
                        iter_mid_slot_counts[str(sid)] = int(len(grp))

    triplet_gate_fail_rate = 0.0
    triplet_answered_after_fail_rate = 0.0
    triplet_fail_reason_counts: Dict[str, int] = {}
    triplet_fail_step_counts: Dict[str, int] = {}
    triplet_think_rescue_sample_count = 0
    triplet_think_rescue_step_count = 0
    if "triplet_gate_failed" in df.columns:
        gate_failed = df["triplet_gate_failed"].fillna(False).astype(bool)
        if len(df) > 0:
            triplet_gate_fail_rate = float(gate_failed.mean())
        if "abstained" in df.columns and len(df) > 0:
            answered_after_fail = gate_failed & ~df["abstained"].fillna(False).astype(bool)
            triplet_answered_after_fail_rate = float(answered_after_fail.mean())
        if gate_failed.any() and "triplet_fail_reason" in df.columns:
            reasons = df.loc[gate_failed, "triplet_fail_reason"].fillna("").astype(str)
            reasons = reasons[reasons != ""]
            for reason, grp in reasons.groupby(reasons):
                triplet_fail_reason_counts[str(reason)] = int(len(grp))
        if gate_failed.any() and "triplet_fail_step_1based" in df.columns:
            steps = df.loc[gate_failed, "triplet_fail_step_1based"]
            valid = steps.notna() & (steps > 0)
            step_vals = steps.loc[valid].astype(int)
            for step_val, grp in step_vals.groupby(step_vals):
                triplet_fail_step_counts[str(int(step_val))] = int(len(grp))
    if "triplet_n_think_rescues" in df.columns:
        rescue_counts = df["triplet_n_think_rescues"].fillna(0).astype(float)
        triplet_think_rescue_sample_count = int((rescue_counts > 0).sum())
        triplet_think_rescue_step_count = int(rescue_counts.sum())

    source_dist: Dict[str, float] = {}
    source_acc: Dict[str, float] = {}
    if "verify_source" in df.columns:
        src_series = df["verify_source"].fillna("").astype(str)
        mask = src_series != ""
        sub_df = df.loc[mask]
        if len(sub_df) > 0:
            denom = float(len(sub_df))
            for src, sub in sub_df.groupby("verify_source"):
                source_dist[str(src)] = float(len(sub) / denom)
                source_acc[str(src)] = float(sub["is_correct"].mean())

    oracle_best = 0.0
    if {"extract_em", "reason_em", "tasi_em"}.issubset(df.columns):
        cand_em = df[["extract_em", "reason_em", "tasi_em"]].fillna(0.0).max(axis=1)
        oracle_best = float(cand_em.mean())

    verify_auc = float("nan")
    if "verify_score" in df.columns and df["verify_score"].notna().any():
        try:
            verify_auc = _safe_auc(y_true, df["verify_score"].fillna(0.0).values)
        except Exception:
            verify_auc = float("nan")

    return EvalSummary(
        n_samples=int(len(df)),
        accuracy=float(df["is_correct"].mean()),
        em_mean=float(df["em"].mean()),
        f1_mean=float(df["f1"].mean()),
        yesno_acc=float(df.loc[yesno_mask, "is_correct"].mean()) if yesno_mask.any() else 0.0,
        open_acc=float(df.loc[~yesno_mask, "is_correct"].mean()) if (~yesno_mask).any() else 0.0,
        tasi_only_acc=float(df["tasi_em"].mean()) if "tasi_em" in df.columns else 0.0,
        tasi_only_f1=float(df["tasi_f1"].mean()) if "tasi_f1" in df.columns else 0.0,
        always_yes_acc_among_yesno=float(df.loc[yesno_mask, "always_yes_em"].mean())
            if yesno_mask.any() else 0.0,
        always_yes_acc_overall=float(df["always_yes_em"].mean())
            if "always_yes_em" in df.columns else 0.0,
        llmpred_acc=float(df["llmpred_em"].mean()) if "llmpred_em" in df.columns else 0.0,
        extract_only_acc=extract_acc,
        reason_only_acc=reason_acc,
        abstain_rate=abstain_rate,
        accuracy_when_answered=acc_when_answered,
          abstain_reason_distribution=abstain_reason_dist,
          iter_mid_abstain_step_counts=iter_mid_step_counts,
          iter_mid_abstain_slot_counts=iter_mid_slot_counts,
          triplet_gate_fail_rate=triplet_gate_fail_rate,
          triplet_answered_after_fail_rate=triplet_answered_after_fail_rate,
          triplet_fail_reason_counts=triplet_fail_reason_counts,
          triplet_fail_step_counts=triplet_fail_step_counts,
          triplet_think_rescue_sample_count=triplet_think_rescue_sample_count,
          triplet_think_rescue_step_count=triplet_think_rescue_step_count,
          source_distribution=source_dist,
        source_accuracy=source_acc,
        oracle_best_acc=oracle_best,
        verify_auc=float(verify_auc) if not (verify_auc != verify_auc) else 0.0,
        correct_score_means=correct_means,
        incorrect_score_means=incorrect_means,
        correlations=correlations,
        aucs=aucs,
        hop_breakdown=hop_break,
    )


def _summary_to_dict(s: EvalSummary) -> Dict[str, object]:
    return {
        "n_samples": s.n_samples,
        "accuracy_final": s.accuracy,
        "em_mean": s.em_mean,
        "f1_mean": s.f1_mean,
        "yesno_accuracy": s.yesno_acc,
        "open_accuracy": s.open_acc,
        "tasi_only_accuracy": s.tasi_only_acc,
        "tasi_only_f1": s.tasi_only_f1,
        "baseline_always_yes_among_yesno": s.always_yes_acc_among_yesno,
        "baseline_always_yes_overall": s.always_yes_acc_overall,
        "dataset_llm_predicted_accuracy": s.llmpred_acc,
        "extract_only_acc": s.extract_only_acc,
        "reason_only_acc": s.reason_only_acc,
        "abstain_rate": s.abstain_rate,
        "accuracy_when_answered": s.accuracy_when_answered,
        "abstain_reason_distribution": s.abstain_reason_distribution,
        "iter_mid_abstain_step_counts": s.iter_mid_abstain_step_counts,
        "iter_mid_abstain_slot_counts": s.iter_mid_abstain_slot_counts,
        "triplet_gate_fail_rate": s.triplet_gate_fail_rate,
        "triplet_answered_after_fail_rate": s.triplet_answered_after_fail_rate,
        "triplet_fail_reason_counts": s.triplet_fail_reason_counts,
        "triplet_fail_step_counts": s.triplet_fail_step_counts,
        "triplet_think_rescue_sample_count": s.triplet_think_rescue_sample_count,
        "triplet_think_rescue_step_count": s.triplet_think_rescue_step_count,
        "verifier_source_distribution": s.source_distribution,
        "verifier_source_accuracy": s.source_accuracy,
        "oracle_best_candidate_acc": s.oracle_best_acc,
        "verify_score_auc": s.verify_auc,
        "correct_score_means": s.correct_score_means,
        "incorrect_score_means": s.incorrect_score_means,
        "correlations_with_is_correct": s.correlations,
        "aucs_with_is_correct": s.aucs,
        "hop_breakdown": s.hop_breakdown,
    }


def print_summary(s: EvalSummary) -> None:
    print(f"\n=== Eval Summary (n={s.n_samples}) ===")
    print(f"  ACC final = {s.accuracy:.3f}  EM = {s.em_mean:.3f}  F1 = {s.f1_mean:.3f}")
    print(f"    yes/no acc = {s.yesno_acc:.3f},  open acc = {s.open_acc:.3f}")
    print(f"  TASI-only ACC = {s.tasi_only_acc:.3f}, F1 = {s.tasi_only_f1:.3f}")
    print(f"  Baseline always-yes (among yesno) = {s.always_yes_acc_among_yesno:.3f}")
    print(f"  Baseline always-yes (overall)     = {s.always_yes_acc_overall:.3f}")
    print(f"  Dataset LLM-pred ACC              = {s.llmpred_acc:.3f}")
    if s.abstain_rate or s.extract_only_acc or s.reason_only_acc or s.oracle_best_acc:
        print("  -- Verifier breakdown --")
        print(f"    extract-only ACC      = {s.extract_only_acc:.3f}")
        print(f"    reason-only ACC       = {s.reason_only_acc:.3f}")
        print(f"    abstain rate          = {s.abstain_rate:.3f}")
        print(f"    accuracy_when_answered= {s.accuracy_when_answered:.3f}")
        print(f"    oracle best candidate = {s.oracle_best_acc:.3f}")
        print(f"    verify_score AUC      = {s.verify_auc:.3f}")
        if s.source_distribution:
            print("    -- chosen source --")
            for src in sorted(s.source_distribution.keys()):
                pct = s.source_distribution[src]
                acc = s.source_accuracy.get(src, float("nan"))
                print(f"      {src:>14s} : pick={pct:.3f}  acc={acc:.3f}")
        if s.abstain_reason_distribution:
            print("    -- abstain reason (tasi-gated) --")
            for r in sorted(s.abstain_reason_distribution.keys()):
                print(f"      {r:>14s} : share={s.abstain_reason_distribution[r]:.3f}")
        if s.iter_mid_abstain_step_counts:
            tot = sum(s.iter_mid_abstain_step_counts.values())
            print("    -- iterative mid-abstain: step (1-based, random slot order) --")
            for k in sorted(s.iter_mid_abstain_step_counts.keys(), key=lambda x: int(x)):
                c = s.iter_mid_abstain_step_counts[k]
                frac = (c / tot) if tot else 0.0
                print(f"      step {k:>3s} : n={c:>4d}  ({frac:.1%} of logged mid-abstain)")
        if s.iter_mid_abstain_slot_counts:
            tot_s = sum(s.iter_mid_abstain_slot_counts.values())
            print("    -- iterative mid-abstain: slot id --")
            for k in sorted(s.iter_mid_abstain_slot_counts.keys(), key=str):
                c = s.iter_mid_abstain_slot_counts[k]
                frac = (c / tot_s) if tot_s else 0.0
                print(f"      {k:>14s} : n={c:>4d}  ({frac:.1%})")
    if s.triplet_gate_fail_rate or s.triplet_fail_step_counts:
        print("  -- Triplet-fill legacy gate --")
        print(f"    gate fail rate             = {s.triplet_gate_fail_rate:.3f}")
        print(f"    answered after gate fail   = {s.triplet_answered_after_fail_rate:.3f}")
        if s.triplet_think_rescue_sample_count or s.triplet_think_rescue_step_count:
            print(f"    think rescue samples       = {s.triplet_think_rescue_sample_count}")
            print(f"    think rescue steps         = {s.triplet_think_rescue_step_count}")
        if s.triplet_fail_reason_counts:
            print("    -- fail reason counts --")
            for k in sorted(s.triplet_fail_reason_counts.keys()):
                print(f"      {k:>22s} : n={s.triplet_fail_reason_counts[k]:>4d}")
        if s.triplet_fail_step_counts:
            print("    -- fail step counts (1-based) --")
            for k in sorted(s.triplet_fail_step_counts.keys(), key=lambda x: int(x)):
                print(f"      step {k:>3s} : n={s.triplet_fail_step_counts[k]:>4d}")
    print("  -- Mean scores by group (correct vs incorrect) --")
    for col in SCORE_COLS:
        c = s.correct_score_means.get(col, float("nan"))
        i = s.incorrect_score_means.get(col, float("nan"))
        delta = c - i if not (np.isnan(c) or np.isnan(i)) else float("nan")
        print(f"    {col:>22s} : correct={c:.4f}  incorrect={i:.4f}  Δ={delta:+.4f}")
    print("  -- Pearson r / AUC with is_correct --")
    for col in SCORE_COLS:
        r = s.correlations.get(col, float("nan"))
        a = s.aucs.get(col, float("nan"))
        print(f"    {col:>22s} :  r = {r:+.3f}   AUC = {a:.3f}")
    if s.hop_breakdown:
        print("  -- by n_hops --")
        for hop in sorted(s.hop_breakdown.keys()):
            d = s.hop_breakdown[hop]
            print(f"    hops={hop:>2d}  n={d['n']:>4d}  acc={d['acc']:.3f}  f1={d['f1']:.3f}  tasi_only={d.get('tasi_only_acc', float('nan')):.3f}")


# ---------------------------------------------------------------------------
# 전체 평가
# ---------------------------------------------------------------------------
def _json_ready(obj: Any) -> Any:
    """Convert numpy scalars and NaN/Inf to JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return obj


def _save_triplet_debug_artifacts(
    df: pd.DataFrame,
    debug_records: List[Dict[str, Any]],
    *,
    output_dir: Path,
    output_name: str,
) -> None:
    """Write detailed triplet-fill trace JSON and compact death-step tables."""
    debug_path = output_dir / f"{output_name}_triplet_debug.json"
    payload = {
        "schema_version": 1,
        "description": (
            "Detailed trace for qa-mode=triplet_fill. The legacy_gate block "
            "records where doc-only slot filling would have stopped; when "
            "answer_on_failure is true, final.answer is still attempted from "
            "the partial chain."
        ),
        "n_records": int(len(debug_records)),
        "records": debug_records,
    }
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, ensure_ascii=False)

    death_cols = [
        "problem_index", "dataset", "uid", "question_id", "question_text",
        "ground_truth_answer", "predicted_answer", "is_correct", "em", "f1",
        "abstained", "triplet_answer_on_failure", "triplet_gate_failed",
        "triplet_doc_top_k", "triplet_think_rescue_enabled",
        "triplet_n_think_rescues", "triplet_fail_step_1based", "triplet_fail_reason",
        "triplet_ok_pair", "triplet_step_ok_pairs",
        "triplet_min_doc_score", "triplet_min_think_score",
        "triplet_remaining_slots", "triplet_evidence_scope",
        "triplet_route_doc_strict_available_steps",
        "triplet_route_doc_prefix_available_steps",
        "triplet_route_doc_future_only_steps",
        "triplet_route_doc_unavailable_steps",
        "triplet_route_doc_anywhere_available_steps",
        "triplet_route_doc_order_mismatch_rate",
        "triplet_route_doc_longest_exact_prefix",
        "triplet_route_think_strict_available_steps",
        "triplet_route_think_prefix_available_steps",
        "triplet_route_think_future_only_steps",
        "triplet_route_think_unavailable_steps",
        "triplet_route_think_anywhere_available_steps",
        "triplet_route_think_order_mismatch_rate",
        "triplet_route_think_longest_exact_prefix",
        "triplet_route_both_longest_exact_prefix",
        "triplet_step_trace",
    ]
    cols = [c for c in death_cols if c in df.columns]
    death_df = df.loc[:, cols].copy()
    death_csv = output_dir / f"{output_name}_triplet_death_steps.csv"
    death_df.to_csv(death_csv, index=False)

    fail = df["triplet_gate_failed"].fillna(False).astype(bool) \
        if "triplet_gate_failed" in df.columns else pd.Series([], dtype=bool)
    answered_after_fail = (
        fail & ~df["abstained"].fillna(False).astype(bool)
        if "abstained" in df.columns and len(fail) else pd.Series([], dtype=bool)
    )
    by_reason: Dict[str, int] = {}
    if "triplet_fail_reason" in df.columns and len(fail):
        reasons = df.loc[fail, "triplet_fail_reason"].fillna("").astype(str)
        for reason, grp in reasons[reasons != ""].groupby(reasons[reasons != ""]):
            by_reason[str(reason)] = int(len(grp))
    by_step: Dict[str, int] = {}
    if "triplet_fail_step_1based" in df.columns and len(fail):
        steps = df.loc[fail, "triplet_fail_step_1based"]
        valid = steps.notna() & (steps > 0)
        for step, grp in steps.loc[valid].astype(int).groupby(steps.loc[valid].astype(int)):
            by_step[str(int(step))] = int(len(grp))
    death_summary = {
        "n_samples": int(len(df)),
        "n_gate_failed": int(fail.sum()) if len(fail) else 0,
        "gate_fail_rate": float(fail.mean()) if len(fail) else 0.0,
        "n_answered_after_gate_failure": int(answered_after_fail.sum())
            if len(answered_after_fail) else 0,
        "by_fail_reason": by_reason,
        "by_fail_step_1based": by_step,
        "csv": str(death_csv),
        "debug_json": str(debug_path),
    }
    death_json = output_dir / f"{output_name}_triplet_death_steps.json"
    with open(death_json, "w", encoding="utf-8") as f:
        json.dump(_json_ready(death_summary), f, indent=2, ensure_ascii=False)
    route_cols = [
        c for c in df.columns
        if c in {
            "problem_index", "dataset", "uid", "question_id", "is_correct",
            "em", "f1", "triplet_evidence_scope", "triplet_step_ok_pairs",
        } or c.startswith("triplet_route_")
    ]
    if route_cols:
        route_df = df.loc[:, route_cols].copy()
        route_csv = output_dir / f"{output_name}_triplet_route_alignment.csv"
        route_df.to_csv(route_csv, index=False)
        route_summary: Dict[str, Any] = {
            "n_samples": int(len(df)),
            "csv": str(route_csv),
        }
        for kind in ("doc", "think"):
            prefix = f"triplet_route_{kind}_"
            kind_summary: Dict[str, Any] = {}
            for name in (
                "strict_available_steps",
                "prefix_available_steps",
                "future_only_steps",
                "unavailable_steps",
                "anywhere_available_steps",
            ):
                col = prefix + name
                if col in df.columns:
                    kind_summary[name] = int(df[col].fillna(0).astype(float).sum())
            mismatch_col = prefix + "order_mismatch_rate"
            if mismatch_col in df.columns:
                kind_summary["mean_order_mismatch_rate"] = float(
                    df[mismatch_col].fillna(0).astype(float).mean()
                )
            longest_col = prefix + "longest_exact_prefix"
            if longest_col in df.columns:
                vals = df[longest_col].fillna(0).astype(int)
                kind_summary["mean_longest_exact_prefix"] = float(vals.mean())
                kind_summary["longest_exact_prefix_counts"] = {
                    str(int(k)): int(v) for k, v in vals.value_counts().sort_index().items()
                }
            route_summary[kind] = kind_summary
        both_col = "triplet_route_both_longest_exact_prefix"
        if both_col in df.columns:
            vals = df[both_col].fillna(0).astype(int)
            route_summary["both_longest_exact_prefix_counts"] = {
                str(int(k)): int(v) for k, v in vals.value_counts().sort_index().items()
            }
        route_json = output_dir / f"{output_name}_triplet_route_alignment.json"
        with open(route_json, "w", encoding="utf-8") as f:
            json.dump(_json_ready(route_summary), f, indent=2, ensure_ascii=False)
        logger.info(
            "[evaluate] saved triplet route alignment artifacts %s / %s",
            route_csv, route_json,
        )
    logger.info(
        "[evaluate] saved triplet debug %s and death-step artifacts %s / %s",
        debug_path, death_csv, death_json,
    )


def evaluate_all(
    samples: Sequence[GraphSample],
    pipeline: Optional[TASIPipeline] = None,
    *,
    llm_backend: Optional[LLMBackend] = None,
    output_dir: Optional[Path] = None,
    output_name: str = "tasi_eval",
    save_plots: bool = False,
    prompt_mode: str = "extract",
    use_verifier: bool = False,
    verifier_weights: Optional[Dict[str, float]] = None,
    abstain_threshold: float = 0.0,
    verifier_use_extract: bool = True,
    verifier_use_reason: bool = True,
    verifier_use_tasi: bool = True,
    qa_mode: str = "auto",
    gated_k_per_slot: int = 5,
    gated_pre_threshold: float = 0.0,
    gated_post_uplift_min: float = -0.05,
    gated_enable_pre: bool = True,
    gated_enable_post: bool = True,
    inject_alignment_signal: bool = False,
    iter_abstain: bool = False,
    triplet_fill_threshold: float = 0.50,
    triplet_fill_max_steps: int = 16,
    triplet_fill_answer_on_fail: bool = False,
    triplet_fill_doc_top_k: int = 1,
    triplet_fill_think_rescue: bool = False,
    triplet_fill_evidence_scope: str = "legacy_full",
) -> "tuple[pd.DataFrame, EvalSummary]":
    if pipeline is None:
        pipeline = TASIPipeline()

    rows: List[Dict[str, object]] = []
    triplet_debug_records: List[Dict[str, Any]] = []
    iterator = tqdm(samples, desc="evaluate", dynamic_ncols=True)
    for sample_index, s in enumerate(iterator):
        try:
            row = evaluate_single(
                s, pipeline,
                sample_index=sample_index,
                llm_backend=llm_backend,
                prompt_mode=prompt_mode,
                use_verifier=use_verifier,
                verifier_weights=verifier_weights,
                abstain_threshold=abstain_threshold,
                verifier_use_extract=verifier_use_extract,
                verifier_use_reason=verifier_use_reason,
                verifier_use_tasi=verifier_use_tasi,
                qa_mode=qa_mode,
                gated_k_per_slot=gated_k_per_slot,
                gated_pre_threshold=gated_pre_threshold,
                gated_post_uplift_min=gated_post_uplift_min,
                gated_enable_pre=gated_enable_pre,
                gated_enable_post=gated_enable_post,
                inject_alignment_signal=inject_alignment_signal,
                iter_abstain=iter_abstain,
                triplet_fill_threshold=triplet_fill_threshold,
                triplet_fill_max_steps=triplet_fill_max_steps,
                triplet_fill_answer_on_fail=triplet_fill_answer_on_fail,
                triplet_fill_doc_top_k=triplet_fill_doc_top_k,
                triplet_fill_think_rescue=triplet_fill_think_rescue,
                triplet_fill_evidence_scope=triplet_fill_evidence_scope,
            )
            debug_record = row.pop("_triplet_debug", None)
            if isinstance(debug_record, dict):
                triplet_debug_records.append(debug_record)
            rows.append(row)
        except Exception as exc:
            logger.exception(f"sample {s.uid} failed: {exc}")
    df = pd.DataFrame(rows)

    summary = _summarize(df)
    sel_report: Dict[str, object] = {}
    if "verify_score" in df.columns and df["verify_score"].notna().any():
        sel_report = selective_report(df)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{output_name}.csv"
        df.to_csv(csv_path, index=False)
        json_path = output_dir / f"{output_name}_summary.json"
        out_dict = _summary_to_dict(summary)
        if sel_report:
            out_dict["selective_prediction"] = sel_report
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"[evaluate] saved {csv_path} and {json_path}")
        if save_plots:
            _save_plots(df, output_dir, output_name)
        if triplet_debug_records:
            _save_triplet_debug_artifacts(
                df,
                triplet_debug_records,
                output_dir=output_dir,
                output_name=output_name,
            )

    if sel_report:
        print_selective_report(sel_report)

    return df, summary


# ---------------------------------------------------------------------------
# Selective prediction: threshold sweep, risk-coverage curve, AURC
# ---------------------------------------------------------------------------
DEFAULT_SWEEP_THRESHOLDS: Sequence[float] = (
    0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
)
DEFAULT_REPORT_COVERAGES: Sequence[float] = (0.30, 0.50, 0.70, 0.90, 1.00)


def _selective_at_threshold(df: pd.DataFrame,
                            tau: float,
                            score_col: str = "verify_score") -> Dict[str, float]:
    """`score_col` >= tau 인 sample 만 답한다고 가정했을 때의 지표."""
    if df.empty or score_col not in df.columns:
        return {"threshold": float(tau), "coverage": 0.0,
                "answered": 0, "abstained": 0,
                "acc_when_answered": float("nan"),
                "acc_overall": float("nan"),
                "f1_when_answered": float("nan"),
                "risk": float("nan"),
                "score_col": score_col}
    n = len(df)
    answered_mask = df[score_col].fillna(-1.0) >= tau
    n_ans = int(answered_mask.sum())
    n_abs = n - n_ans
    coverage = n_ans / n if n else 0.0
    if n_ans == 0:
        return {"threshold": float(tau), "coverage": 0.0,
                "answered": 0, "abstained": int(n_abs),
                "acc_when_answered": float("nan"),
                "acc_overall": 0.0,
                "f1_when_answered": float("nan"),
                "risk": float("nan"),
                "score_col": score_col}
    sub = df.loc[answered_mask]
    acc_ans = float(sub["is_correct"].mean())
    f1_ans = float(sub["f1"].mean()) if "f1" in sub.columns else float("nan")
    # acc_overall: 답한 sample 만 정답으로 카운트, abstain 은 0 으로 카운트
    acc_overall = float(sub["is_correct"].sum() / n)
    return {
        "threshold": float(tau),
        "coverage": float(coverage),
        "answered": int(n_ans),
        "abstained": int(n_abs),
        "acc_when_answered": acc_ans,
        "acc_overall": acc_overall,
        "f1_when_answered": f1_ans,
        "risk": float(1.0 - acc_ans),
        "score_col": score_col,
    }


def selective_sweep(
    df: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    score_col: str = "verify_score",
) -> List[Dict[str, float]]:
    return [_selective_at_threshold(df, t, score_col=score_col) for t in thresholds]


def percentile_sweep(
    df: pd.DataFrame,
    coverages: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    score_col: str = "verify_score",
) -> List[Dict[str, float]]:
    """절대 threshold 대신 'score 상위 X% 만 답한다' 로 sweep.

    score column 마다 분포가 다른 경우 (TASI 5-score 등) 동일 grid 비교에 유용.
    """
    if df.empty or score_col not in df.columns:
        return []
    s = df[score_col].fillna(-1.0)
    n = len(df)
    out: List[Dict[str, float]] = []
    for cov in coverages:
        cov = float(max(0.0, min(1.0, cov)))
        if cov <= 0.0:
            continue
        k = max(1, int(round(cov * n)))
        tau = float(s.nlargest(k).iloc[-1]) if k <= n else float(s.min())
        rec = _selective_at_threshold(df, tau, score_col=score_col)
        rec["target_coverage"] = cov
        out.append(rec)
    return out


def _risk_coverage_curve(df: pd.DataFrame,
                         score_col: str = "verify_score") -> Dict[str, List[float]]:
    """`score_col` 을 내림차순으로 정렬해 누적 coverage / risk 곡선 산출.

    가장 confidence 높은 답부터 1개씩 채택할 때의 selective accuracy.
    """
    if df.empty or score_col not in df.columns:
        return {"coverage": [], "risk": [], "acc_when_answered": []}
    sub = df[[score_col, "is_correct"]].copy()
    sub[score_col] = sub[score_col].fillna(-1.0)
    sub = sub.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(sub)
    correct_cum = sub["is_correct"].astype(int).cumsum().to_numpy()
    coverage = np.arange(1, n + 1) / float(n)
    acc_when_answered = correct_cum / np.arange(1, n + 1)
    risk = 1.0 - acc_when_answered
    return {
        "coverage": [float(c) for c in coverage],
        "acc_when_answered": [float(a) for a in acc_when_answered],
        "risk": [float(r) for r in risk],
    }


def _aurc(curve: Dict[str, List[float]]) -> float:
    """Area under risk–coverage curve. 작을수록 좋음."""
    cov = curve.get("coverage", [])
    risk = curve.get("risk", [])
    if len(cov) < 2:
        return float("nan")
    cov_a = np.asarray(cov)
    risk_a = np.asarray(risk)
    return float(np.trapz(risk_a, cov_a))


def _e_aurc(curve: Dict[str, List[float]], overall_acc: float) -> float:
    """Excess AURC: random selector (= 항상 overall risk) 대비 얼마나 낮은가.

    random AURC = (1 - overall_acc).
    e_aurc = random_aurc - aurc  (양수면 verifier 가 random 보다 좋음).
    """
    a = _aurc(curve)
    if a != a:
        return float("nan")
    random_aurc = float(1.0 - overall_acc)
    return float(random_aurc - a)


def _acc_at_coverage(curve: Dict[str, List[float]],
                     target_coverages: Sequence[float] = DEFAULT_REPORT_COVERAGES,
                     ) -> Dict[str, float]:
    cov = curve.get("coverage", [])
    acc = curve.get("acc_when_answered", [])
    if not cov or not acc:
        return {f"acc@cov={c:.2f}": float("nan") for c in target_coverages}
    cov_a = np.asarray(cov)
    acc_a = np.asarray(acc)
    out: Dict[str, float] = {}
    for c in target_coverages:
        idx = int(np.searchsorted(cov_a, c, side="left"))
        idx = min(max(idx, 0), len(cov_a) - 1)
        out[f"acc@cov={c:.2f}"] = float(acc_a[idx])
    return out


def selective_report(df: pd.DataFrame,
                     thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
                     score_col: str = "verify_score",
                     coverage_grid: Sequence[float] = (
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     ),
                     ) -> Dict[str, object]:
    """selective prediction 에 관한 모든 지표를 묶어서 dict 로 반환.

    score_col 을 바꾸면 임의 신호 (relevance_score, total_sum 등) 로 같은
    분석을 재실행할 수 있다.
    """
    if df.empty or score_col not in df.columns:
        return {
            "score_col": score_col,
            "threshold_sweep": [],
            "coverage_sweep": [],
            "risk_coverage_curve": {"coverage": [], "risk": [], "acc_when_answered": []},
            "aurc": float("nan"),
            "e_aurc": float("nan"),
            "acc_at_coverage": {},
            "score_auc": float("nan"),
        }
    curve = _risk_coverage_curve(df, score_col=score_col)
    overall_acc = float(df["is_correct"].mean())
    aurc_v = _aurc(curve)
    e_aurc_v = _e_aurc(curve, overall_acc)
    acc_at = _acc_at_coverage(curve)
    sweep = selective_sweep(df, thresholds, score_col=score_col)
    cov_sweep = percentile_sweep(df, coverage_grid, score_col=score_col)
    try:
        score_auc = _safe_auc(
            df["is_correct"].astype(int).values,
            df[score_col].fillna(0.0).values,
        )
    except Exception:
        score_auc = float("nan")
    return {
        "score_col": score_col,
        "threshold_sweep": sweep,
        "coverage_sweep": cov_sweep,
        "risk_coverage_curve": curve,
        "aurc": aurc_v,
        "e_aurc": e_aurc_v,
        "acc_at_coverage": acc_at,
        "score_auc": float(score_auc) if score_auc == score_auc else float("nan"),
    }


def print_selective_report(report: Dict[str, object]) -> None:
    score_col = report.get("score_col", "verify_score")
    print(f"\n  -- Selective prediction (score = {score_col}) --")
    sweep = report.get("threshold_sweep", []) or []
    if sweep:
        print(f"    {'τ':>6s}  {'cov':>6s}  {'ans':>5s}  {'abs':>5s}  {'acc_ans':>8s}  {'acc_all':>8s}  {'risk':>6s}")
        for r in sweep:
            print(f"    {r['threshold']:>6.2f}  {r['coverage']:>6.3f}  "
                  f"{r['answered']:>5d}  {r['abstained']:>5d}  "
                  f"{r['acc_when_answered']:>8.3f}  {r['acc_overall']:>8.3f}  "
                  f"{r['risk']:>6.3f}")
    cov_sweep = report.get("coverage_sweep", []) or []
    if cov_sweep:
        print(f"    -- coverage grid (score top-X% answered) --")
        print(f"    {'cov':>6s}  {'τ':>7s}  {'acc_ans':>8s}  {'acc_all':>8s}  {'risk':>6s}")
        for r in cov_sweep:
            print(f"    {r.get('target_coverage', r['coverage']):>6.2f}  "
                  f"{r['threshold']:>7.3f}  {r['acc_when_answered']:>8.3f}  "
                  f"{r['acc_overall']:>8.3f}  {r['risk']:>6.3f}")
    print(f"    AURC  = {report.get('aurc', float('nan')):.4f}  (lower = better)")
    print(f"    E-AURC= {report.get('e_aurc', float('nan')):+.4f}  (>0 means score > random)")
    print(f"    score AUC vs is_correct = {report.get('score_auc', float('nan')):.3f}")
    aac = report.get("acc_at_coverage", {}) or {}
    if aac:
        line = "  ".join(f"{k}={v:.3f}" for k, v in aac.items())
        print(f"    {line}")


def _save_plots(df: pd.DataFrame, output_dir: Path, output_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[evaluate] matplotlib not available, skipping plots")
        return

    cols = [c for c in SCORE_COLS if c in df.columns]
    if not cols:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.ravel()
    correct_mask = df["is_correct"].astype(bool)
    for ax, col in zip(axes, cols):
        ax.hist(df.loc[correct_mask, col].dropna(), bins=20, alpha=0.6, label="correct")
        ax.hist(df.loc[~correct_mask, col].dropna(), bins=20, alpha=0.6, label="incorrect")
        ax.set_title(col)
        ax.legend()
    for i in range(len(cols), len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"TASI score distribution — {output_name}")
    fig.tight_layout()
    out = output_dir / f"{output_name}_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"[evaluate] saved plot: {out}")

    if "verify_score" in df.columns and df["verify_score"].notna().any():
        try:
            curve = _risk_coverage_curve(df)
            cov = curve.get("coverage", [])
            risk = curve.get("risk", [])
            acc = curve.get("acc_when_answered", [])
            if cov and risk:
                fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
                ax2[0].plot(cov, risk, label="verifier", lw=2)
                overall_acc = float(df["is_correct"].mean())
                ax2[0].plot([0, 1], [1.0 - overall_acc, 1.0 - overall_acc],
                            "--", color="gray", label=f"random ({1.0-overall_acc:.2f})")
                ax2[0].set_xlabel("Coverage")
                ax2[0].set_ylabel("Risk = 1 − acc(answered)")
                ax2[0].set_title("Risk–Coverage Curve")
                ax2[0].set_xlim(0, 1)
                ax2[0].legend()
                ax2[0].grid(alpha=0.3)

                ax2[1].plot(cov, acc, color="C2", lw=2)
                ax2[1].axhline(overall_acc, color="gray", ls="--",
                               label=f"overall acc = {overall_acc:.3f}")
                ax2[1].set_xlabel("Coverage")
                ax2[1].set_ylabel("Selective accuracy (answered)")
                ax2[1].set_title("Selective Accuracy vs Coverage")
                ax2[1].set_xlim(0, 1)
                ax2[1].set_ylim(0, 1)
                ax2[1].legend()
                ax2[1].grid(alpha=0.3)
                fig2.suptitle(f"Selective prediction — {output_name}  AURC={_aurc(curve):.3f}")
                fig2.tight_layout()
                out2 = output_dir / f"{output_name}_risk_coverage.png"
                fig2.savefig(out2, dpi=120)
                plt.close(fig2)
                logger.info(f"[evaluate] saved plot: {out2}")
        except Exception as exc:
            logger.warning(f"[evaluate] risk-coverage plot failed: {exc}")
