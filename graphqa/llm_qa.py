"""LLM-augmented final QA.

설계:
  TASI 가 (1) UNKNOWN 슬롯을 1:1 매칭으로 채우고, (2) 슬롯 별로 가장 alignment
  점수가 높은 D triple 들을 evidence 로 추출한다. 이 슬롯값/evidence/원래
  질문을 LLM 에 prompt 로 넘겨 최종 답만 받는다.

  → "TASI 가 추론한 답"을 LLM 이 검증 / 자연어로 정제 / yes-no 결정 만 하는
    역할이라 LLM 부담이 작고 응답이 짧다 (max_new_tokens ≤ 32 정도).

지원 백엔드:
  - "qwen-local"  : transformers 로 Qwen2.5-* Instruct (HF 캐시에서 로드).
  - "openai"      : OpenAI API.
  - "dummy"       : LLM 호출 없이 TASI 답을 그대로 반환 (디버그용).
"""
from __future__ import annotations

import copy
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graphqa.data.schema import GraphSample, Triple, is_unknown
from graphqa.qa import (
    QAResult,
    SlotCandidate,
    answer_question,
    score_answer,
    _is_yesno_question,
    _slot_id,
)
from graphqa.tasi.embedding import SentenceEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence 추출
# ---------------------------------------------------------------------------
def _evidence_for_slots(
    fillings: Dict[str, SlotCandidate],
    D: Sequence[Triple],
    *,
    top_k: int = 6,
) -> List[str]:
    """슬롯 채움에 기여한 D triple 중 상위 K개를 evidence 텍스트로 변환."""
    pairs: List[Tuple[float, int]] = []
    seen_d: set = set()
    for cand in fillings.values():
        for q_i, d_j, sc in cand.support_pairs:
            if d_j in seen_d:
                continue
            seen_d.add(d_j)
            pairs.append((sc, d_j))
    pairs.sort(reverse=True)
    out: List[str] = []
    for _, j in pairs[:top_k]:
        if j < 0 or j >= len(D):
            continue
        t = D[j]
        line = f"- ({t.head}, {t.relation}, {t.tail})"
        if t.context:
            line += f"  [context: {t.context.strip()[:160]}]"
        out.append(line)
    return out


def _format_slots(fillings: Dict[str, SlotCandidate]) -> List[str]:
    if not fillings:
        return []
    out = []
    for slot, c in sorted(fillings.items()):
        out.append(f"  {slot} = {c.value}  (score={c.score:.2f})")
    return out


def _render_node(
    text: str,
    is_unk: bool,
    fillings: Dict[str, SlotCandidate],
) -> str:
    """Q triple 의 head/tail 한 칸을 렌더링.

    - 채워진 슬롯  → "[ENT1=Iran]"
    - 못 채운 슬롯 → "?ENT3"  (LLM 이 답으로 추출해야 할 자리)
    - 일반 토큰    → 그대로
    """
    if not is_unk:
        return text
    sid = _slot_id(text) or "ENT?"
    if sid in fillings and fillings[sid].value:
        return f"[{sid}={fillings[sid].value}]"
    return f"?{sid}"


def _filled_query_chain(
    sample: GraphSample,
    fillings: Dict[str, SlotCandidate],
) -> List[str]:
    """sample.Q triple 들에 슬롯값을 치환한 자연어 chain."""
    lines: List[str] = []
    for i, t in enumerate(sample.Q, 1):
        h = _render_node(t.head, t.head_unknown, fillings)
        tl = _render_node(t.tail, t.tail_unknown, fillings)
        line = f"  {i}. ({h}) -- {t.relation} --> ({tl})"
        if t.context:
            line += f"   [ctx: {t.context.strip()[:120]}]"
        lines.append(line)
    return lines


def _slot_type_hints(sample: GraphSample) -> List[str]:
    """Q_def 의 (ENTk, is_a, TYPE) 형태를 자연어로."""
    out: List[str] = []
    for t in sample.Q_def:
        sid = _slot_id(t.head) if t.head_unknown else None
        if not sid:
            continue
        tail = t.tail
        if is_unknown(tail):
            continue
        out.append(f"  {sid} is a {tail}")
    return out


def _unfilled_slots(
    sample: GraphSample,
    fillings: Dict[str, SlotCandidate],
) -> List[str]:
    """Q triple 에서 등장하는 unknown 슬롯 중 fillings 에 없는 것들."""
    seen: List[str] = []
    for t in sample.Q:
        for tok, unk in [(t.head, t.head_unknown), (t.tail, t.tail_unknown)]:
            if not unk:
                continue
            sid = _slot_id(tok)
            if sid and sid not in fillings and sid not in seen:
                seen.append(sid)
    return seen


def build_prompt(
    sample: GraphSample,
    fillings: Dict[str, SlotCandidate],
    tasi_answer: str,
    is_yesno: bool,
    *,
    top_k_evidence: int = 6,
    prompt_mode: str = "extract",
) -> Tuple[str, str]:
    """system / user prompt 반환.

    prompt_mode:
      - "extract" : chain 의 entity 중에서만 답을 고르라고 강하게 지시. (현재 기본)
      - "reason"  : chain·evidence 를 reasoning 단서로 *제공만* 하고, LLM 이
                    자유롭게 답을 만들도록 둠. chain 외부의 entity 도 답이 될 수 있음.
    """
    chain_lines = _filled_query_chain(sample, fillings) or ["  (none)"]
    type_lines = _slot_type_hints(sample)
    slot_lines = _format_slots(fillings) or ["  (none)"]
    evid_lines = _evidence_for_slots(fillings, sample.D, top_k=top_k_evidence) or ["  (none)"]
    unfilled = _unfilled_slots(sample, fillings)

    mode = (prompt_mode or "extract").lower()
    if mode not in ("extract", "reason"):
        mode = "extract"

    if mode == "extract":
        if is_yesno:
            instruction = (
                "Decide whether the answer to the question is 'yes' or 'no'. "
                "Use the filled query chain (with entities plugged in) and the evidence triples. "
                "Output ONLY 'yes' or 'no' on a single line — no explanation."
            )
        else:
            unfilled_hint = (
                f"The answer most likely fills the slot(s): {', '.join('?'+s for s in unfilled)}. "
                if unfilled
                else "The answer is one of the entities that appears in the filled query chain. "
            )
            instruction = (
                "Read the question and the filled query chain together. "
                f"{unfilled_hint}"
                "If a slot is shown as '?ENTk' you should output the entity that fills it; "
                "otherwise output the entity in the chain that the question is asking about. "
                "Output ONLY the final answer (a short noun phrase) on a single line — "
                "no explanation, no quotes, no extra words."
            )
        system = (
            "You are an extractive multi-hop question answering assistant. "
            "You answer questions by reading a *filled query chain* (a sequence of "
            "(head, relation, tail) triples whose UNKNOWN slots have been replaced "
            "with concrete entities) together with supporting evidence triples. "
            "Your answer must be one entity drawn from the chain or the evidence; "
            "do not invent entities that do not appear there."
        )
    else:  # reason mode
        if is_yesno:
            instruction = (
                "Use the reasoning chain and the evidence triples together with the question "
                "to decide whether the answer is 'yes' or 'no'. "
                "Output ONLY 'yes' or 'no' on a single line — no explanation."
            )
        else:
            instruction = (
                "Use the reasoning chain and the evidence triples as supporting facts. "
                "Reason internally about the multi-hop question; "
                "the chain may contain mistakes — feel free to disagree with it when the evidence "
                "or your knowledge clearly contradicts it. "
                "The answer should be a short noun phrase. "
                "Output ONLY the final answer on a single line — "
                "no explanation, no quotes, no extra words."
            )
        system = (
            "You are a multi-hop question answering assistant. "
            "You receive (a) a reasoning chain produced by a graph-alignment system "
            "(some slots may be wrong), and (b) supporting evidence triples extracted from "
            "the source documents. Use them as hints to derive the final answer to the question. "
            "Prefer answers grounded in the evidence; the chain is only auxiliary."
        )

    chain_label = (
        "Filled query chain (UNKNOWN slots replaced with concrete entities; "
        "'?ENTk' means the slot was not filled and is likely the answer):"
        if mode == "extract" else
        "Reasoning chain (graph-alignment hypothesis; entries may be noisy):"
    )
    evidence_label = (
        "Top supporting evidence triples (head, relation, tail):"
        if mode == "extract" else
        "Supporting evidence triples extracted from documents (head, relation, tail):"
    )
    fallback_label = (
        "Graph-based fallback answer (only use if chain+evidence are insufficient): "
        if mode == "extract" else
        "Graph-based candidate answer (treat as a hint, not as the truth): "
    )

    user_parts: List[str] = [
        f"Question: {sample.question}",
        "",
        chain_label,
        *chain_lines,
    ]
    if type_lines:
        user_parts += ["", "Slot type hints:", *type_lines]
    user_parts += [
        "",
        "Slot fillings (with confidence):",
        *slot_lines,
        "",
        evidence_label,
        *evid_lines,
        "",
        f"{fallback_label}{tasi_answer if tasi_answer else '(none)'}",
        "",
        instruction,
    ]
    user = "\n".join(user_parts)
    return system, user


# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------
class LLMBackend:
    name: str = "base"

    def generate(self, system: str, user: str, *, max_new_tokens: int = 32) -> str:
        raise NotImplementedError


class DummyBackend(LLMBackend):
    name = "dummy"

    def generate(self, system: str, user: str, *, max_new_tokens: int = 32) -> str:
        return ""


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> None:
        from openai import OpenAI  # type: ignore
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, system: str, user: str, *, max_new_tokens: int = 32) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning(f"[openai] {exc}")
            return ""


class QwenLocalBackend(LLMBackend):
    name = "qwen-local"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        logger.info(f"[qwen-local] loading {model_name} on {device} ({torch_dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map={"": device} if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info("[qwen-local] ready")

    def generate(self, system: str, user: str, *, max_new_tokens: int = 32) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return decoded


def make_backend(name: str, **kwargs) -> LLMBackend:
    name = (name or "").lower()
    if name == "qwen-local" or name == "qwen":
        return QwenLocalBackend(**kwargs)
    if name == "openai":
        return OpenAIBackend(**kwargs)
    if name in ("none", "off", "dummy", ""):
        return DummyBackend()
    raise ValueError(f"unknown LLM backend: {name}")


# ---------------------------------------------------------------------------
# 후처리: LLM 답 정리
# ---------------------------------------------------------------------------
_QUOTE_RE = re.compile(r"^[\"'\(\[\{]+|[\"'\)\]\}]+$")


def _clean_llm_answer(s: str, is_yesno: bool) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.split("\n")[0].strip()
    s = _QUOTE_RE.sub("", s).strip()
    if is_yesno:
        low = s.lower()
        if "yes" in low and "no" not in low:
            return "yes"
        if "no" in low and "yes" not in low:
            return "no"
        if low.startswith("y"):
            return "yes"
        if low.startswith("n"):
            return "no"
        return "yes"
    return s


# ---------------------------------------------------------------------------
# Public: LLM-augmented answer
# ---------------------------------------------------------------------------
@dataclass
class LLMQAResult:
    tasi_answer: str
    llm_answer: str
    final_answer: str
    em: float
    f1: float
    is_correct: bool


def _llm_one_shot(
    sample: GraphSample,
    qa_res: QAResult,
    backend: LLMBackend,
    *,
    prompt_mode: str,
    top_k_evidence: int,
    max_new_tokens: int,
) -> str:
    """주어진 prompt_mode 로 LLM 한 번 호출 → cleaned answer."""
    is_yesno = qa_res.is_yesno
    system, user = build_prompt(
        sample, qa_res.slot_fillings, qa_res.predicted_answer, is_yesno,
        top_k_evidence=top_k_evidence, prompt_mode=prompt_mode,
    )
    if isinstance(backend, DummyBackend):
        return qa_res.predicted_answer
    try:
        raw = backend.generate(system, user, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.warning(f"[llm-qa] backend failed ({prompt_mode}): {exc}")
        raw = ""
    return _clean_llm_answer(raw, is_yesno) or qa_res.predicted_answer


def answer_question_with_llm(
    sample: GraphSample,
    encoder: SentenceEncoder,
    backend: LLMBackend,
    *,
    top_k_per_slot: int = 8,
    min_align: float = 0.32,
    top_k_evidence: int = 6,
    max_new_tokens: int = 32,
    prompt_mode: str = "extract",
) -> Tuple[QAResult, LLMQAResult]:
    """TASI 답 + LLM 답 모두 산출."""
    qa_res = answer_question(sample, encoder,
                             top_k_per_slot=top_k_per_slot, min_align=min_align)
    is_yesno = qa_res.is_yesno
    system, user = build_prompt(sample, qa_res.slot_fillings, qa_res.predicted_answer,
                                is_yesno, top_k_evidence=top_k_evidence,
                                prompt_mode=prompt_mode)
    raw_llm = ""
    if isinstance(backend, DummyBackend):
        raw_llm = qa_res.predicted_answer
    else:
        try:
            raw_llm = backend.generate(system, user, max_new_tokens=max_new_tokens)
        except Exception as exc:
            logger.warning(f"[llm-qa] backend failed: {exc}")
            raw_llm = ""

    llm_ans = _clean_llm_answer(raw_llm, is_yesno) or qa_res.predicted_answer
    em, f1 = score_answer(llm_ans, sample.answer, sample.answer_aliases)

    return qa_res, LLMQAResult(
        tasi_answer=qa_res.predicted_answer,
        llm_answer=llm_ans,
        final_answer=llm_ans,
        em=em,
        f1=f1,
        is_correct=bool(em >= 1.0),
    )


# ---------------------------------------------------------------------------
# Verifier-augmented QA: ensemble + TASI verification + abstain
# ---------------------------------------------------------------------------
@dataclass
class VerifiedQAResult:
    extract_answer: str
    reason_answer: str
    tasi_answer: str
    final_answer: str
    em: float
    f1: float
    is_correct: bool
    verify_score: float
    verify_grounding: float
    verify_chain: float
    verify_type: float
    verify_source: str
    abstained: bool
    candidate_scores: List[Dict[str, object]]


def answer_question_with_verifier(
    sample: GraphSample,
    encoder: SentenceEncoder,
    backend: LLMBackend,
    *,
    top_k_per_slot: int = 8,
    min_align: float = 0.32,
    top_k_evidence: int = 6,
    max_new_tokens: int = 32,
    use_extract: bool = True,
    use_reason: bool = True,
    use_tasi: bool = True,
    verifier_weights: Optional[Dict[str, float]] = None,
    abstain_threshold: float = 0.0,
) -> Tuple[QAResult, VerifiedQAResult]:
    """후보 ensemble + TASI verifier 로 최종 답을 결정.

    파이프라인:
      1) TASI 가 슬롯/evidence 만들고 _decide_open 답 (a_t)
      2) LLM(extract) 호출 → a_e   (use_extract=True 일 때만)
      3) LLM(reason)  호출 → a_r   (use_reason=True 일 때만)
      4) verifier 가 후보 {a_t, a_e, a_r} 를 grounding/chain/type 으로 채점
      5) 최고 점수 답 선택. score < abstain_threshold 이면 abstain → a_e (또는 a_t)
    """
    from graphqa.verifier import CandidatePool, pick_best  # local import to avoid cycle

    qa_res = answer_question(sample, encoder,
                             top_k_per_slot=top_k_per_slot, min_align=min_align)
    is_yesno = qa_res.is_yesno
    tasi_ans = qa_res.predicted_answer

    a_extract = ""
    if use_extract and not isinstance(backend, DummyBackend):
        a_extract = _llm_one_shot(sample, qa_res, backend,
                                  prompt_mode="extract",
                                  top_k_evidence=top_k_evidence,
                                  max_new_tokens=max_new_tokens)
    a_reason = ""
    if use_reason and not isinstance(backend, DummyBackend):
        a_reason = _llm_one_shot(sample, qa_res, backend,
                                 prompt_mode="reason",
                                 top_k_evidence=top_k_evidence,
                                 max_new_tokens=max_new_tokens)

    pool = CandidatePool()
    if use_extract and a_extract:
        pool.add("llm_extract", a_extract)
    if use_reason and a_reason:
        pool.add("llm_reason", a_reason)
    if use_tasi and tasi_ans:
        pool.add("tasi", tasi_ans)

    abstain_fallback = a_extract or a_reason or tasi_ans

    best, all_results, abstained = pick_best(
        pool, sample, qa_res.slot_fillings, encoder,
        weights=verifier_weights,
        abstain_threshold=abstain_threshold,
        abstain_fallback=abstain_fallback,
        is_yesno=is_yesno,
    )
    final_ans = best.answer or abstain_fallback or ""
    em, f1 = score_answer(final_ans, sample.answer, sample.answer_aliases)
    cand_dump = [r.to_dict() for r in all_results]

    return qa_res, VerifiedQAResult(
        extract_answer=a_extract,
        reason_answer=a_reason,
        tasi_answer=tasi_ans,
        final_answer=final_ans,
        em=em,
        f1=f1,
        is_correct=bool(em >= 1.0),
        verify_score=float(best.score),
        verify_grounding=float(best.grounding),
        verify_chain=float(best.chain),
        verify_type=float(best.type_match),
        verify_source=best.source,
        abstained=bool(abstained),
        candidate_scores=cand_dump,
    )


# =============================================================================
# TASI-gated LLM QA (사용자 제안 신 파이프라인)
#   1) TASI 가 슬롯 별 top-K 후보를 만든다.
#   2) total_sum (5-score 가중합) 이 너무 낮으면 LLM 호출 없이 abstain.
#   3) LLM 에게 (a) 슬롯을 채우고 (b) 답을 내라 — 한 번의 호출.
#   4) LLM 이 채운 chain 의 새 alignment 가 향상 안 되면 abstain.
# =============================================================================
import json as _json  # local alias to avoid shadowing
import numpy as _np


@dataclass
class TasiGatedQAResult:
    final_answer: str
    em: float
    f1: float
    is_correct: bool
    abstained: bool
    abstain_reason: str
    pre_gate_score: float          # total_sum (확신 prior)
    post_gate_uplift: float        # chain alignment uplift after fill
    llm_filled_slots: Dict[str, str]
    llm_raw_response: str
    candidate_pool: Dict[str, List[Tuple[str, float]]]
    is_yesno: bool


def _format_topk_candidates(
    pool: Dict[str, List[Tuple[str, float, float]]],
) -> List[str]:
    if not pool:
        return ["  (none)"]
    lines: List[str] = []
    for slot in sorted(pool.keys()):
        cs = pool[slot]
        if not cs:
            continue
        cand_str = " | ".join(f"{v} ({sc:.2f})" for v, sc, _ in cs)
        lines.append(f"  {slot} : {cand_str}")
    return lines


def _format_query_chain(sample: GraphSample) -> List[str]:
    out: List[str] = []
    for t in sample.Q:
        h = t.head if not t.head_unknown else f"<{_slot_id(t.head) or t.head}>"
        tl = t.tail if not t.tail_unknown else f"<{_slot_id(t.tail) or t.tail}>"
        out.append(f"  ({h}, {t.relation}, {tl})")
    return out


def _alignment_signal_lines(sample_alignment) -> List[str]:
    """SampleAlignment 객체를 prompt 친화 라인으로 변환."""
    from graphqa.alignment import alignment_quality_label
    if sample_alignment is None:
        return []
    qd = sample_alignment.align_QD_score
    qsr = sample_alignment.align_QSr_score
    qt = sample_alignment.align_QT_score

    def _row(name: str, score: float) -> str:
        if score != score:  # NaN
            return f"  {name:<28s}: n/a"
        return f"  {name:<28s}: cosine={score:.3f}  ({alignment_quality_label(score)})"

    return [
        _row("query ↔ documents", qd),
        _row("query ↔ search-rewrite", qsr),
        _row("query ↔ think-steps", qt),
    ]


def build_tasi_gated_prompt(
    sample: GraphSample,
    candidate_pool: Dict[str, List[Tuple[str, float, float]]],
    evidence_lines: List[str],
    is_yesno: bool,
    *,
    inject_alignment_signal: bool = False,
    sample_alignment=None,
) -> Tuple[str, str]:
    """1-call prompt: slot filling + 최종 답 (JSON 응답).

    inject_alignment_signal=True 면 (Q,D)/(Q,Sr)/(Q,T) 의 sentence-cosine
    alignment 점수를 'Alignment signal' 섹션으로 prompt 에 추가하여 LLM 이
    저신뢰 sample 을 더 신중히 다루도록 유도.
    """
    chain_lines = _format_query_chain(sample)
    cand_lines = _format_topk_candidates(candidate_pool)

    type_lines: List[str] = []
    for t in sample.Q_def or []:
        if t.head_unknown and t.tail and not is_unknown(t.tail):
            sid = _slot_id(t.head) or t.head
            type_lines.append(f"  {sid} is a {t.tail}")

    answer_format = (
        "'yes' or 'no'" if is_yesno else "a short noun phrase (the entity itself)"
    )

    system_base = (
        "You are a multi-hop QA reasoner. You are given a question, a reasoning "
        "chain in (head, relation, tail) form with UNKNOWN slots, and for each "
        "UNKNOWN slot a list of candidate entities extracted from the supporting "
        "documents (with TASI alignment scores). Use the candidates as strong "
        "hints; if none clearly fits, you may pick from the supporting evidence "
        "triples. Then use the filled chain to answer the question.\n"
        "Output ONLY a single line of strict JSON, no commentary, no markdown."
    )
    if inject_alignment_signal:
        system_base += (
            "\nYou will also see 'Alignment signal' — sentence-embedding cosine "
            "scores between the query triples and the (documents / search-rewrite "
            "/ think-steps). 'low' means the supporting context may be insufficient; "
            "'high' means strong support. If multiple signals are 'low' and the "
            "evidence does not clearly determine the answer, prefer answering "
            '"unknown" over guessing.'
        )

    user_parts: List[str] = [
        f"Question: {sample.question}",
        "",
        "Reasoning chain (UNKNOWN slots in <…>):",
        *chain_lines,
    ]
    if type_lines:
        user_parts += ["", "Slot type hints:", *type_lines]
    if inject_alignment_signal:
        user_parts += ["", "Alignment signal (sentence-embedding cosine):",
                       *_alignment_signal_lines(sample_alignment)]
    user_parts += [
        "",
        "Top candidate entities per slot (from TASI):",
        *cand_lines,
        "",
        "Top supporting evidence triples (head, relation, tail):",
        *(evidence_lines or ["  (none)"]),
        "",
        "Instructions:",
        "  1. For each UNKNOWN slot, choose the most appropriate filling.",
        "     Prefer the listed candidates; pick from evidence only if none fits.",
        "  2. Using the filled chain, give the final answer to the question.",
        f"     The answer must be {answer_format}.",
    ]
    if inject_alignment_signal:
        user_parts += [
            '  3. If alignment signals indicate low support and the evidence',
            '     is insufficient, output "unknown" as the answer.',
        ]
    user_parts += [
        "",
        'Respond with EXACTLY this JSON schema on ONE line:',
        '{"slots": {"ENT1": "...", "ENT2": "..."}, "answer": "..."}',
    ]
    return system_base, "\n".join(user_parts)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_tasi_gated_response(raw: str) -> Dict[str, object]:
    """JSON 부분만 안전하게 추출."""
    if not raw:
        return {}
    s = raw.strip()
    # 코드펜스 제거
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    m = _JSON_BLOCK_RE.search(s)
    if not m:
        return {}
    block = m.group(0)
    try:
        return _json.loads(block)
    except Exception:
        # 마지막 } 까지 잘라서 한 번 더 시도
        try:
            last = block.rfind("}")
            if last > 0:
                return _json.loads(block[:last + 1])
        except Exception:
            pass
    return {}


def _chain_alignment(
    Q_filled: Sequence[Triple],
    D: Sequence[Triple],
    encoder: SentenceEncoder,
) -> float:
    if not Q_filled or not D:
        return 0.0
    try:
        from graphqa.tasi.align import pairwise_alignment_matrix
        mat = pairwise_alignment_matrix(Q_filled, D, encoder)
    except Exception:
        return 0.0
    if mat.size == 0:
        return 0.0
    return float(_np.clip(mat.max(axis=1).mean(), 0.0, 1.0))


def _apply_filled_slots(
    sample: GraphSample,
    slot_to_value: Dict[str, str],
) -> List[Triple]:
    out: List[Triple] = []
    for t in sample.Q:
        h, tl = t.head, t.tail
        if t.head_unknown:
            sid = _slot_id(t.head)
            if sid and sid in slot_to_value and slot_to_value[sid]:
                h = slot_to_value[sid]
        if t.tail_unknown:
            sid = _slot_id(t.tail)
            if sid and sid in slot_to_value and slot_to_value[sid]:
                tl = slot_to_value[sid]
        out.append(Triple(head=h, relation=t.relation, tail=tl, context=t.context))
    return out


def answer_question_tasi_gated(
    sample: GraphSample,
    encoder: SentenceEncoder,
    backend: LLMBackend,
    *,
    pipeline=None,                   # TASIPipeline (있으면 total_sum 즉시 사용)
    k_per_slot: int = 5,
    top_k_evidence: int = 6,
    max_new_tokens: int = 96,
    pre_gate_threshold: float = 0.0,   # total_sum (혹은 prior) 의 abstain 임계
    post_gate_uplift_min: float = -0.05,  # alignment uplift 이 이보다 작으면 abstain
    enable_pre_gate: bool = True,
    enable_post_gate: bool = True,
    inject_alignment_signal: bool = False,
    sample_alignment=None,
) -> Tuple[QAResult, TasiGatedQAResult]:
    """TASI 간접 사용 + LLM 답변 통합 파이프라인.

    1. TASI 가 슬롯별 top-K 후보 + 5 score 산출
    2. (옵션) pre-gate: prior 가 너무 낮으면 abstain
    3. LLM 한 번 호출로 슬롯 채우기 + 최종 답을 JSON 으로 받음
    4. (옵션) post-gate: 채워진 chain 의 D-alignment uplift 가 음수면 abstain
    5. fallback: 어떤 단계에서든 실패하면 TASI-only 답으로 회수
    """
    from graphqa.qa import topk_slot_candidates  # local import to avoid cycle

    qa_res = answer_question(sample, encoder)
    is_yesno = qa_res.is_yesno

    # ---- pre-gate: TASI total_sum (또는 fallback prior) ----
    pre_score = float("nan")
    if pipeline is not None:
        try:
            scores = pipeline.score_sample(sample)
            pre_score = float(scores.total_sum)
        except Exception:
            pre_score = float("nan")
    if pre_score != pre_score:
        # fallback: 기본 fillings 의 평균 score 정규화
        if qa_res.slot_fillings:
            pre_score = float(_np.mean([
                min(1.0, c.score / 5.0) for c in qa_res.slot_fillings.values()
            ]))
        else:
            pre_score = 0.0

    if enable_pre_gate and pre_score < pre_gate_threshold:
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        return qa_res, TasiGatedQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=True, abstain_reason="pre_gate",
            pre_gate_score=pre_score,
            post_gate_uplift=float("nan"),
            llm_filled_slots={},
            llm_raw_response="",
            candidate_pool={},
            is_yesno=is_yesno,
        )

    # ---- TASI 가 만든 top-K 후보 ----
    pool_full = topk_slot_candidates(sample, encoder, k=k_per_slot)
    pool_short: Dict[str, List[Tuple[str, float]]] = {
        s: [(v, sc) for v, sc, _ in cs] for s, cs in pool_full.items()
    }
    evidence_lines = _evidence_for_slots(qa_res.slot_fillings, sample.D,
                                         top_k=top_k_evidence)

    # ---- LLM 호출 (1번) ----
    if isinstance(backend, DummyBackend):
        # 디버그: TASI 답 그대로
        em, f1 = score_answer(qa_res.predicted_answer, sample.answer,
                              sample.answer_aliases)
        return qa_res, TasiGatedQAResult(
            final_answer=qa_res.predicted_answer,
            em=em, f1=f1, is_correct=bool(em >= 1.0),
            abstained=False, abstain_reason="",
            pre_gate_score=pre_score,
            post_gate_uplift=0.0,
            llm_filled_slots={s: c.value for s, c in qa_res.slot_fillings.items()},
            llm_raw_response="(dummy)",
            candidate_pool=pool_short,
            is_yesno=is_yesno,
        )

    if inject_alignment_signal and sample_alignment is None:
        try:
            from graphqa.alignment import compute_sample_alignment
            sample_alignment = compute_sample_alignment(sample, encoder)
        except Exception as exc:
            logger.warning(f"[tasi-gated] alignment compute failed: {exc}")
            sample_alignment = None

    system, user = build_tasi_gated_prompt(
        sample, pool_full, evidence_lines, is_yesno,
        inject_alignment_signal=inject_alignment_signal,
        sample_alignment=sample_alignment,
    )
    raw = ""
    try:
        raw = backend.generate(system, user, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.warning(f"[tasi-gated] backend failed: {exc}")
    parsed = _parse_tasi_gated_response(raw)

    llm_slots: Dict[str, str] = {}
    if isinstance(parsed.get("slots"), dict):
        for k, v in parsed["slots"].items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            sid = _slot_id(k) or k.upper()
            llm_slots[sid] = v.strip()
    raw_answer = ""
    if isinstance(parsed.get("answer"), str):
        raw_answer = parsed["answer"]
    cleaned = _clean_llm_answer(raw_answer, is_yesno)

    # ---- post-gate: 채워진 chain 이 D 와 더 잘 정렬되는지 ----
    pre_align = _chain_alignment(sample.Q, sample.D, encoder)
    Q_filled = _apply_filled_slots(sample, llm_slots) if llm_slots else sample.Q
    post_align = _chain_alignment(Q_filled, sample.D, encoder)
    uplift = float(post_align - pre_align)

    if enable_post_gate and uplift < post_gate_uplift_min:
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        return qa_res, TasiGatedQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=True, abstain_reason="post_gate",
            pre_gate_score=pre_score,
            post_gate_uplift=uplift,
            llm_filled_slots=llm_slots,
            llm_raw_response=raw,
            candidate_pool=pool_short,
            is_yesno=is_yesno,
        )

    # parsing 실패 시 TASI 답으로 fallback
    if not cleaned:
        cleaned = qa_res.predicted_answer
    em, f1 = score_answer(cleaned, sample.answer, sample.answer_aliases)
    return qa_res, TasiGatedQAResult(
        final_answer=cleaned,
        em=em, f1=f1, is_correct=bool(em >= 1.0),
        abstained=False, abstain_reason="",
        pre_gate_score=pre_score,
        post_gate_uplift=uplift,
        llm_filled_slots=llm_slots,
        llm_raw_response=raw,
        candidate_pool=pool_short,
        is_yesno=is_yesno,
    )


# ===========================================================================
# Evidence-only mode (TASI 미사용)
#   LLM 이 보는 것 :
#     - question
#     - reasoning chain Q (UNKNOWN 슬롯 표기)
#     - (선택) slot type hints — sample.Q_def 에서 추출
#     - (선택) Alignment signal (Q,D)/(Q,Sr)/(Q,T)  ← sentence-cosine 기반만
#     - top-K supporting evidence triples from D       ← cosine top-K (TASI X)
#   LLM 이 하는 일 :
#     - UNKNOWN 슬롯 infill (TASI candidate 미공개, evidence 만 보고 채움)
#     - 채워진 chain + evidence 로 최종 답 생성
#   abstain : 없음 (선택적으로 cosine 평균이 너무 낮으면 abstain 가능)
# ===========================================================================


@dataclass
class EvidenceQAResult:
    final_answer: str
    em: float
    f1: float
    is_correct: bool
    abstained: bool
    abstain_reason: str
    cosine_min_pair: float        # min(QD, QSr, QT)  (사후 abstain 결정용)
    llm_filled_slots: Dict[str, str]
    llm_raw_response: str
    is_yesno: bool


def _evidence_by_cosine(
    sample: GraphSample,
    encoder: SentenceEncoder,
    top_k: int = 12,
) -> List[str]:
    """D 의 triple 을 (question + Q chain) 과의 sentence cosine 으로 정렬해서 top-K.

    TASI 의 free-matching / PPR / weighted alignment 을 일절 사용하지 않음.
    """
    if not sample.D:
        return []
    import numpy as _np
    q_texts: List[str] = [sample.question]
    for t in sample.Q:
        q_texts.append(f"{t.head} {t.relation} {t.tail}")
    d_texts: List[str] = [f"{t.head} {t.relation} {t.tail}" for t in sample.D]
    q_emb = encoder.encode(q_texts)
    d_emb = encoder.encode(d_texts)
    q_emb = q_emb / (_np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    d_emb = d_emb / (_np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-12)
    sim = d_emb @ q_emb.T              # (|D|, |Q|+1)
    score_per_d = sim.max(axis=1)      # 각 d 에 대해 question/Q 중 best
    order = _np.argsort(-score_per_d)[:top_k]
    out: List[str] = []
    for j in order:
        t = sample.D[int(j)]
        line = f"  ({t.head}, {t.relation}, {t.tail})"
        if getattr(t, "context", None):
            ctx = t.context.strip()
            if ctx:
                line += f"  [ctx: {ctx[:140]}]"
        out.append(line)
    return out


def build_evidence_prompt(
    sample: GraphSample,
    evidence_lines: List[str],
    is_yesno: bool,
    *,
    inject_alignment_signal: bool = False,
    sample_alignment=None,
) -> Tuple[str, str]:
    """Evidence-only prompt — TASI candidate / score 미사용.

    LLM 이 받는 것: question, chain (UNKNOWN slots), type hints, evidence.
    옵션: alignment signal (sentence-embedding cosine만).
    """
    chain_lines = _format_query_chain(sample)

    type_lines: List[str] = []
    for t in sample.Q_def or []:
        if t.head_unknown and t.tail and not is_unknown(t.tail):
            sid = _slot_id(t.head) or t.head
            type_lines.append(f"  {sid} is a {t.tail}")

    answer_format = (
        "'yes' or 'no'" if is_yesno
        else "a short noun phrase (the entity itself)"
    )

    system_base = (
        "You are a multi-hop QA reasoner. You are given a question, a reasoning "
        "chain in (head, relation, tail) form with UNKNOWN slots in <…>, and a "
        "list of supporting evidence triples extracted from documents. "
        "Use ONLY the evidence to fill the UNKNOWN slots, and then answer the "
        "question using the filled chain.\n"
        "Output ONLY a single line of strict JSON, no commentary, no markdown."
    )
    if inject_alignment_signal:
        system_base += (
            "\nYou will also see 'Alignment signal' — sentence-embedding cosine "
            "scores between the query triples and (documents / search-rewrite / "
            "think-steps). 'low' means the supporting context is likely "
            "insufficient; 'high' means strong support. If multiple signals are "
            "'low' AND the evidence does not clearly determine the answer, "
            'prefer answering "unknown" over guessing.'
        )

    user_parts: List[str] = [
        f"Question: {sample.question}",
        "",
        "Reasoning chain (UNKNOWN slots in <…>):",
        *chain_lines,
    ]
    if type_lines:
        user_parts += ["", "Slot type hints:", *type_lines]
    if inject_alignment_signal:
        user_parts += [
            "",
            "Alignment signal (sentence-embedding cosine):",
            *_alignment_signal_lines(sample_alignment),
        ]
    user_parts += [
        "",
        "Supporting evidence triples (head, relation, tail):",
        *(evidence_lines or ["  (none)"]),
        "",
        "Instructions:",
        "  1. Read the evidence carefully. For each UNKNOWN slot, infer the "
        "best filling using ONLY the evidence triples (do not guess outside).",
        "  2. Using the filled chain, give the final answer to the question.",
        f"     The answer must be {answer_format}.",
    ]
    if inject_alignment_signal:
        user_parts += [
            '  3. If alignment signals indicate low support and the evidence',
            '     is insufficient, output "unknown" as the answer.',
        ]
    user_parts += [
        "",
        'Respond with EXACTLY this JSON schema on ONE line:',
        '{"slots": {"ENT1": "...", "ENT2": "..."}, "answer": "..."}',
    ]
    return system_base, "\n".join(user_parts)


def answer_question_evidence(
    sample: GraphSample,
    encoder: SentenceEncoder,
    backend: LLMBackend,
    *,
    top_k_evidence: int = 12,
    max_new_tokens: int = 96,
    inject_alignment_signal: bool = False,
    sample_alignment=None,
    cosine_abstain_threshold: float = -1.0,  # default: never abstain
) -> Tuple[QAResult, EvidenceQAResult]:
    """LLM 이 evidence + chain 만 보고 직접 infill + 답.

    TASI 는 일체 사용하지 않음. Slot filling 은 LLM 이 evidence 보고 직접 수행.
    abstain 은 (선택) sentence-cosine alignment 평균이 임계 미만일 때만.
    """
    qa_res = answer_question(sample, encoder)   # only kept for tasi_em/f1 컬럼 호환
    is_yesno = qa_res.is_yesno

    # alignment signal 계산 (필요시)
    if (inject_alignment_signal or cosine_abstain_threshold > -1.0) \
            and sample_alignment is None:
        try:
            from graphqa.alignment import compute_sample_alignment
            sample_alignment = compute_sample_alignment(sample, encoder)
        except Exception as exc:
            logger.warning(f"[evidence] alignment compute failed: {exc}")
            sample_alignment = None

    cosine_min_pair = float("nan")
    if sample_alignment is not None:
        vals = [v for v in (sample_alignment.align_QD_score,
                            sample_alignment.align_QSr_score,
                            sample_alignment.align_QT_score)
                if v == v]   # filter NaN
        if vals:
            cosine_min_pair = float(min(vals))

    # cosine-based abstain (옵션)
    if cosine_abstain_threshold > -1.0 and cosine_min_pair == cosine_min_pair \
            and cosine_min_pair < cosine_abstain_threshold:
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        return qa_res, EvidenceQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=True, abstain_reason="cosine_low",
            cosine_min_pair=cosine_min_pair,
            llm_filled_slots={},
            llm_raw_response="",
            is_yesno=is_yesno,
        )

    # evidence 추출 (TASI 미사용)
    evidence_lines = _evidence_by_cosine(sample, encoder, top_k=top_k_evidence)

    # dummy backend → "no" / "" 로 빈 답
    if isinstance(backend, DummyBackend):
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        return qa_res, EvidenceQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=False, abstain_reason="",
            cosine_min_pair=cosine_min_pair,
            llm_filled_slots={},
            llm_raw_response="(dummy)",
            is_yesno=is_yesno,
        )

    system, user = build_evidence_prompt(
        sample, evidence_lines, is_yesno,
        inject_alignment_signal=inject_alignment_signal,
        sample_alignment=sample_alignment,
    )

    raw = ""
    try:
        raw = backend.generate(system, user, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.warning(f"[evidence] backend failed: {exc}")
    parsed = _parse_tasi_gated_response(raw)  # 동일 JSON 파서 재활용

    llm_slots: Dict[str, str] = {}
    if isinstance(parsed.get("slots"), dict):
        for k, v in parsed["slots"].items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            sid = _slot_id(k) or k.upper()
            llm_slots[sid] = v.strip()

    raw_answer = ""
    if isinstance(parsed.get("answer"), str):
        raw_answer = parsed["answer"]
    cleaned = _clean_llm_answer(raw_answer, is_yesno)

    if not cleaned:
        cleaned = ""    # parsing 실패시 빈 답 (TASI fallback 안 함)

    em, f1 = score_answer(cleaned, sample.answer, sample.answer_aliases)
    return qa_res, EvidenceQAResult(
        final_answer=cleaned,
        em=em, f1=f1, is_correct=bool(em >= 1.0),
        abstained=False, abstain_reason="",
        cosine_min_pair=cosine_min_pair,
        llm_filled_slots=llm_slots,
        llm_raw_response=raw,
        is_yesno=is_yesno,
    )


# ---------------------------------------------------------------------------
# Iterative slot filling + final QA (실험 base / align+abstain 비교군)
#   - UNKNOWN slot id 목록을 **랜덤 순서**로 하나씩 채움 (한 스텝당 LLM 1회).
#   - 각 스텝: evidence = D cosine top-K (question + 현재 chain).
#   - 비교군: 스텝마다 alignment signal 주입 + 중간 **abstain** 허용.
#   - 최종: 채워진 chain + question 으로 answer JSON 한 번.
#   (엔티티 단위 cosine / 역순 triple 매칭 등은 추후 — 현재는 evidence 와
#    alignment 신호에 기존 임베딩 유틸 사용.)
# ---------------------------------------------------------------------------


@dataclass
class IterativeSlotQAResult:
    final_answer: str
    em: float
    f1: float
    is_correct: bool
    abstained: bool
    abstain_reason: str
    filled_slots: Dict[str, str]
    n_slot_steps: int          # 실제 시도한 슬롯 스텝 수 (abstain 시 중단)
    n_llm_calls: int           # 슬롯 스텝 + 최종 답 1
    final_Q: List[Triple]      # 평가 후 align 재계산용 (채운 만큼)
    llm_raw_trace: str         # 마지막 응답 위주 truncate
    is_yesno: bool
    inject_step_alignment: bool
    # 랜덤 순서 슬롯 채우기 기준: 중간 abstain 시 몇 번째 시도에서 멈췄는지 (1-based), 없으면 0
    iter_planned_slot_count: int = 0
    iter_mid_abstain_step_1based: int = 0
    iter_mid_abstain_slot_id: str = ""


def _collect_unknown_slot_ids(q: Sequence[Triple]) -> List[str]:
    """Q 에 등장하는 UNKNOWN slot id 를 첫 등장 순서로 모은 뒤, 호출부에서 shuffle."""
    out: List[str] = []
    seen: set = set()
    for t in q:
        for tok in (t.head, t.tail):
            if is_unknown(tok):
                sid = _slot_id(tok) or str(tok).strip()
                if sid and sid not in seen:
                    seen.add(sid)
                    out.append(sid)
    return out


def _clone_sample_with_q(sample: GraphSample, q_new: List[Triple]) -> GraphSample:
    s = copy.copy(sample)
    s.Q = q_new
    return s


def _build_iter_step_prompt(
    sample: GraphSample,
    slot_to_value: Dict[str, str],
    target_sid: str,
    evidence_lines: List[str],
    *,
    inject_alignment: bool,
    sample_alignment,
) -> Tuple[str, str]:
    q_cur = _apply_filled_slots(sample, slot_to_value)
    chain_lines = _format_query_chain(_clone_sample_with_q(sample, q_cur))

    type_lines: List[str] = []
    for t in sample.Q_def or []:
        if t.head_unknown and t.tail and not is_unknown(t.tail):
            sid = _slot_id(t.head) or t.head
            if sid == target_sid:
                type_lines.append(f"  {sid} is a {t.tail}")

    sys_b = (
        "You are resolving ONE UNKNOWN placeholder in a multi-hop reasoning chain. "
        "Use ONLY the listed evidence triples from documents; do not invent facts.\n"
        "Output ONLY a single line of strict JSON, no commentary."
    )
    if inject_alignment:
        sys_b += (
            "\nYou see 'Alignment signal' (Q↔D / Q↔Sr / Q↔T cosine summaries). "
            "If support is clearly insufficient or alignment is too weak to pick a "
            'value from evidence, you may abstain with JSON: '
            '{"abstain": true, "reason": "..."} instead of filling the slot.'
        )

    user_parts: List[str] = [
        f"Question: {sample.question}",
        "",
        "Current reasoning chain (UNKNOWN still in <…> where not yet filled):",
        *chain_lines,
        "",
        f"This turn: fill exactly ONE slot: `{target_sid}`",
    ]
    if type_lines:
        user_parts += ["", "Slot type hints:", *type_lines]
    if inject_alignment and sample_alignment is not None:
        user_parts += [
            "",
            "Alignment signal (sentence-embedding cosine):",
            *_alignment_signal_lines(sample_alignment),
        ]
    user_parts += [
        "",
        "Supporting evidence triples:",
        *(evidence_lines or ["  (none)"]),
        "",
        "Respond with EXACTLY ONE of these JSON shapes on ONE line:",
        f'  {{"{target_sid}": "<short string from evidence>"}}',
    ]
    if inject_alignment:
        user_parts += ['  or  {"abstain": true, "reason": "..."}']
    user_parts.append("")
    return sys_b, "\n".join(user_parts)


def _build_iter_final_prompt(
    sample: GraphSample,
    q_filled: List[Triple],
    is_yesno: bool,
) -> Tuple[str, str]:
    s2 = _clone_sample_with_q(sample, q_filled)
    chain_lines = _format_query_chain(s2)
    remaining = _collect_unknown_slot_ids(q_filled)
    answer_format = (
        "'yes' or 'no'" if is_yesno
        else "a short noun phrase (the entity or phrase itself)"
    )
    if remaining:
        system = (
            "You answer a multi-hop question using the question and the reasoning "
            "chain below. Some UNKNOWN placeholders may remain; infer the best "
            "answer from the available chain. Output ONLY one line JSON: "
            "{\"answer\": \"...\"}."
        )
        chain_title = "Reasoning chain (some UNKNOWN slots may remain):"
    else:
        system = (
            "You answer a multi-hop question using the fully filled reasoning chain "
            "below. Output ONLY one line JSON: {\"answer\": \"...\"}."
        )
        chain_title = "Filled reasoning chain:"
    user = "\n".join([
        f"Question: {sample.question}",
        "",
        chain_title,
        *chain_lines,
        "",
        f"The answer must be {answer_format}.",
    ])
    return system, user


def _parse_step_slot_json(raw: str, target_sid: str) -> Tuple[bool, str, str]:
    """반환: (abstain, reason_or_empty, value_or_empty)."""
    p = _parse_tasi_gated_response(raw)
    if isinstance(p.get("abstain"), bool) and p["abstain"]:
        r = p.get("reason", "")
        return True, str(r) if r is not None else "", ""
    if target_sid in p and isinstance(p[target_sid], str):
        return False, "", p[target_sid].strip()
    # {"slots": {"ENT1": "..."}}
    slots = p.get("slots")
    if isinstance(slots, dict):
        for k, v in slots.items():
            sk = _slot_id(str(k)) or str(k).upper()
            if sk == target_sid and isinstance(v, str):
                return False, "", v.strip()
    return False, "", ""


def answer_question_iterative(
    sample: GraphSample,
    encoder: SentenceEncoder,
    backend: LLMBackend,
    *,
    inject_step_alignment: bool = False,
    allow_mid_abstain: bool = False,
    top_k_evidence: int = 12,
    max_new_tokens_step: int = 64,
    max_new_tokens_final: int = 96,
) -> Tuple[QAResult, IterativeSlotQAResult]:
    """랜덤 순서로 UNKNOWN slot 하나씩 채운 뒤 최종 답.

    allow_mid_abstain=True 일 때만 스텝 중 abstain 허용 (비교군).
    inject_step_alignment=True 이면 각 스텝 prompt 에 align 신호 포함.
    """
    qa_res = answer_question(sample, encoder)
    is_yesno = qa_res.is_yesno

    slot_ids = _collect_unknown_slot_ids(sample.Q)
    random.shuffle(slot_ids)
    n_planned_slots = len(slot_ids)

    slot_to_value: Dict[str, str] = {}
    raw_trace_parts: List[str] = []
    n_calls = 0

    if isinstance(backend, DummyBackend):
        # TASI 슬롯 채움 값으로 바로 채우고 최종은 빈 답 (스모크용)
        for sid in slot_ids:
            if sid in qa_res.slot_fillings:
                slot_to_value[sid] = qa_res.slot_fillings[sid].value
        q_fin = _apply_filled_slots(sample, slot_to_value)
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        return qa_res, IterativeSlotQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=False, abstain_reason="",
            filled_slots=dict(slot_to_value),
            n_slot_steps=len(slot_ids),
            n_llm_calls=0,
            final_Q=q_fin,
            llm_raw_trace="(dummy-iterative)",
            is_yesno=is_yesno,
            inject_step_alignment=inject_step_alignment,
            iter_planned_slot_count=n_planned_slots,
            iter_mid_abstain_step_1based=0,
            iter_mid_abstain_slot_id="",
        )

    abstained = False
    abstain_reason = ""
    steps_done = 0
    mid_abstain_step_1based = 0
    mid_abstain_slot_id = ""

    for step_1based, sid in enumerate(slot_ids, start=1):
        q_partial = _apply_filled_slots(sample, slot_to_value)
        cur = _clone_sample_with_q(sample, q_partial)
        evidence_lines = _evidence_by_cosine(cur, encoder, top_k=top_k_evidence)

        align_obj = None
        if inject_step_alignment or allow_mid_abstain:
            try:
                from graphqa.alignment import compute_sample_alignment
                align_obj = compute_sample_alignment(cur, encoder)
            except Exception as exc:
                logger.warning(f"[iterative] alignment failed: {exc}")

        use_align_in_prompt = bool(inject_step_alignment)
        sys_s, usr_s = _build_iter_step_prompt(
            sample, slot_to_value, sid, evidence_lines,
            inject_alignment=use_align_in_prompt,
            sample_alignment=align_obj,
        )
        raw_s = ""
        try:
            raw_s = backend.generate(sys_s, usr_s, max_new_tokens=max_new_tokens_step)
        except Exception as exc:
            logger.warning(f"[iterative] step LLM failed: {exc}")
        n_calls += 1
        raw_trace_parts.append(raw_s[:400])

        do_abstain, reason, val = _parse_step_slot_json(raw_s, sid)
        if allow_mid_abstain and do_abstain:
            abstained = True
            abstain_reason = reason or "mid_abstain"
            mid_abstain_step_1based = step_1based
            mid_abstain_slot_id = sid
            steps_done += 1
            break
        if val:
            slot_to_value[sid] = val
        steps_done += 1

    if abstained:
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        q_fin = _apply_filled_slots(sample, slot_to_value)
        return qa_res, IterativeSlotQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=True, abstain_reason=abstain_reason,
            filled_slots=dict(slot_to_value),
            n_slot_steps=steps_done,
            n_llm_calls=n_calls,
            final_Q=q_fin,
            llm_raw_trace=" | ".join(raw_trace_parts)[:800],
            is_yesno=is_yesno,
            inject_step_alignment=inject_step_alignment,
            iter_planned_slot_count=n_planned_slots,
            iter_mid_abstain_step_1based=mid_abstain_step_1based,
            iter_mid_abstain_slot_id=mid_abstain_slot_id,
        )

    q_fin = _apply_filled_slots(sample, slot_to_value)
    sys_f, usr_f = _build_iter_final_prompt(sample, q_fin, is_yesno)
    raw_f = ""
    try:
        raw_f = backend.generate(sys_f, usr_f, max_new_tokens=max_new_tokens_final)
    except Exception as exc:
        logger.warning(f"[iterative] final LLM failed: {exc}")
    n_calls += 1
    raw_trace_parts.append(raw_f[:400])

    parsed = _parse_tasi_gated_response(raw_f)
    ans = ""
    if isinstance(parsed.get("answer"), str):
        ans = parsed["answer"]
    cleaned = _clean_llm_answer(ans, is_yesno)
    em, f1 = score_answer(cleaned, sample.answer, sample.answer_aliases)
    return qa_res, IterativeSlotQAResult(
        final_answer=cleaned,
        em=em, f1=f1, is_correct=bool(em >= 1.0),
        abstained=False, abstain_reason="",
        filled_slots=dict(slot_to_value),
        n_slot_steps=steps_done,
        n_llm_calls=n_calls,
        final_Q=q_fin,
        llm_raw_trace=" | ".join(raw_trace_parts)[:800],
        is_yesno=is_yesno,
        inject_step_alignment=inject_step_alignment,
        iter_planned_slot_count=n_planned_slots,
        iter_mid_abstain_step_1based=0,
        iter_mid_abstain_slot_id="",
    )


# ---------------------------------------------------------------------------
# Deterministic triplet-driven slot filling + final LLM answer
#   - Pick the query triple with the fewest remaining UNKNOWN entities.
#   - Select the nearest document/think triple by whole-triple sentence cosine.
#   - Validate the selected pair by field-level cosine over known fields only,
#     trying both forward and head/tail-swapped document/think orientations.
#   - Fill from document when document validation passes. Think validation is
#     diagnostic only; answer gating is document-only for this mode.
# ---------------------------------------------------------------------------


@dataclass
class TripletFillQAResult:
    final_answer: str
    em: float
    f1: float
    is_correct: bool
    abstained: bool
    abstain_reason: str
    filled_slots: Dict[str, str]
    n_slot_steps: int
    n_llm_calls: int
    final_Q: List[Triple]
    llm_raw_trace: str
    is_yesno: bool
    doc_ok: bool
    think_ok: bool
    ok_pair: str
    min_doc_score: float
    min_think_score: float
    step_ok_pairs: str
    step_trace: str
    gate_failed: bool
    gate_fail_step_1based: int
    gate_fail_reason: str
    answer_on_failure: bool
    think_rescue_enabled: bool
    n_think_rescue: int
    remaining_slots: List[str]
    evidence_scope: str
    route_summary: Dict[str, Any]
    debug: Dict[str, Any]


def _unknown_entity_count(t: Triple) -> int:
    return int(t.head_unknown) + int(t.tail_unknown)


def _triple_sentence_for_match(t: Triple) -> str:
    """Whole-triple candidate retrieval text. UNKNOWN entities are omitted."""
    txt = t.to_text()
    return txt if txt else (t.relation or "")


def _triple_debug(
    t: Optional[Triple],
    *,
    graph: str,
    index: int,
) -> Dict[str, object]:
    if t is None:
        return {
            "graph": graph,
            "index": int(index),
            "exists": False,
        }
    unknown_fields: List[str] = []
    if t.head_unknown:
        unknown_fields.append("head")
    if t.tail_unknown:
        unknown_fields.append("tail")
    return {
        "graph": graph,
        "index": int(index),
        "exists": True,
        "head": t.head,
        "relation": t.relation,
        "tail": t.tail,
        "context": t.context or "",
        "raw": t.raw or "",
        "match_text": _triple_sentence_for_match(t),
        "unknown_fields": unknown_fields,
    }


def _select_next_query_triple(q_cur: Sequence[Triple]) -> Tuple[int, Optional[Triple]]:
    cands: List[Tuple[int, int, Triple]] = []
    for i, t in enumerate(q_cur):
        n_unk = _unknown_entity_count(t)
        if n_unk > 0:
            cands.append((n_unk, i, t))
    if not cands:
        return -1, None
    cands.sort(key=lambda x: (x[0], x[1]))
    _, idx, triple = cands[0]
    return idx, triple


def _best_whole_triple_match(
    q: Triple,
    pool: Sequence[Triple],
    encoder: SentenceEncoder,
) -> Tuple[int, float]:
    if not pool:
        return -1, float("nan")
    q_text = _triple_sentence_for_match(q)
    cand_texts = [_triple_sentence_for_match(t) for t in pool]
    try:
        embs = encoder.encode([q_text, *cand_texts])
        qv = embs[0]
        sims = embs[1:] @ qv
        j = int(_np.argmax(sims))
        return j, float(sims[j])
    except Exception:
        best_j, best_s = -1, float("-inf")
        for j, cand in enumerate(pool):
            try:
                s = float(encoder.cosine(q_text, _triple_sentence_for_match(cand)))
            except Exception:
                s = 0.0
            if s > best_s:
                best_j, best_s = j, s
        return best_j, best_s


def _topk_whole_triple_matches(
    q: Triple,
    pool: Sequence[Triple],
    encoder: SentenceEncoder,
    *,
    top_k: int,
) -> List[Tuple[int, float]]:
    if not pool:
        return []
    k = max(1, int(top_k))
    q_text = _triple_sentence_for_match(q)
    cand_texts = [_triple_sentence_for_match(t) for t in pool]
    try:
        embs = encoder.encode([q_text, *cand_texts])
        qv = embs[0]
        sims = embs[1:] @ qv
        order = _np.argsort(-sims)[:k]
        return [(int(j), float(sims[int(j)])) for j in order]
    except Exception:
        scored: List[Tuple[int, float]] = []
        for j, cand in enumerate(pool):
            try:
                s = float(encoder.cosine(q_text, _triple_sentence_for_match(cand)))
            except Exception:
                s = 0.0
            scored.append((j, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def _cosine_field(a: str, b: str, encoder: SentenceEncoder) -> float:
    if not a or not b or is_unknown(a) or is_unknown(b):
        return float("nan")
    aa = " ".join(a.strip().lower().split())
    bb = " ".join(b.strip().lower().split())
    if aa and aa == bb:
        return 1.0
    try:
        return float(encoder.cosine(a, b))
    except Exception:
        return 0.0


def _field_min_match(
    q: Triple,
    cand: Optional[Triple],
    encoder: SentenceEncoder,
) -> Tuple[float, bool, Dict[str, object]]:
    """Known-field min cosine, max over forward vs swapped head/tail."""
    if cand is None:
        return float("nan"), False, {
            "candidate_exists": False,
            "selected_orientation": "forward",
            "selected_fields": [],
            "orientations": {},
        }

    def score(reverse: bool) -> Tuple[float, Dict[str, object]]:
        ch = cand.tail if reverse else cand.head
        ct = cand.head if reverse else cand.tail
        fields: List[Dict[str, object]] = []
        vals: List[float] = []
        if not q.head_unknown:
            v = _cosine_field(q.head, ch, encoder)
            fields.append({
                "field": "head",
                "used": bool(v == v),
                "query_value": q.head,
                "candidate_value": ch,
                "cosine": float(v) if v == v else None,
            })
            if v == v:
                vals.append(float(v))
        else:
            fields.append({
                "field": "head",
                "used": False,
                "skip_reason": "query_field_unknown",
                "query_value": q.head,
                "candidate_value": ch,
                "cosine": None,
            })
        if q.relation and cand.relation:
            v = _cosine_field(q.relation, cand.relation, encoder)
            fields.append({
                "field": "relation",
                "used": bool(v == v),
                "query_value": q.relation,
                "candidate_value": cand.relation,
                "cosine": float(v) if v == v else None,
            })
            if v == v:
                vals.append(float(v))
        else:
            fields.append({
                "field": "relation",
                "used": False,
                "skip_reason": "missing_relation",
                "query_value": q.relation,
                "candidate_value": cand.relation,
                "cosine": None,
            })
        if not q.tail_unknown:
            v = _cosine_field(q.tail, ct, encoder)
            fields.append({
                "field": "tail",
                "used": bool(v == v),
                "query_value": q.tail,
                "candidate_value": ct,
                "cosine": float(v) if v == v else None,
            })
            if v == v:
                vals.append(float(v))
        else:
            fields.append({
                "field": "tail",
                "used": False,
                "skip_reason": "query_field_unknown",
                "query_value": q.tail,
                "candidate_value": ct,
                "cosine": None,
            })
        min_score = float(min(vals)) if vals else 0.0
        return min_score, {
            "orientation": "inverse_head_tail" if reverse else "forward",
            "score": min_score,
            "fields": fields,
        }

    fwd_s, fwd_detail = score(False)
    rev_s, rev_detail = score(True)
    if rev_s > fwd_s:
        return rev_s, True, {
            "candidate_exists": True,
            "selected_orientation": "inverse_head_tail",
            "selected_fields": rev_detail["fields"],
            "orientations": {
                "forward": fwd_detail,
                "inverse_head_tail": rev_detail,
            },
        }
    return fwd_s, False, {
        "candidate_exists": True,
        "selected_orientation": "forward",
        "selected_fields": fwd_detail["fields"],
        "orientations": {
            "forward": fwd_detail,
            "inverse_head_tail": rev_detail,
        },
    }


def _best_field_match_from_topk(
    q: Triple,
    pool: Sequence[Triple],
    encoder: SentenceEncoder,
    *,
    top_k: int,
    field_threshold: float,
    graph_label: str = "D",
) -> Tuple[int, float, float, bool, Dict[str, object], List[Dict[str, object]]]:
    """Select the top-whole-cosine candidate with the best field-level score."""
    top_matches = _topk_whole_triple_matches(q, pool, encoder, top_k=top_k)
    if not top_matches:
        score, inv, detail = _field_min_match(q, None, encoder)
        return -1, float("nan"), score, inv, detail, []

    best: Optional[Tuple[float, float, int, int, float, bool, Dict[str, object]]] = None
    summaries: List[Dict[str, object]] = []
    for rank, (idx, whole_score) in enumerate(top_matches, start=1):
        cand = pool[idx] if 0 <= idx < len(pool) else None
        field_score, inv, detail = _field_min_match(q, cand, encoder)
        ok = bool(field_score == field_score and field_score >= field_threshold)
        summaries.append({
            "rank_by_whole": int(rank),
            "index": int(idx),
            "triple": _triple_debug(cand, graph=graph_label, index=idx),
            "whole_cosine": float(whole_score) if whole_score == whole_score else None,
            "field_score": float(field_score) if field_score == field_score else None,
            "field_ok": bool(ok),
            "used_inverse_head_tail": bool(inv),
            "selected_orientation": detail.get("selected_orientation", ""),
            "selected_fields": detail.get("selected_fields", []),
        })
        # Prefer field agreement. Use whole-triple score only as a tie-breaker.
        key = (
            float(field_score) if field_score == field_score else float("-inf"),
            float(whole_score) if whole_score == whole_score else float("-inf"),
            -int(rank),
        )
        if best is None or key > best[:3]:
            best = (key[0], key[1], key[2], idx, whole_score, inv, detail)

    assert best is not None
    _, _, _, best_idx, best_whole, best_inv, best_detail = best
    best_field = float(best[0])
    return int(best_idx), float(best_whole), best_field, bool(best_inv), best_detail, summaries


def _fill_slots_from_aligned_triple(
    q: Triple,
    cand: Triple,
    *,
    used_inverse: bool,
) -> Dict[str, str]:
    filled: Dict[str, str] = {}
    if q.head_unknown:
        sid = _slot_id(q.head)
        val = cand.tail if used_inverse else cand.head
        if sid and val and not is_unknown(val):
            filled[sid] = val.strip()
    if q.tail_unknown:
        sid = _slot_id(q.tail)
        val = cand.head if used_inverse else cand.tail
        if sid and val and not is_unknown(val):
            filled[sid] = val.strip()
    return filled


_GENERIC_THINK_RESCUE_VALUES = {
    "a person", "an individual", "someone", "somebody", "a man", "a woman",
    "a country", "a nation", "a city", "a town", "a village", "a place",
    "a location", "a region", "a state", "a province", "an island",
    "a year", "a date", "a time", "a number", "an age",
    "a film", "a movie", "a song", "an album", "a book", "a novel",
    "a work", "a series", "a genre", "a language", "a currency",
    "an organization", "an organisation", "a company", "a team",
    "a school", "a university", "a party", "a political party",
    "unknown", "none", "n/a",
}


def _is_generic_think_rescue_value(value: str) -> bool:
    v = " ".join((value or "").strip().split())
    if not v or is_unknown(v):
        return True
    norm = v.lower().strip(" .,:;!?\"'")
    if norm in _GENERIC_THINK_RESCUE_VALUES:
        return True
    words = norm.split()
    if len(words) <= 4 and words and words[0] in {"a", "an"}:
        # Be conservative: article-led lowercase descriptions are usually type
        # hints from the think graph rather than concrete slot values.
        body = " ".join(words[1:])
        if body and v == v.lower():
            return True
    return False


def _filter_think_rescue_fills(
    fills: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    kept: Dict[str, str] = {}
    rejected: Dict[str, str] = {}
    for sid, val in fills.items():
        if _is_generic_think_rescue_value(val):
            rejected[sid] = val
        else:
            kept[sid] = val
    return kept, rejected


def _format_triplet_fill_trace(records: Sequence[Dict[str, object]]) -> str:
    chunks: List[str] = []
    for r in records:
        fills = r.get("fills", "")
        source = r.get("fill_source", "") or "-"
        chunks.append(
            "step={step} q={q_idx} doc={doc_ok}:{doc_score:.3f} "
            "think={think_ok}:{think_score:.3f} fill={fills} src={source}".format(
                step=r.get("step", 0),
                q_idx=r.get("q_idx", -1),
                doc_ok=int(bool(r.get("doc_ok", False))),
                doc_score=float(r.get("doc_score", float("nan"))),
                think_ok=int(bool(r.get("think_ok", False))),
                think_score=float(r.get("think_score", float("nan"))),
                fills=fills or "-",
                source=source,
            )
        )
    return " | ".join(chunks)


_TRIPLET_EVIDENCE_SCOPES = {
    "legacy_full",
    "combined_full",
    "combined_strict",
    "combined_prefix",
}


def _triplet_key(t: Triple) -> Tuple[str, str, str, str]:
    return (
        " ".join((t.head or "").split()),
        " ".join((t.relation or "").split()),
        " ".join((t.tail or "").split()),
        " ".join((t.context or "").split()),
    )


def _dedupe_pool_with_sources(
    triples_by_step: Sequence[Tuple[int, Sequence[Triple]]],
) -> Tuple[List[Triple], List[List[int]]]:
    pool: List[Triple] = []
    source_steps: List[List[int]] = []
    idx_by_key: Dict[Tuple[str, str, str, str], int] = {}
    for step_pos, triples in triples_by_step:
        for t in triples:
            key = _triplet_key(t)
            if key in idx_by_key:
                j = idx_by_key[key]
                if step_pos not in source_steps[j]:
                    source_steps[j].append(step_pos)
                continue
            idx_by_key[key] = len(pool)
            pool.append(t)
            source_steps.append([step_pos])
    return pool, source_steps


def _step_positions_for_scope(
    sample: GraphSample,
    evidence_scope: str,
    fill_step_1based: int,
) -> Optional[List[int]]:
    if evidence_scope == "legacy_full":
        return None
    n_steps = len(sample.step_evidence or [])
    if evidence_scope == "combined_full":
        return list(range(n_steps))
    cur = max(0, int(fill_step_1based) - 1)
    if evidence_scope == "combined_strict":
        return [cur] if cur < n_steps else []
    if evidence_scope == "combined_prefix":
        return list(range(min(cur + 1, n_steps)))
    return None


def _evidence_pool_for_step(
    sample: GraphSample,
    *,
    evidence_scope: str,
    fill_step_1based: int,
    kind: str,
) -> Tuple[List[Triple], List[List[int]], Dict[str, object]]:
    positions = _step_positions_for_scope(sample, evidence_scope, fill_step_1based)
    if positions is None:
        pool = list(sample.D if kind == "doc" else sample.T)
        source_steps = [[] for _ in pool]
        return pool, source_steps, {
            "scope": evidence_scope,
            "source_step_positions_0based": None,
            "n_source_steps_total": len(sample.step_evidence or []),
            "n_candidates": len(pool),
        }
    steps = list(sample.step_evidence or [])
    triples_by_step: List[Tuple[int, Sequence[Triple]]] = []
    for pos in positions:
        if 0 <= pos < len(steps):
            st = steps[pos]
            triples_by_step.append(
                (pos, st.doc_triples if kind == "doc" else st.think_triples)
            )
    pool, source_steps = _dedupe_pool_with_sources(triples_by_step)
    return pool, source_steps, {
        "scope": evidence_scope,
        "source_step_positions_0based": positions,
        "source_step_positions_1based": [p + 1 for p in positions],
        "n_source_steps_total": len(steps),
        "n_candidates": len(pool),
    }


def _best_match_in_pool_for_kind(
    q: Triple,
    pool: Sequence[Triple],
    encoder: SentenceEncoder,
    *,
    kind: str,
    top_k: int,
    field_threshold: float,
) -> Tuple[int, float, float, bool, Dict[str, object], List[Dict[str, object]]]:
    graph_label = "D" if kind == "doc" else "T"
    return _best_field_match_from_topk(
        q,
        pool,
        encoder,
        top_k=top_k,
        field_threshold=field_threshold,
        graph_label=graph_label,
    )


def _route_alignment_for_kind(
    q: Triple,
    sample: GraphSample,
    encoder: SentenceEncoder,
    *,
    fill_step_1based: int,
    kind: str,
    top_k: int,
    field_threshold: float,
) -> Dict[str, object]:
    steps = list(sample.step_evidence or [])
    fill_step_0based = max(0, int(fill_step_1based) - 1)
    per_step: List[Dict[str, object]] = []
    ok_steps: List[int] = []
    best_step = -1
    best_score = float("-inf")
    best_whole = float("-inf")

    for pos, st in enumerate(steps):
        pool = st.doc_triples if kind == "doc" else st.think_triples
        idx, whole, score, inv, _detail, _summary = _best_match_in_pool_for_kind(
            q,
            pool,
            encoder,
            kind=kind,
            top_k=top_k,
            field_threshold=field_threshold,
        )
        ok = bool(score == score and score >= field_threshold)
        if ok:
            ok_steps.append(pos)
        score_key = float(score) if score == score else float("-inf")
        whole_key = float(whole) if whole == whole else float("-inf")
        if (score_key, whole_key) > (best_score, best_whole):
            best_step = pos
            best_score = score_key
            best_whole = whole_key
        per_step.append({
            "step_0based": int(pos),
            "step_1based": int(pos + 1),
            "candidate_count": int(len(pool)),
            "best_index": int(idx),
            "best_whole_cosine": float(whole) if whole == whole else None,
            "best_field_score": float(score) if score == score else None,
            "ok": bool(ok),
            "used_inverse_head_tail": bool(inv),
        })

    strict_available = fill_step_0based in ok_steps
    prefix_available = any(p <= fill_step_0based for p in ok_steps)
    future_only = (not prefix_available) and any(p > fill_step_0based for p in ok_steps)
    unavailable = not ok_steps
    if strict_available:
        status = "strict_available"
    elif prefix_available:
        status = "prefix_available"
    elif future_only:
        status = "future_only"
    else:
        status = "unavailable"
    return {
        "kind": kind,
        "fill_step_1based": int(fill_step_1based),
        "fill_step_0based": int(fill_step_0based),
        "status": status,
        "strict_available": bool(strict_available),
        "prefix_available": bool(prefix_available),
        "future_only": bool(future_only),
        "unavailable": bool(unavailable),
        "ok_steps_0based": [int(p) for p in ok_steps],
        "ok_steps_1based": [int(p + 1) for p in ok_steps],
        "best_step_0based": int(best_step),
        "best_step_1based": int(best_step + 1) if best_step >= 0 else 0,
        "best_step_matches_fill_order": bool(best_step == fill_step_0based),
        "best_field_score": best_score if best_score != float("-inf") else None,
        "best_whole_cosine": best_whole if best_whole != float("-inf") else None,
        "per_step": per_step,
    }


def _route_summary_from_steps(step_debug: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "n_steps": int(len(step_debug)),
    }
    for kind in ("doc", "think"):
        counts = {
            "strict_available": 0,
            "prefix_available": 0,
            "future_only": 0,
            "unavailable": 0,
        }
        exact_flags: List[bool] = []
        for rec in step_debug:
            diag = ((rec.get("route_alignment") or {}) if isinstance(rec, dict) else {}).get(kind)
            if not isinstance(diag, dict):
                counts["unavailable"] += 1
                exact_flags.append(False)
                continue
            status = str(diag.get("status", "unavailable"))
            if status not in counts:
                status = "unavailable"
            counts[status] += 1
            exact_flags.append(bool(diag.get("best_step_matches_fill_order", False)))
        longest = 0
        for flag in exact_flags:
            if not flag:
                break
            longest += 1
        anywhere = len(step_debug) - counts["unavailable"]
        mismatch = (counts["future_only"] / anywhere) if anywhere else 0.0
        prefix_or_strict = counts["strict_available"] + counts["prefix_available"]
        summary[kind] = {
            **counts,
            "anywhere_available": int(anywhere),
            "prefix_or_strict_available": int(prefix_or_strict),
            "order_mismatch_rate": float(mismatch),
            "longest_exact_route_prefix": int(longest),
        }

    both_longest = 0
    for rec in step_debug:
        route = rec.get("route_alignment") if isinstance(rec, dict) else {}
        doc = route.get("doc") if isinstance(route, dict) else {}
        think = route.get("think") if isinstance(route, dict) else {}
        if (
            isinstance(doc, dict)
            and isinstance(think, dict)
            and bool(doc.get("best_step_matches_fill_order", False))
            and bool(think.get("best_step_matches_fill_order", False))
        ):
            both_longest += 1
            continue
        break
    summary["both_longest_exact_route_prefix"] = int(both_longest)
    return summary


def answer_question_triplet_fill(
    sample: GraphSample,
    encoder: SentenceEncoder,
    backend: LLMBackend,
    *,
    field_threshold: float = 0.50,
    max_steps: int = 16,
    answer_on_failure: bool = False,
    doc_top_k: int = 1,
    think_rescue: bool = False,
    evidence_scope: str = "legacy_full",
    max_new_tokens_final: int = 96,
) -> Tuple[QAResult, TripletFillQAResult]:
    """Fill query UNKNOWNs from aligned document triples, then ask LLM.

    Document validation controls whether slot filling can continue. When
    answer_on_failure=True, a document gate failure is still recorded as the
    legacy stop point, but the final LLM answer is attempted using the partial
    chain. Think validation is recorded only for the 0/0, 1/0, 0/1, 1/1
    diagnostic split. If doc_top_k > 1, document candidates are first narrowed
    by whole-triple cosine and then selected by the best field-level score. If
    think_rescue=True, a doc-failed/think-ok step may fill concrete UNKNOWN
    values from the aligned think triple and continue.
    """
    evidence_scope = (evidence_scope or "legacy_full").lower()
    if evidence_scope not in _TRIPLET_EVIDENCE_SCOPES:
        raise ValueError(
            f"unknown triplet evidence scope: {evidence_scope}. "
            f"expected one of {sorted(_TRIPLET_EVIDENCE_SCOPES)}"
        )

    qa_res = answer_question(sample, encoder)
    is_yesno = qa_res.is_yesno

    slot_to_value: Dict[str, str] = {}
    step_records: List[Dict[str, object]] = []
    step_debug: List[Dict[str, object]] = []
    gate_failed = False
    gate_fail_reason = ""
    gate_fail_step_1based = 0
    n_think_rescue = 0
    n_calls = 0

    planned_slots = _collect_unknown_slot_ids(sample.Q)
    if not planned_slots:
        q_fin = list(sample.Q)
    else:
        q_fin = _apply_filled_slots(sample, slot_to_value)

    for step_1based in range(1, max(1, int(max_steps)) + 1):
        q_cur = _apply_filled_slots(sample, slot_to_value)
        q_idx, q_next = _select_next_query_triple(q_cur)
        if q_next is None:
            q_fin = q_cur
            break

        d_pool, d_sources, d_pool_info = _evidence_pool_for_step(
            sample,
            evidence_scope=evidence_scope,
            fill_step_1based=step_1based,
            kind="doc",
        )
        t_pool, t_sources, t_pool_info = _evidence_pool_for_step(
            sample,
            evidence_scope=evidence_scope,
            fill_step_1based=step_1based,
            kind="think",
        )

        d_idx, d_whole, d_score, d_inv, d_detail, d_topk = _best_field_match_from_topk(
            q_next,
            d_pool,
            encoder,
            top_k=doc_top_k,
            field_threshold=field_threshold,
        )
        d_triple = d_pool[d_idx] if 0 <= d_idx < len(d_pool) else None
        d_ok = bool(d_score == d_score and d_score >= field_threshold)

        t_idx, t_whole = _best_whole_triple_match(q_next, t_pool, encoder)
        t_triple = t_pool[t_idx] if 0 <= t_idx < len(t_pool) else None
        t_score, t_inv, t_detail = _field_min_match(q_next, t_triple, encoder)
        t_ok = bool(t_score == t_score and t_score >= field_threshold)
        d_source_steps = d_sources[d_idx] if 0 <= d_idx < len(d_sources) else []
        t_source_steps = t_sources[t_idx] if 0 <= t_idx < len(t_sources) else []

        doc_route = _route_alignment_for_kind(
            q_next,
            sample,
            encoder,
            fill_step_1based=step_1based,
            kind="doc",
            top_k=doc_top_k,
            field_threshold=field_threshold,
        )
        think_route = _route_alignment_for_kind(
            q_next,
            sample,
            encoder,
            fill_step_1based=step_1based,
            kind="think",
            top_k=doc_top_k,
            field_threshold=field_threshold,
        )

        fills: Dict[str, str] = {}
        fill_source = ""
        think_rescue_raw_fills: Dict[str, str] = {}
        think_rescue_rejected: Dict[str, str] = {}
        think_rescue_skip_reason = ""
        think_rescue_applied = False
        if d_ok and d_triple is not None:
            fills = _fill_slots_from_aligned_triple(q_next, d_triple,
                                                    used_inverse=d_inv)
            fill_source = "document" if fills else ""
        elif think_rescue and t_ok and t_triple is not None:
            if _unknown_entity_count(q_next) != 1:
                think_rescue_skip_reason = "requires_exactly_one_unknown_entity"
            else:
                raw_fills = _fill_slots_from_aligned_triple(q_next, t_triple,
                                                            used_inverse=t_inv)
                think_rescue_raw_fills = dict(sorted(raw_fills.items()))
                fills, think_rescue_rejected = _filter_think_rescue_fills(raw_fills)
                if fills:
                    fill_source = "think_rescue"
                    think_rescue_applied = True
                    n_think_rescue += 1
        if fills:
            for sid, val in fills.items():
                slot_to_value[sid] = val

        stop_reason = ""
        if not d_ok and not think_rescue_applied:
            stop_reason = "doc_alignment_failed"
        elif not fills:
            stop_reason = "doc_alignment_no_fill"

        step_records.append({
            "step": step_1based,
            "q_idx": q_idx,
            "doc_idx": d_idx,
            "think_idx": t_idx,
            "doc_whole": d_whole,
            "think_whole": t_whole,
            "doc_score": d_score,
            "think_score": t_score,
            "doc_ok": d_ok,
            "think_ok": t_ok,
            "doc_inverse": d_inv,
            "think_inverse": t_inv,
            "fill_source": fill_source,
            "think_rescue_applied": think_rescue_applied,
            "fills": ",".join(f"{k}={v}" for k, v in sorted(fills.items())),
            "doc_route_status": doc_route.get("status", ""),
            "think_route_status": think_route.get("status", ""),
        })
        step_debug.append({
            "step_1based": int(step_1based),
            "selected_query": _triple_debug(q_next, graph="Q", index=q_idx),
            "selected_document": _triple_debug(d_triple, graph="D", index=d_idx),
            "selected_think": _triple_debug(t_triple, graph="T", index=t_idx),
            "evidence": {
                "scope": evidence_scope,
                "document_pool": {
                    **d_pool_info,
                    "selected_source_steps_0based": [int(x) for x in d_source_steps],
                    "selected_source_steps_1based": [int(x + 1) for x in d_source_steps],
                },
                "think_pool": {
                    **t_pool_info,
                    "selected_source_steps_0based": [int(x) for x in t_source_steps],
                    "selected_source_steps_1based": [int(x + 1) for x in t_source_steps],
                },
            },
            "whole_triple_match": {
                "document_cosine": float(d_whole) if d_whole == d_whole else None,
                "think_cosine": float(t_whole) if t_whole == t_whole else None,
            },
            "document_topk_field_select": {
                "top_k": int(max(1, int(doc_top_k))),
                "selected_doc_index": int(d_idx),
                "candidates": d_topk,
            },
            "field_validation": {
                "threshold": float(field_threshold),
                "document": {
                    "ok": bool(d_ok),
                    "score": float(d_score) if d_score == d_score else None,
                    "used_inverse_head_tail": bool(d_inv),
                    "detail": d_detail,
                },
                "think": {
                    "ok": bool(t_ok),
                    "score": float(t_score) if t_score == t_score else None,
                    "used_inverse_head_tail": bool(t_inv),
                    "detail": t_detail,
                },
            },
            "action": {
                "filled_from": fill_source,
                "fills": dict(sorted(fills.items())),
                "slot_values_after_step": dict(sorted(slot_to_value.items())),
                "would_stop_here": bool(stop_reason),
                "would_stop_without_think_rescue": bool(not d_ok),
                "legacy_stop_reason": stop_reason,
                "answer_on_failure": bool(answer_on_failure and stop_reason),
                "think_rescue_enabled": bool(think_rescue),
                "think_rescue_applied": bool(think_rescue_applied),
                "think_rescue_skip_reason": think_rescue_skip_reason,
                "think_rescue_raw_fills": think_rescue_raw_fills,
                "think_rescue_rejected_generic": think_rescue_rejected,
            },
            "route_alignment": {
                "doc": doc_route,
                "think": think_route,
            },
        })

        if stop_reason:
            gate_failed = True
            gate_fail_reason = stop_reason
            gate_fail_step_1based = step_1based
            q_fin = q_cur
            break
    else:
        gate_failed = True
        gate_fail_reason = "max_steps"
        gate_fail_step_1based = int(max(1, int(max_steps)))
        q_fin = _apply_filled_slots(sample, slot_to_value)

    q_fin = _apply_filled_slots(sample, slot_to_value)
    remaining = _collect_unknown_slot_ids(q_fin)
    if remaining and not gate_failed:
        gate_failed = True
        gate_fail_reason = "unfilled_slots"
        gate_fail_step_1based = len(step_records)

    doc_vals = [bool(r.get("doc_ok", False)) for r in step_records]
    think_vals = [bool(r.get("think_ok", False)) for r in step_records]
    doc_ok_all = bool(all(doc_vals)) if doc_vals else True
    think_ok_all = bool(all(think_vals)) if think_vals else True
    ok_pair = f"{int(doc_ok_all)}/{int(think_ok_all)}"
    step_ok_pairs = "|".join(
        f"{int(bool(r.get('doc_ok', False)))}/{int(bool(r.get('think_ok', False)))}"
        for r in step_records
    )
    min_doc = float(min([float(r.get("doc_score", 0.0)) for r in step_records],
                        default=float("nan")))
    min_think = float(min([float(r.get("think_score", 0.0)) for r in step_records],
                          default=float("nan")))
    step_trace = _format_triplet_fill_trace(step_records)
    route_summary = _route_summary_from_steps(step_debug)
    actual_abstained = bool(gate_failed and not answer_on_failure)

    debug_base: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "triplet_fill_doc_only",
        "evidence_scope": evidence_scope,
        "threshold": float(field_threshold),
        "max_steps": int(max_steps),
        "doc_top_k": int(max(1, int(doc_top_k))),
        "answer_on_failure": bool(answer_on_failure),
        "think_rescue": {
            "enabled": bool(think_rescue),
            "n_applied_steps": int(n_think_rescue),
        },
        "legacy_gate": {
            "failed": bool(gate_failed),
            "fail_step_1based": int(gate_fail_step_1based),
            "fail_reason": gate_fail_reason,
            "would_have_abstained": bool(gate_failed),
            "actual_abstained": bool(actual_abstained),
        },
        "slots": {
            "planned": list(planned_slots),
            "filled": dict(sorted(slot_to_value.items())),
            "remaining": list(remaining),
        },
        "initial_query_triples": [
            _triple_debug(t, graph="Q", index=i) for i, t in enumerate(sample.Q)
        ],
        "final_query_triples": [
            _triple_debug(t, graph="Q_final", index=i) for i, t in enumerate(q_fin)
        ],
        "steps": step_debug,
        "compact_step_trace": step_trace,
        "route_alignment": route_summary,
        "doc_think_ok": {
            "sample_pair": ok_pair,
            "step_pairs": step_ok_pairs,
            "doc_ok_all": bool(doc_ok_all),
            "think_ok_all": bool(think_ok_all),
            "min_doc_score": min_doc if min_doc == min_doc else None,
            "min_think_score": min_think if min_think == min_think else None,
        },
    }

    if actual_abstained:
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        debug_base["final"] = {
            "abstained": True,
            "predicted_answer": "",
            "llm_raw_trace": "",
            "em": float(em),
            "f1": float(f1),
            "is_correct": False,
        }
        return qa_res, TripletFillQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=True, abstain_reason=gate_fail_reason,
            filled_slots=dict(slot_to_value),
            n_slot_steps=len(step_records),
            n_llm_calls=n_calls,
            final_Q=q_fin,
            llm_raw_trace="",
            is_yesno=is_yesno,
            doc_ok=doc_ok_all,
            think_ok=think_ok_all,
            ok_pair=ok_pair,
            min_doc_score=min_doc,
            min_think_score=min_think,
            step_ok_pairs=step_ok_pairs,
            step_trace=step_trace,
            gate_failed=gate_failed,
            gate_fail_step_1based=gate_fail_step_1based,
            gate_fail_reason=gate_fail_reason,
            answer_on_failure=answer_on_failure,
            think_rescue_enabled=think_rescue,
            n_think_rescue=n_think_rescue,
            remaining_slots=list(remaining),
            evidence_scope=evidence_scope,
            route_summary=route_summary,
            debug=debug_base,
        )

    if isinstance(backend, DummyBackend):
        em, f1 = score_answer("", sample.answer, sample.answer_aliases)
        debug_base["final"] = {
            "abstained": False,
            "predicted_answer": "",
            "llm_raw_trace": "(dummy-triplet-fill)",
            "em": float(em),
            "f1": float(f1),
            "is_correct": False,
        }
        return qa_res, TripletFillQAResult(
            final_answer="",
            em=em, f1=f1, is_correct=False,
            abstained=False, abstain_reason=gate_fail_reason,
            filled_slots=dict(slot_to_value),
            n_slot_steps=len(step_records),
            n_llm_calls=0,
            final_Q=q_fin,
            llm_raw_trace="(dummy-triplet-fill)",
            is_yesno=is_yesno,
            doc_ok=doc_ok_all,
            think_ok=think_ok_all,
            ok_pair=ok_pair,
            min_doc_score=min_doc,
            min_think_score=min_think,
            step_ok_pairs=step_ok_pairs,
            step_trace=step_trace,
            gate_failed=gate_failed,
            gate_fail_step_1based=gate_fail_step_1based,
            gate_fail_reason=gate_fail_reason,
            answer_on_failure=answer_on_failure,
            think_rescue_enabled=think_rescue,
            n_think_rescue=n_think_rescue,
            remaining_slots=list(remaining),
            evidence_scope=evidence_scope,
            route_summary=route_summary,
            debug=debug_base,
        )

    sys_f, usr_f = _build_iter_final_prompt(sample, q_fin, is_yesno)
    raw_f = ""
    try:
        raw_f = backend.generate(sys_f, usr_f, max_new_tokens=max_new_tokens_final)
    except Exception as exc:
        logger.warning(f"[triplet-fill] final LLM failed: {exc}")
    n_calls += 1

    parsed = _parse_tasi_gated_response(raw_f)
    ans = parsed.get("answer", "") if isinstance(parsed, dict) else ""
    cleaned = _clean_llm_answer(str(ans), is_yesno) if ans else ""
    em, f1 = score_answer(cleaned, sample.answer, sample.answer_aliases)
    debug_base["final"] = {
        "abstained": False,
        "predicted_answer": cleaned,
        "llm_raw_trace": raw_f[:800],
        "em": float(em),
        "f1": float(f1),
        "is_correct": bool(em >= 1.0),
    }
    return qa_res, TripletFillQAResult(
        final_answer=cleaned,
        em=em, f1=f1, is_correct=bool(em >= 1.0),
        abstained=False, abstain_reason=gate_fail_reason,
        filled_slots=dict(slot_to_value),
        n_slot_steps=len(step_records),
        n_llm_calls=n_calls,
        final_Q=q_fin,
        llm_raw_trace=raw_f[:800],
        is_yesno=is_yesno,
        doc_ok=doc_ok_all,
        think_ok=think_ok_all,
        ok_pair=ok_pair,
        min_doc_score=min_doc,
        min_think_score=min_think,
        step_ok_pairs=step_ok_pairs,
        step_trace=step_trace,
        gate_failed=gate_failed,
        gate_fail_step_1based=gate_fail_step_1based,
        gate_fail_reason=gate_fail_reason,
        answer_on_failure=answer_on_failure,
        think_rescue_enabled=think_rescue,
        n_think_rescue=n_think_rescue,
        remaining_slots=list(remaining),
        evidence_scope=evidence_scope,
        route_summary=route_summary,
        debug=debug_base,
    )
