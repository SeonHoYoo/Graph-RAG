"""TASI-based answer verification.

LLM 또는 TASI 가 산출한 답 후보들을 *그래프* 기반으로 사후 검증한다.
3가지 신호를 합산:

  (G) grounding         : 답 entity 가 D 어딘가에 실제로 등장하는가
  (C) chain consistency : 답을 chain 의 (가장 약한) 슬롯에 끼워 넣었을 때
                          새 chain 이 D 와 더 잘 정렬되는가 (alignment uplift)
  (T) type match        : 답이 Q_def 의 (ENTk is a TYPE) 정의와 의미적으로 부합하는가

verify_answer(a, ctx) → VerificationResult(score, parts)
pick_best(candidates, ctx, abstain_threshold) → (best_answer, score, abstained)

이 모듈은 LLM 을 호출하지 않는다. 모든 신호는 sentence-transformer 임베딩 +
TASI alignment 로 계산되므로 sample 당 수십 ms.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from graphqa.data.schema import GraphSample, Triple, is_unknown
from graphqa.qa import (
    SlotCandidate,
    _norm_answer,
    _slot_id,
)
from graphqa.tasi.align import pairwise_alignment_matrix
from graphqa.tasi.embedding import SentenceEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 가벼운 유틸
# ---------------------------------------------------------------------------
_TOKEN_SPLIT_RE = re.compile(r"\s+")


def _tokens(s: str) -> List[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(_norm_answer(s)) if t]


def _token_set(s: str) -> set:
    return set(_tokens(s))


def _max_token_overlap(answer: str, text: str) -> float:
    """답 토큰이 text 안에 얼마나 들어있는지 (0~1)."""
    a = _token_set(answer)
    if not a:
        return 0.0
    t = _token_set(text)
    if not t:
        return 0.0
    return len(a & t) / max(1, len(a))


# ---------------------------------------------------------------------------
# (G) Grounding: 답이 D 에 실제로 나타나는가
# ---------------------------------------------------------------------------
def grounding_score(
    answer: str,
    D: Sequence[Triple],
    encoder: Optional[SentenceEncoder] = None,
    *,
    include_context: bool = True,
) -> float:
    """답 entity 가 D 의 head/tail/context 어딘가에 등장하는가 (0~1).

    1. token overlap 우선 (정확/부분 매칭).
    2. 매칭 없으면 임베딩 코사인 유사도 fallback.
    """
    if not answer or not D:
        return 0.0
    a_norm = _norm_answer(answer)
    if not a_norm:
        return 0.0
    a_toks = _token_set(answer)
    if not a_toks:
        return 0.0

    best = 0.0
    best_text = ""
    for t in D:
        nodes = [t.head, t.tail]
        if include_context and t.context:
            nodes.append(t.context)
        for txt in nodes:
            if not txt:
                continue
            if a_norm == _norm_answer(txt):
                return 1.0
            ov = _max_token_overlap(answer, txt)
            if ov > best:
                best = ov
                best_text = txt
    if best >= 0.5:
        return float(min(1.0, best))

    if encoder is None:
        return float(best)
    pool: List[str] = []
    for t in D:
        if t.head:
            pool.append(t.head)
        if t.tail:
            pool.append(t.tail)
    if not pool:
        return float(best)
    pool = list(dict.fromkeys(pool))[:64]
    try:
        a_emb = encoder.encode([answer])[0]
        p_emb = encoder.encode(pool)
        sims = p_emb @ a_emb / (
            np.linalg.norm(p_emb, axis=1) * (np.linalg.norm(a_emb) + 1e-12) + 1e-12
        )
        sim = float(np.max(sims)) if len(sims) else 0.0
        return float(max(best, max(0.0, (sim - 0.4) / 0.6)))
    except Exception:
        return float(best)


# ---------------------------------------------------------------------------
# (C) Chain consistency uplift: 답을 슬롯에 끼웠을 때 정렬이 좋아지는가
# ---------------------------------------------------------------------------
def chain_alignment_score(
    Q: Sequence[Triple],
    D: Sequence[Triple],
    encoder: SentenceEncoder,
) -> float:
    """Q triple 들이 D 와 평균적으로 얼마나 정렬되는가 (0~1)."""
    if not Q or not D:
        return 0.0
    try:
        mat = pairwise_alignment_matrix(Q, D, encoder)
    except Exception as exc:
        logger.warning(f"[verifier] alignment matrix failed: {exc}")
        return 0.0
    if mat.size == 0:
        return 0.0
    best_per_q = mat.max(axis=1)
    return float(np.clip(best_per_q.mean(), 0.0, 1.0))


def _replace_unknown(t: Triple, slot_to_value: Dict[str, str]) -> Triple:
    """triple 의 unknown head/tail 을 slot_to_value 로 치환해서 새 triple 반환."""
    h, tl = t.head, t.tail
    if t.head_unknown:
        sid = _slot_id(t.head)
        if sid and sid in slot_to_value and slot_to_value[sid]:
            h = slot_to_value[sid]
    if t.tail_unknown:
        sid = _slot_id(t.tail)
        if sid and sid in slot_to_value and slot_to_value[sid]:
            tl = slot_to_value[sid]
    return Triple(head=h, relation=t.relation, tail=tl, context=t.context)


def chain_uplift_score(
    answer: str,
    sample: GraphSample,
    fillings: Dict[str, SlotCandidate],
    encoder: SentenceEncoder,
) -> float:
    """답을 가장 약한 슬롯에 끼웠을 때 chain 이 D 와 더 잘 정렬되면 양수.

    구현:
      1. 채워진 슬롯값 vs 답을 비교해 "답의 자리" 슬롯을 추정:
         - 미채워진 슬롯이 있으면 그 자리
         - 없으면 가장 confidence 낮은 슬롯
      2. 그 슬롯값을 답으로 교체한 chain 을 만든 뒤, alignment 재계산
      3. (new - old) 를 [-0.5, +0.5] 로 클립 후 [0,1] 로 리스케일
    """
    if not answer or not sample.Q or not sample.D:
        return 0.5

    sid_to_val: Dict[str, str] = {sid: c.value for sid, c in fillings.items()}
    used_slot_ids: List[str] = []
    for t in sample.Q:
        for tok, unk in [(t.head, t.head_unknown), (t.tail, t.tail_unknown)]:
            if unk:
                sid = _slot_id(tok)
                if sid and sid not in used_slot_ids:
                    used_slot_ids.append(sid)
    if not used_slot_ids:
        return 0.5

    target = None
    for sid in used_slot_ids:
        if sid not in sid_to_val or not sid_to_val[sid]:
            target = sid
            break
    if target is None:
        target = min(used_slot_ids,
                     key=lambda s: fillings[s].score if s in fillings else 0.0)

    Q_old = [_replace_unknown(t, sid_to_val) for t in sample.Q]
    new_map = dict(sid_to_val)
    new_map[target] = answer
    Q_new = [_replace_unknown(t, new_map) for t in sample.Q]

    try:
        old_align = chain_alignment_score(Q_old, sample.D, encoder)
        new_align = chain_alignment_score(Q_new, sample.D, encoder)
    except Exception as exc:
        logger.warning(f"[verifier] chain_uplift failed: {exc}")
        return 0.5

    delta = float(np.clip(new_align - old_align, -0.5, 0.5))
    return float(np.clip(0.5 + delta, 0.0, 1.0))


# ---------------------------------------------------------------------------
# (T) Type match: 답이 Q_def 의 type 정의와 부합하는가
# ---------------------------------------------------------------------------
def type_match_score(
    answer: str,
    sample: GraphSample,
    encoder: Optional[SentenceEncoder] = None,
) -> float:
    """답과 Q_def 의 type 정의 (가장 매칭 점수 높은) 사이의 임베딩 유사도.

    Q_def 가 비어있으면 0.5 (중립) 반환.
    """
    if not answer:
        return 0.0
    if not sample.Q_def:
        return 0.5
    types: List[str] = []
    for t in sample.Q_def:
        if t.head_unknown and t.tail and not is_unknown(t.tail):
            types.append(t.tail)
    if not types:
        return 0.5
    if encoder is None:
        return 0.5
    try:
        a_emb = encoder.encode([answer])[0]
        t_emb = encoder.encode(types)
        sims = t_emb @ a_emb / (
            np.linalg.norm(t_emb, axis=1) * (np.linalg.norm(a_emb) + 1e-12) + 1e-12
        )
        sim = float(np.max(sims)) if len(sims) else 0.5
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# 종합 verifier
# ---------------------------------------------------------------------------
DEFAULT_VERIFIER_WEIGHTS: Dict[str, float] = {
    "grounding": 0.5,
    "chain": 0.3,
    "type": 0.2,
}


@dataclass
class VerificationResult:
    answer: str
    score: float = 0.0
    grounding: float = 0.0
    chain: float = 0.0
    type_match: float = 0.0
    source: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "verify_score": self.score,
            "verify_grounding": self.grounding,
            "verify_chain": self.chain,
            "verify_type": self.type_match,
            "verify_source": self.source,
        }


def verify_answer(
    answer: str,
    sample: GraphSample,
    fillings: Dict[str, SlotCandidate],
    encoder: SentenceEncoder,
    *,
    weights: Optional[Dict[str, float]] = None,
    source: str = "",
) -> VerificationResult:
    """단일 후보에 대한 verifier 점수 산출."""
    w = weights or DEFAULT_VERIFIER_WEIGHTS
    if not answer:
        return VerificationResult(answer="", source=source)

    g = grounding_score(answer, sample.D, encoder)
    c = chain_uplift_score(answer, sample, fillings, encoder)
    t = type_match_score(answer, sample, encoder)

    score = (
        w.get("grounding", 0.5) * g
        + w.get("chain", 0.3) * c
        + w.get("type", 0.2) * t
    )
    return VerificationResult(
        answer=answer,
        score=float(score),
        grounding=float(g),
        chain=float(c),
        type_match=float(t),
        source=source,
    )


@dataclass
class CandidatePool:
    """검증 대상 후보들."""
    items: List[Tuple[str, str]] = field(default_factory=list)   # (source, answer)

    def add(self, source: str, answer: Optional[str]) -> None:
        if not answer:
            return
        self.items.append((source, answer.strip()))

    def unique(self) -> List[Tuple[str, str]]:
        seen: Dict[str, Tuple[str, str]] = {}
        for src, ans in self.items:
            key = _norm_answer(ans)
            if not key:
                continue
            if key in seen:
                continue
            seen[key] = (src, ans)
        return list(seen.values())


def pick_best(
    pool: CandidatePool,
    sample: GraphSample,
    fillings: Dict[str, SlotCandidate],
    encoder: SentenceEncoder,
    *,
    weights: Optional[Dict[str, float]] = None,
    abstain_threshold: float = 0.0,
    abstain_fallback: Optional[str] = None,
    is_yesno: bool = False,
) -> Tuple[VerificationResult, List[VerificationResult], bool]:
    """후보들 중 verifier score 최고 답 선택.

    - is_yesno=True 면 verification 을 우회하고 후보 list 첫 답을 그대로 사용
      (yes/no 는 grounding/chain 기반 검증이 의미가 약함).
    - score 가 abstain_threshold 미만이면 abstain → fallback 또는 첫 후보.

    반환: (best, all_results, abstained)
    """
    cands = pool.unique()
    if not cands:
        return VerificationResult(answer="", source="empty"), [], True

    if is_yesno:
        first_src, first_ans = cands[0]
        r = VerificationResult(answer=first_ans, score=1.0, source=first_src)
        return r, [r], False

    results: List[VerificationResult] = []
    for src, ans in cands:
        r = verify_answer(ans, sample, fillings, encoder,
                          weights=weights, source=src)
        results.append(r)
    results.sort(key=lambda x: x.score, reverse=True)
    best = results[0]
    if best.score < abstain_threshold:
        if abstain_fallback:
            fallback = VerificationResult(
                answer=abstain_fallback, score=best.score, source="abstain_fallback",
                grounding=best.grounding, chain=best.chain, type_match=best.type_match,
            )
            return fallback, results, True
        return best, results, True
    return best, results, False
