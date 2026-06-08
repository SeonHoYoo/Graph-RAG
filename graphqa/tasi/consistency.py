"""Multi-hop propagation consistency.

개선 (v2):
  - hard product 외에 soft 모드 (log-mean, mean) 옵션 추가 → 한 step만 0이어도
    전체 PC가 0 으로 폭발하는 문제 완화.
  - epsilon smoothing (Laplace-style) 으로 sparse multi-hop 그래프(musique 등)에서
    의미 신호가 0 으로 깎이는 현상 방지.
  - "엔터티 토큰 기반 soft 교집합" 옵션: 정확 매칭이 아니라 head/tail 토큰
    overlap 율을 사용 (entity 표면형 차이를 견딤).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence, Set

from graphqa.data.schema import GraphStep, Triple, is_unknown
from graphqa.tasi.ppr import _norm_node

logger = logging.getLogger(__name__)


def _entities_used_in(triples: Sequence[Triple]) -> Set[str]:
    """triple 시퀀스에서 등장한 known entity 집합."""
    out: Set[str] = set()
    for t in triples:
        if not t.head_unknown:
            n = _norm_node(t.head)
            if n:
                out.add(n)
        if not t.tail_unknown:
            n = _norm_node(t.tail)
            if n:
                out.add(n)
    return out


def _entity_token_set(triples: Sequence[Triple]) -> Set[str]:
    """known entity의 토큰(>=3자) 집합 — soft 교집합용."""
    toks: Set[str] = set()
    for t in triples:
        for s in (t.head, t.tail):
            if not s or is_unknown(s):
                continue
            for tok in _norm_node(s).split():
                if len(tok) >= 3:
                    toks.add(tok)
    return toks


def _step_pc(
    E_t: Set[str],
    next_triples: Sequence[Triple],
    *,
    soft: bool,
    epsilon: float,
) -> float:
    """한 step 의 PC(t, t+1) 계산."""
    if not E_t:
        return 1.0

    if soft:
        # token-level soft overlap: |tok(E_t) ∩ tok(next)| / |tok(E_t)|
        a = set()
        for ent in E_t:
            for tok in ent.split():
                if len(tok) >= 3:
                    a.add(tok)
        b = _entity_token_set(next_triples)
        if not a:
            return 1.0
        inter = len(a & b)
        return max(epsilon, inter / len(a))
    else:
        E_next = _entities_used_in(next_triples)
        inter = len(E_t & E_next)
        pc = inter / len(E_t)
        return pc if pc > 0 else epsilon


def propagation_consistency(
    steps: Sequence[GraphStep],
    epsilon: float = 0.05,
    *,
    mode: str = "log_mean",  # "product" | "log_mean" | "mean"
    soft: bool = True,
) -> Dict[str, object]:
    """multi-hop step 간 entity 연속성.

    - PC(t, t+1) = soft 면 토큰 단위 overlap, 아니면 정확 entity 교집합.
    - 결합 방식:
        product : Π PC(t, t+1)               (원래 정의 — 한 step 0이면 폭발)
        log_mean: exp(mean(log(PC + ε)))      (geometric mean, 부드러움)  — 기본
        mean    : Σ PC / (T-1)                (가장 부드러움, multi-hop penalty 약함)

    Args:
        steps: GraphStep 리스트. step 1개 이하면 PC=1.0.
        epsilon: 0 PC 를 epsilon 으로 대체 (smoothing).
        mode  : product | log_mean | mean.
        soft  : 토큰 단위 overlap 사용 여부.

    Returns:
        {
          'pc_total': float,
          'pc_per_step': List[float],
          'entities_per_step': List[set],
        }
    """
    if not steps or len(steps) < 2:
        return {
            "pc_total": 1.0,
            "pc_per_step": [],
            "entities_per_step": [_entities_used_in(s.triples) for s in (steps or [])],
        }

    ents_per_step = [_entities_used_in(s.triples) for s in steps]

    pcs: List[float] = []
    for t in range(len(steps) - 1):
        pcs.append(_step_pc(
            ents_per_step[t], steps[t + 1].triples, soft=soft, epsilon=epsilon,
        ))

    if not pcs:
        pc_total = 1.0
    elif mode == "product":
        pc_total = 1.0
        for v in pcs:
            pc_total *= v
    elif mode == "mean":
        pc_total = sum(pcs) / len(pcs)
    else:  # log_mean (geometric mean)
        log_sum = 0.0
        for v in pcs:
            log_sum += math.log(max(v, epsilon))
        pc_total = math.exp(log_sum / len(pcs))

    return {
        "pc_total": float(pc_total),
        "pc_per_step": pcs,
        "entities_per_step": ents_per_step,
    }
