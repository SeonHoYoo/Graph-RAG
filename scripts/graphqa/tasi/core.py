"""TASI = WA(A,B) × PC(A) — 통합 함수."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from graphqa.data.schema import GraphStep, Triple
from graphqa.tasi.align import (
    DEFAULT_W_O,
    DEFAULT_W_R,
    DEFAULT_W_S,
    free_matching,
)
from graphqa.tasi.consistency import propagation_consistency
from graphqa.tasi.embedding import SentenceEncoder
from graphqa.tasi.ppr import (
    compute_ppr,
    compute_triple_weights,
    triples_to_entities,
)

logger = logging.getLogger(__name__)


@dataclass
class TasiResult:
    """TASI 계산 결과 컨테이너."""
    tasi: float
    wa: float                      # weighted alignment Σ w(τ_A) · max align(τ_A, τ_B)
    pc: float                      # propagation consistency (Π)
    n_a: int = 0
    n_b: int = 0
    matched_pairs: List = field(default_factory=list)
    pc_per_step: List[float] = field(default_factory=list)
    weights_a: Optional[np.ndarray] = None
    best_score: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "tasi": self.tasi,
            "wa": self.wa,
            "pc": self.pc,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "pc_per_step": self.pc_per_step,
            "matched_pairs": self.matched_pairs,
        }


def tasi(
    graph_A: Sequence[Triple],
    graph_B: Sequence[Triple],
    encoder: SentenceEncoder,
    anchors: Optional[Iterable[str]] = None,
    steps: Optional[Sequence[GraphStep]] = None,
    *,
    w_S: float = DEFAULT_W_S,
    w_R: float = DEFAULT_W_R,
    w_O: float = DEFAULT_W_O,
    use_hungarian: bool = False,
    allow_inverse: bool = True,
    ppr_alpha: float = 0.15,
    pc_mode: str = "log_mean",
    pc_soft: bool = True,
    pc_epsilon: float = 0.05,
) -> TasiResult:
    """최종 TASI score 계산.

    TASI(A, B) = WA(A, B) × PC

    - WA(A, B) = Σ w(τ_A) · max_{τ_B} align(τ_A, τ_B), normalize Σw=1
    - w(τ_A)는 graph_A 상의 PPR 기반 구조 중요도
    - anchors=None 이면 graph_A의 모든 known entity가 시드
    - steps=None 이면 PC=1.0

    Args:
        graph_A: anchor 그래프 (스코어 분모 / 가중치 출처)
        graph_B: 비교 대상 그래프
        encoder: sentence encoder
        anchors: PPR 시드 entity 리스트
        steps: A의 multi-hop step 리스트 (PC 계산용)
        use_hungarian: True면 1:1 매칭, False면 free matching (max)
        allow_inverse: relation 방향 자동 반전 허용
    """
    n_a = len(graph_A)
    n_b = len(graph_B)
    if n_a == 0 or n_b == 0:
        return TasiResult(tasi=0.0, wa=0.0, pc=1.0, n_a=n_a, n_b=n_b)

    # 1) PPR-based weights on A
    if anchors is None:
        anchor_set = list(triples_to_entities(graph_A, include_unknown=False))
    else:
        anchor_set = list(anchors)
    ppr = compute_ppr(graph_A, anchors=anchor_set, alpha=ppr_alpha)
    weights = compute_triple_weights(graph_A, ppr, normalize=True)

    # 2) Weighted alignment (free matching)
    fm = free_matching(
        graph_A, graph_B, encoder,
        weights_A=weights,
        use_hungarian=use_hungarian,
        w_S=w_S, w_R=w_R, w_O=w_O,
        allow_inverse=allow_inverse,
    )
    wa = float(fm["weighted_score"])

    # 3) Propagation consistency
    if steps:
        pc_info = propagation_consistency(
            steps, epsilon=pc_epsilon, mode=pc_mode, soft=pc_soft,
        )
        pc = float(pc_info["pc_total"])
        pc_per_step = list(pc_info["pc_per_step"])
    else:
        pc = 1.0
        pc_per_step = []

    score = float(wa * pc)
    return TasiResult(
        tasi=score,
        wa=wa,
        pc=pc,
        n_a=n_a,
        n_b=n_b,
        matched_pairs=fm["matched_pairs"],
        pc_per_step=pc_per_step,
        weights_a=weights,
        best_score=fm["best_score"],
    )
