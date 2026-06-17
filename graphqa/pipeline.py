"""Module 2: TASIPipeline — 다섯 가지 그래프 쌍 비교.

개선 (v2):
  - PC를 single-step 비교에서도 정의 가능하도록 호출부 정리 (PC=1로 둠).
  - 곱셈 total 외에 weighted sum total 추가 — 분리력(AUC)이 더 좋은 형태.
  - PC 파라미터(mode, soft, epsilon)를 외부에서 주입 가능.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from graphqa.data.schema import GraphSample, GraphStep, Triple
from graphqa.tasi.core import TasiResult, tasi
from graphqa.tasi.embedding import SentenceEncoder, get_default_encoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weighted-sum total 의 기본 가중치 (AUC 친화적)
#
# 데이터 분석에서 단일 신호 분리력 (AUC) 순서가
#   relevance(0.637) > search_quality(0.612) > retrieval(0.587)
#   > consistency(0.587) > alignment(0.586)
# 였음. relevance, search_quality 에 더 큰 가중치를 주고, retrieval/alignment 는
# 약하게 두었다.
# ---------------------------------------------------------------------------
DEFAULT_TOTAL_WEIGHTS = {
    "relevance": 0.30,
    "consistency": 0.20,
    "alignment": 0.20,
    "search_quality": 0.15,
    "retrieval": 0.15,
}


@dataclass
class PipelineScores:
    """5가지 TASI score + 통합 점수.

    네이밍:
      - relevance     : TASI(Q,  D)
      - consistency   : TASI(T,  D)
      - alignment     : TASI(T,  Q)
      - search_quality: TASI(Sr, Q)
      - retrieval     : TASI(Sr, D)
      - total_product : 위 5개의 곱  (원래 정의)
      - total_sum     : 위 5개의 가중합 (AUC 친화적, 기본 total)
    """
    relevance: TasiResult
    consistency: TasiResult
    alignment: TasiResult
    search_quality: TasiResult
    retrieval: TasiResult
    total_product: float = 0.0
    total_sum: float = 0.0
    weights_total: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        ws = self.weights_total or DEFAULT_TOTAL_WEIGHTS
        self.weights_total = ws

        self.total_product = float(
            self.relevance.tasi
            * self.consistency.tasi
            * self.alignment.tasi
            * self.search_quality.tasi
            * self.retrieval.tasi
        )
        wsum = (
            ws["relevance"] * self.relevance.tasi
            + ws["consistency"] * self.consistency.tasi
            + ws["alignment"] * self.alignment.tasi
            + ws["search_quality"] * self.search_quality.tasi
            + ws["retrieval"] * self.retrieval.tasi
        )
        denom = sum(ws.values()) or 1.0
        self.total_sum = float(wsum / denom)

    @property
    def total(self) -> float:
        """기본 total = weighted sum (AUC 친화적)."""
        return self.total_sum

    def to_flat_dict(self) -> Dict[str, float]:
        """단일 평탄화 사전 — 표/CSV 출력용."""
        out: Dict[str, float] = {}
        for name in ("relevance", "consistency", "alignment", "search_quality", "retrieval"):
            r: TasiResult = getattr(self, name)
            out[f"{name}_tasi"] = r.tasi
            out[f"{name}_wa"] = r.wa
            out[f"{name}_pc"] = r.pc
        out["total_tasi_score"] = self.total_sum   # primary
        out["total_product"] = self.total_product
        out["total_sum"] = self.total_sum
        return out


class TASIPipeline:
    """4 그래프 (Q, T, D, Sr) → 5개 TASI 점수 산출."""

    def __init__(
        self,
        encoder: Optional[SentenceEncoder] = None,
        *,
        use_hungarian: bool = False,
        allow_inverse: bool = True,
        use_steps_for_T: bool = True,
        ppr_alpha: float = 0.15,
        pc_mode: str = "log_mean",
        pc_soft: bool = True,
        pc_epsilon: float = 0.05,
        total_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.encoder = encoder or get_default_encoder()
        self.use_hungarian = use_hungarian
        self.allow_inverse = allow_inverse
        self.use_steps_for_T = use_steps_for_T
        self.ppr_alpha = ppr_alpha
        self.pc_mode = pc_mode
        self.pc_soft = pc_soft
        self.pc_epsilon = pc_epsilon
        self.total_weights = total_weights or DEFAULT_TOTAL_WEIGHTS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def compute_all_scores(
        self,
        Q: Sequence[Triple],
        T: Sequence[Triple],
        D: Sequence[Triple],
        Sr: Sequence[Triple],
        *,
        T_steps: Optional[Sequence[GraphStep]] = None,
        Q_anchors: Optional[Sequence[str]] = None,
    ) -> PipelineScores:
        """5가지 비교 모두 수행."""
        steps_for_T = T_steps if (self.use_steps_for_T and T_steps) else None

        common = dict(
            use_hungarian=self.use_hungarian,
            allow_inverse=self.allow_inverse,
            ppr_alpha=self.ppr_alpha,
            pc_mode=self.pc_mode,
            pc_soft=self.pc_soft,
            pc_epsilon=self.pc_epsilon,
        )

        relevance = tasi(Q, D, self.encoder, anchors=Q_anchors, **common)
        consistency = tasi(T, D, self.encoder, steps=steps_for_T, **common)
        alignment = tasi(T, Q, self.encoder, steps=steps_for_T, **common)
        search_quality = tasi(Sr, Q, self.encoder, **common)
        retrieval = tasi(Sr, D, self.encoder, **common)

        return PipelineScores(
            relevance=relevance,
            consistency=consistency,
            alignment=alignment,
            search_quality=search_quality,
            retrieval=retrieval,
            weights_total=self.total_weights,
        )

    def score_sample(self, sample: GraphSample) -> PipelineScores:
        """GraphSample 객체 한 건에 대해 모든 score 계산."""
        return self.compute_all_scores(
            Q=sample.Q + sample.Q_def,
            T=sample.T,
            D=sample.D,
            Sr=sample.Sr,
            T_steps=sample.T_steps,
        )
