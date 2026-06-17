"""Triple 단위 sentence-embedding cosine alignment.

TASI 의 free-matching alignment 와 별개로, *순수* sentence cosine 만으로
두 triple 집합 간 정렬 점수와 binary 라벨을 매긴다. 사용 목적:

  (1) (Q, D) / (Q, Sr) / (Q, T) 의 alignment 0/1 × QA 정답 여부의
      2×2 confusion matrix 분석.
  (2) LLM prompt 에 "alignment quality signal" 로 주입.

**Triple 쌍 셀 점수:** head / relation / tail 을 각각 임베딩해 코사인을 낸 뒤,
UNKNOWN 이 아닌 필드만 모아 **그중 최솟값**을 한 방향의 점수로 쓴다.
B 쪽이 수동태 등으로 주어·목적어가 바뀐 경우를 위해 **B 의 head·tail 스왑**
방향도 같은 식으로 계산하고, 두 방향 중 **큰 값**을 셀 점수로 한다
(`pairwise_alignment_matrix_field_min`).

**집계 규칙 (샘플):** 각 query triple i 에 대해 ``m_i = max_j cell(Q_i, B_j)``,
샘플 점수는 **min_i m_i** (AND).  참고용 **mean_i m_i** 는 ``align_*_mean``.

scale 직관 (sentence-transformers/all-MiniLM-L6-v2 기준):
  - 의미적으로 거의 동일한 두 문장   : ~0.80–0.95
  - 같은 주제, 다른 표현              : ~0.55–0.75
  - 살짝 관련                         : ~0.35–0.55
  - 무관                              : ~0.00–0.30

→ 본 코드의 default threshold = 0.50.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from graphqa.data.schema import GraphSample, Triple
from graphqa.tasi.align import pairwise_alignment_matrix_field_min
from graphqa.tasi.embedding import SentenceEncoder


DEFAULT_ALIGN_THRESHOLDS: Sequence[float] = (0.30, 0.40, 0.50, 0.60, 0.70)
DEFAULT_ALIGN_THRESHOLD: float = 0.50


def pair_alignment(
    A: Sequence[Triple],
    B: Sequence[Triple],
    encoder: SentenceEncoder,
) -> Dict[str, object]:
    """A 의 각 triple 에 대해 B 와의 최고 셀 점수를 구한 뒤, **min 으로 집계**(AND).

    셀 점수 ``cell(Q_i, B_j)``: head/relation/tail 각각 임베딩 코사인(UNKNOWN 제외),
    필드별 값의 **min**; 정방향 vs B head/tail 스왑 중 **max**.
    각 query triple i: ``m_i = max_j cell(Q_i, B_j)``.
    샘플 점수 ``score = min_i m_i`` → **모든** Q 줄이 각각 어느 B 와 τ 이상으로
    맞아야 ``score >= τ`` (align Yes).  (한 줄만 잘 맞고 나머지는 낮아도
    mean 은 높게 나올 수 있어서, mean 집계는 ``score_mean_max`` 로 별도 보관.)

    반환:
      {
        "score": float (min-of-max, AND 집계),
        "score_mean_max": float (mean-of-max, 완화된 참고값),
        "max_per_a": np.ndarray (|A|,),  각 triple_a 의 best 셀 점수
        "n_a": int, "n_b": int,
        "frac_aligned": dict (각 τ에 대해 m_i>=τ 인 Q triple 비율)
      }
    """
    out: Dict[str, object] = {
        "score": float("nan"),
        "score_mean_max": float("nan"),
        "max_per_a": np.zeros((0,), dtype=np.float32),
        "n_a": int(len(A)),
        "n_b": int(len(B)),
        "frac_aligned": {},
    }
    if not A or not B:
        return out
    sim = pairwise_alignment_matrix_field_min(A, B, encoder, allow_inverse=True)
    if sim.size == 0:
        return out
    max_per_a = sim.max(axis=1)
    out["score"] = float(np.min(max_per_a))
    out["score_mean_max"] = float(np.mean(max_per_a))
    out["max_per_a"] = max_per_a.astype(np.float32)
    out["frac_aligned"] = {
        f"{tau:.2f}": float((max_per_a >= tau).mean())
        for tau in DEFAULT_ALIGN_THRESHOLDS
    }
    return out


@dataclass
class SampleAlignment:
    """샘플 한 개에 대한 3 pair alignment summary."""
    align_QD_score: float = float("nan")    # query ↔ document (min-of-max, AND)
    align_QSr_score: float = float("nan")   # query ↔ search-rewrite
    align_QT_score: float = float("nan")    # query ↔ think
    align_QD_mean: float = float("nan")   # mean-of-max (참고·완화)
    align_QSr_mean: float = float("nan")
    align_QT_mean: float = float("nan")
    align_QD_frac: Dict[str, float] = None  # type: ignore[assignment]
    align_QSr_frac: Dict[str, float] = None  # type: ignore[assignment]
    align_QT_frac: Dict[str, float] = None  # type: ignore[assignment]

    def to_row(self) -> Dict[str, float]:
        row: Dict[str, float] = {
            "align_QD": self.align_QD_score,
            "align_QSr": self.align_QSr_score,
            "align_QT": self.align_QT_score,
            "align_QD_mean": self.align_QD_mean,
            "align_QSr_mean": self.align_QSr_mean,
            "align_QT_mean": self.align_QT_mean,
        }
        for tau in DEFAULT_ALIGN_THRESHOLDS:
            t = f"{tau:.2f}"
            row[f"align_QD_bin@{t}"] = float(int(self.align_QD_score >= tau)) \
                if np.isfinite(self.align_QD_score) else float("nan")
            row[f"align_QSr_bin@{t}"] = float(int(self.align_QSr_score >= tau)) \
                if np.isfinite(self.align_QSr_score) else float("nan")
            row[f"align_QT_bin@{t}"] = float(int(self.align_QT_score >= tau)) \
                if np.isfinite(self.align_QT_score) else float("nan")
        return row


def compute_sample_alignment(
    sample: GraphSample,
    encoder: SentenceEncoder,
) -> SampleAlignment:
    """sample 한 개에 대해 (Q,D) (Q,Sr) (Q,T) alignment score 산출."""
    qd = pair_alignment(sample.Q, sample.D, encoder) if sample.D else {}
    qsr = pair_alignment(sample.Q, sample.Sr, encoder) if sample.Sr else {}
    qt = pair_alignment(sample.Q, sample.T, encoder) if sample.T else {}
    return SampleAlignment(
        align_QD_score=float(qd.get("score", float("nan"))) if qd else float("nan"),
        align_QSr_score=float(qsr.get("score", float("nan"))) if qsr else float("nan"),
        align_QT_score=float(qt.get("score", float("nan"))) if qt else float("nan"),
        align_QD_mean=float(qd.get("score_mean_max", float("nan"))) if qd else float("nan"),
        align_QSr_mean=float(qsr.get("score_mean_max", float("nan"))) if qsr else float("nan"),
        align_QT_mean=float(qt.get("score_mean_max", float("nan"))) if qt else float("nan"),
        align_QD_frac=qd.get("frac_aligned", {}) if qd else {},
        align_QSr_frac=qsr.get("frac_aligned", {}) if qsr else {},
        align_QT_frac=qt.get("frac_aligned", {}) if qt else {},
    )


def alignment_label(score: float, threshold: float = DEFAULT_ALIGN_THRESHOLD) -> int:
    """score >= threshold → 1, else 0. NaN → 0."""
    if score is None or not np.isfinite(score):
        return 0
    return int(score >= threshold)


def alignment_quality_label(score: float) -> str:
    """LLM prompt 표시용 인간 친화 라벨."""
    if score is None or not np.isfinite(score):
        return "unknown"
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def confusion_matrix(
    align_bin: Sequence[int],
    correct_bin: Sequence[int],
) -> Dict[str, int]:
    """2x2 confusion matrix.

    반환 keys: 'TP','FN','FP','TN'  (alignment=positive, correct=positive).
      TP : align=1 & correct=1     (잘 align 됐고 답도 맞음)
      FN : align=0 & correct=1     (align 안 됐는데 답은 맞음)
      FP : align=1 & correct=0     (align 됐는데 답 틀림 — 위험)
      TN : align=0 & correct=0     (align 안 됐고 답도 틀림)
    """
    a = np.asarray(align_bin, dtype=int)
    c = np.asarray(correct_bin, dtype=int)
    tp = int(((a == 1) & (c == 1)).sum())
    fn = int(((a == 0) & (c == 1)).sum())
    fp = int(((a == 1) & (c == 0)).sum())
    tn = int(((a == 0) & (c == 0)).sum())
    n = tp + fn + fp + tn
    out = {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "n": n}
    if n == 0:
        return out
    out["align_rate"] = (tp + fp) / n
    out["correct_rate"] = (tp + fn) / n
    out["acc_when_aligned"] = tp / max(1, tp + fp)
    out["acc_when_not_aligned"] = fn / max(1, fn + tn)
    # phi (= matthews-style) correlation for binary
    denom = np.sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    out["phi"] = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return out
