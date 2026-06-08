"""Triple-level alignment & graph free-matching."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from graphqa.data.schema import Triple, is_unknown
from graphqa.tasi.embedding import SentenceEncoder

logger = logging.getLogger(__name__)


# 가중치 기본값 (논문 제안)
DEFAULT_W_S = 0.35
DEFAULT_W_R = 0.30
DEFAULT_W_O = 0.35


def _normalize_entity(s: str) -> str:
    """간단한 entity 정규화 (lowercase, 공백 정리)."""
    if not s:
        return ""
    return " ".join(s.strip().lower().split())


def _entity_sim(
    e_a: str,
    e_b: str,
    encoder: SentenceEncoder,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """두 entity 문자열 간 cosine similarity. 동일 문자열은 1.0."""
    a = _normalize_entity(e_a)
    b = _normalize_entity(e_b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if embedding_cache is not None:
        if a not in embedding_cache:
            embedding_cache[a] = encoder.encode_one(a)
        if b not in embedding_cache:
            embedding_cache[b] = encoder.encode_one(b)
        return float(np.dot(embedding_cache[a], embedding_cache[b]))
    return encoder.cosine(a, b)


def _relation_sim_directional(
    r_a: str,
    r_b: str,
    encoder: SentenceEncoder,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """방향이 같다고 가정한 relation 유사도."""
    return _entity_sim(r_a, r_b, encoder, embedding_cache)


def align_triple(
    tau_A: Triple,
    tau_B: Triple,
    encoder: SentenceEncoder,
    w_S: float = DEFAULT_W_S,
    w_R: float = DEFAULT_W_R,
    w_O: float = DEFAULT_W_O,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    allow_inverse: bool = True,
) -> Tuple[float, bool]:
    """두 triple 간 의미 유사도.

    Returns:
        (score, used_inverse): 0~1 범위 정렬 점수와 역방향 매칭 여부.

    핵심:
      - UNKNOWN 슬롯이 한쪽이라도 있는 위치는 계산에서 제외하고
        남은 가중치만 사용해 정규화.
      - allow_inverse=True 면 (S_B↔O_B) 스왑한 경우와 비교 후 더 높은 쪽 채택
        → relation 방향 차이 ('directed by' ↔ 'directed') 자동 처리.
    """

    def score_with(reverse_b: bool) -> float:
        if reverse_b:
            sb_h, sb_t = tau_B.tail, tau_B.head
            sb_h_unk, sb_t_unk = tau_B.tail_unknown, tau_B.head_unknown
        else:
            sb_h, sb_t = tau_B.head, tau_B.tail
            sb_h_unk, sb_t_unk = tau_B.head_unknown, tau_B.tail_unknown

        weights: List[float] = []
        sims: List[float] = []

        # Subject
        if not (tau_A.head_unknown or sb_h_unk):
            sims.append(_entity_sim(tau_A.head, sb_h, encoder, embedding_cache))
            weights.append(w_S)

        # Relation (방향 차이는 reverse_b 케이스에서 흡수)
        if tau_A.relation and tau_B.relation:
            sims.append(_relation_sim_directional(tau_A.relation, tau_B.relation, encoder, embedding_cache))
            weights.append(w_R)

        # Object
        if not (tau_A.tail_unknown or sb_t_unk):
            sims.append(_entity_sim(tau_A.tail, sb_t, encoder, embedding_cache))
            weights.append(w_O)

        if not weights:
            # 양쪽이 전부 UNKNOWN인 경우 → relation only로도 못 보면 0
            return 0.0
        ws = sum(weights)
        return sum(w * s for w, s in zip(weights, sims)) / ws

    fwd = score_with(reverse_b=False)
    if not allow_inverse:
        return float(fwd), False
    rev = score_with(reverse_b=True)
    if rev > fwd:
        return float(rev), True
    return float(fwd), False


def pairwise_alignment_matrix(
    A: Sequence[Triple],
    B: Sequence[Triple],
    encoder: SentenceEncoder,
    w_S: float = DEFAULT_W_S,
    w_R: float = DEFAULT_W_R,
    w_O: float = DEFAULT_W_O,
    allow_inverse: bool = True,
) -> np.ndarray:
    """A(N) × B(M) alignment 행렬을 batch encoding으로 빠르게 계산.

    Note:
        이 함수는 cache-friendly 방식으로 모든 entity / relation 문자열을 한 번에
        인코딩한 다음 dot product로 행렬을 만들기 때문에 align_triple를 NxM번
        호출하는 것보다 훨씬 빠르다.
    """
    N, M = len(A), len(B)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # 모든 텍스트 모아서 batch encode
    texts: List[str] = []
    idx: Dict[str, int] = {}

    def add(text: str) -> int:
        t = _normalize_entity(text)
        if t in idx:
            return idx[t]
        idx[t] = len(texts)
        texts.append(t)
        return idx[t]

    a_pieces = [(add(t.head), add(t.relation), add(t.tail), t.head_unknown, t.tail_unknown) for t in A]
    b_pieces = [(add(t.head), add(t.relation), add(t.tail), t.head_unknown, t.tail_unknown) for t in B]
    embs = encoder.encode(texts)  # (V, D), L2 normalized

    # 각 row: N개의 (head, rel, tail) 인덱스
    out = np.zeros((N, M), dtype=np.float32)
    for i, (ah, ar, at, a_h_unk, a_t_unk) in enumerate(a_pieces):
        a_h_vec = embs[ah]
        a_r_vec = embs[ar]
        a_t_vec = embs[at]
        for j, (bh, br, bt, b_h_unk, b_t_unk) in enumerate(b_pieces):
            # forward
            weights, sims = [], []
            if not (a_h_unk or b_h_unk):
                sims.append(float(np.dot(a_h_vec, embs[bh])))
                weights.append(w_S)
            if A[i].relation and B[j].relation:
                sims.append(float(np.dot(a_r_vec, embs[br])))
                weights.append(w_R)
            if not (a_t_unk or b_t_unk):
                sims.append(float(np.dot(a_t_vec, embs[bt])))
                weights.append(w_O)
            fwd = (sum(w * s for w, s in zip(weights, sims)) / sum(weights)) if weights else 0.0

            # reverse (B의 head/tail 스왑 — "directed by" vs "directed" 케이스)
            if allow_inverse:
                weights, sims = [], []
                if not (a_h_unk or b_t_unk):
                    sims.append(float(np.dot(a_h_vec, embs[bt])))
                    weights.append(w_S)
                if A[i].relation and B[j].relation:
                    sims.append(float(np.dot(a_r_vec, embs[br])))
                    weights.append(w_R)
                if not (a_t_unk or b_h_unk):
                    sims.append(float(np.dot(a_t_vec, embs[bh])))
                    weights.append(w_O)
                rev = (sum(w * s for w, s in zip(weights, sims)) / sum(weights)) if weights else 0.0
            else:
                rev = 0.0

            out[i, j] = max(fwd, rev)
    return out


def pairwise_alignment_matrix_field_min(
    A: Sequence[Triple],
    B: Sequence[Triple],
    encoder: SentenceEncoder,
    allow_inverse: bool = True,
) -> np.ndarray:
    """A(N)×B(M) 셀마다: (주어·관계·목적어) 각각 임베딩 코사인을 **독립**으로 보고,

    - UNKNOWN 이 아닌 위치만 포함해, 한 방향(정방향 / B의 head·tail 스왑) 안에서는
      그 방향에서 쓰인 코사인들의 **최솟값**을 셀 점수 후보로 쓴다.
    - 정방향 후보와 스왑 후보 중 **더 큰 값**을 셀 최종 점수로 한다 (수동·능동 순서).

    가중 평균이 아니라 **min** 이므로 graphqa.alignment 의 pair 집계(AND 성향)와 맞춘다.
    """
    N, M = len(A), len(B)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    texts: List[str] = []
    idx: Dict[str, int] = {}

    def add(text: str) -> int:
        t = _normalize_entity(text)
        if t in idx:
            return idx[t]
        idx[t] = len(texts)
        texts.append(t)
        return idx[t]

    a_pieces = [(add(t.head), add(t.relation), add(t.tail), t.head_unknown, t.tail_unknown) for t in A]
    b_pieces = [(add(t.head), add(t.relation), add(t.tail), t.head_unknown, t.tail_unknown) for t in B]
    embs = encoder.encode(texts)

    out = np.zeros((N, M), dtype=np.float32)
    for i, (ah, ar, at, a_h_unk, a_t_unk) in enumerate(a_pieces):
        a_h_vec = embs[ah]
        a_r_vec = embs[ar]
        a_t_vec = embs[at]
        for j, (bh, br, bt, b_h_unk, b_t_unk) in enumerate(b_pieces):
            fwd_sims: List[float] = []
            if not (a_h_unk or b_h_unk):
                fwd_sims.append(float(np.dot(a_h_vec, embs[bh])))
            if A[i].relation and B[j].relation:
                fwd_sims.append(float(np.dot(a_r_vec, embs[br])))
            if not (a_t_unk or b_t_unk):
                fwd_sims.append(float(np.dot(a_t_vec, embs[bt])))
            fwd_min = min(fwd_sims) if fwd_sims else None

            rev_min: Optional[float] = None
            if allow_inverse:
                rev_sims: List[float] = []
                if not (a_h_unk or b_t_unk):
                    rev_sims.append(float(np.dot(a_h_vec, embs[bt])))
                if A[i].relation and B[j].relation:
                    rev_sims.append(float(np.dot(a_r_vec, embs[br])))
                if not (a_t_unk or b_h_unk):
                    rev_sims.append(float(np.dot(a_t_vec, embs[bh])))
                rev_min = min(rev_sims) if rev_sims else None

            cands = [x for x in (fwd_min, rev_min) if x is not None]
            out[i, j] = float(max(cands)) if cands else 0.0
    return out


def free_matching(
    A: Sequence[Triple],
    B: Sequence[Triple],
    encoder: SentenceEncoder,
    weights_A: Optional[np.ndarray] = None,
    use_hungarian: bool = False,
    w_S: float = DEFAULT_W_S,
    w_R: float = DEFAULT_W_R,
    w_O: float = DEFAULT_W_O,
    allow_inverse: bool = True,
) -> Dict[str, object]:
    """A의 각 triple을 B에서 가장 잘 맞는 triple과 매칭.

    use_hungarian=False (기본): 각 τ_A에 대해 max over τ_B (자유 매칭).
    use_hungarian=True: scipy linear_sum_assignment로 1:1 매칭.

    Returns:
        {
          'matrix': (N, M) alignment matrix,
          'best_idx': N-vector (각 A_i 의 best B_j),
          'best_score': N-vector (best score),
          'mean_score': float (단순 평균),
          'weighted_score': float (weights_A 가중 평균),
          'matched_pairs': List[(i, j, score)],
        }
    """
    N, M = len(A), len(B)
    if N == 0 or M == 0:
        return {
            "matrix": np.zeros((N, M), dtype=np.float32),
            "best_idx": np.zeros(N, dtype=np.int64),
            "best_score": np.zeros(N, dtype=np.float32),
            "mean_score": 0.0,
            "weighted_score": 0.0,
            "matched_pairs": [],
        }

    matrix = pairwise_alignment_matrix(
        A, B, encoder,
        w_S=w_S, w_R=w_R, w_O=w_O,
        allow_inverse=allow_inverse,
    )

    if use_hungarian and N > 0 and M > 0:
        # 1:1 (size-mismatch 시 자동 padding 처리)
        cost = -matrix
        if N > M:
            pad = np.full((N, N - M), 1e6, dtype=cost.dtype)
            cost = np.concatenate([cost, pad], axis=1)
        elif M > N:
            pad = np.full((M - N, M), 1e6, dtype=cost.dtype)
            cost = np.concatenate([cost, pad], axis=0)
        row_ind, col_ind = linear_sum_assignment(cost)
        best_idx = np.zeros(N, dtype=np.int64)
        best_score = np.zeros(N, dtype=np.float32)
        matched_pairs: List[Tuple[int, int, float]] = []
        for r, c in zip(row_ind, col_ind):
            if r >= N:
                continue
            if c >= M:
                # 매칭 못함
                best_idx[r] = -1
                best_score[r] = 0.0
                continue
            best_idx[r] = c
            best_score[r] = float(matrix[r, c])
            matched_pairs.append((int(r), int(c), float(matrix[r, c])))
    else:
        best_idx = np.argmax(matrix, axis=1).astype(np.int64)
        best_score = matrix[np.arange(N), best_idx].astype(np.float32)
        matched_pairs = [(int(i), int(j), float(s))
                         for i, (j, s) in enumerate(zip(best_idx, best_score))]

    if weights_A is not None and len(weights_A) == N and float(np.sum(weights_A)) > 0:
        weighted = float(np.sum(weights_A * best_score) / np.sum(weights_A))
    else:
        weighted = float(np.mean(best_score)) if N > 0 else 0.0

    return {
        "matrix": matrix,
        "best_idx": best_idx,
        "best_score": best_score,
        "mean_score": float(np.mean(best_score)) if N > 0 else 0.0,
        "weighted_score": weighted,
        "matched_pairs": matched_pairs,
    }
