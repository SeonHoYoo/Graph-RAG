"""PPR-based structural importance for triples."""
from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

from graphqa.data.schema import Triple, is_unknown

logger = logging.getLogger(__name__)


def _norm_node(s: str) -> str:
    """노드 키 정규화."""
    if not s:
        return ""
    return " ".join(s.strip().lower().split())


def triples_to_entities(triples: Sequence[Triple], include_unknown: bool = False) -> Set[str]:
    """triple 시퀀스에서 (정규화된) entity 집합 추출."""
    ents: Set[str] = set()
    for t in triples:
        for tok, unk in [(t.head, t.head_unknown), (t.tail, t.tail_unknown)]:
            if (not include_unknown) and unk:
                continue
            n = _norm_node(tok)
            if n:
                ents.add(n)
    return ents


def build_nx_graph(triples: Sequence[Triple]) -> nx.Graph:
    """triple 리스트 → 무방향 networkx Graph (PPR용).

    edge에는 'relation'과 등장 횟수 'weight'를 저장.
    UNKNOWN placeholder도 노드로 포함시키면 중심성이 왜곡되기 쉬워서
    UNKNOWN끼리 잇는 케이스는 그대로 두되, 가중치는 동일하게 1로 둔다.
    """
    G = nx.Graph()
    for t in triples:
        h = _norm_node(t.head)
        tt = _norm_node(t.tail)
        if not h or not tt:
            continue
        if G.has_edge(h, tt):
            G[h][tt]["weight"] += 1.0
        else:
            G.add_edge(h, tt, weight=1.0, relation=t.relation)
    return G


def compute_ppr(
    triples: Sequence[Triple],
    anchors: Optional[Iterable[str]] = None,
    alpha: float = 0.15,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Personalized PageRank (anchor 기반).

    Args:
        triples: PPR을 돌릴 그래프의 triple 리스트.
        anchors: 시드 entity 문자열 (정규화 전 형태도 OK).
                 None 이면 known entity 전부를 균등 시드로 사용.
        alpha: teleport(restart) 확률 — networkx는 (1-alpha)가 jump 확률이므로
               아래에서 nx.pagerank의 alpha=1-alpha로 변환.
        max_iter, tol: PageRank 수렴 옵션.

    Returns:
        {정규화된 노드명: PPR score}.

    Notes:
        - 그래프가 비어 있으면 빈 dict.
        - 시드 노드가 그래프에 하나도 없으면 uniform PPR 로 fallback.
    """
    G = build_nx_graph(triples)
    if G.number_of_nodes() == 0:
        return {}

    # personalization 벡터
    if anchors is None:
        seeds = {n: 1.0 for n in G.nodes if not is_unknown(n)}
    else:
        seeds = {}
        for a in anchors:
            an = _norm_node(a)
            if an in G.nodes and not is_unknown(an):
                seeds[an] = 1.0

    # 시드가 없으면 uniform 사용
    if not seeds:
        seeds = {n: 1.0 for n in G.nodes}
    s = sum(seeds.values())
    seeds = {k: v / s for k, v in seeds.items()}

    try:
        pr = nx.pagerank(
            G,
            alpha=1.0 - alpha,
            personalization=seeds,
            max_iter=max_iter,
            tol=tol,
            weight="weight",
        )
    except nx.PowerIterationFailedConvergence:
        logger.warning("[ppr] power iteration failed, falling back to uniform")
        pr = {n: 1.0 / G.number_of_nodes() for n in G.nodes}
    return pr


def triple_weight(
    triple: Triple,
    ppr_scores: Dict[str, float],
    fallback_strategy: str = "mean",
) -> float:
    """w(τ) = sqrt(PPR(h) · PPR(t)).

    UNKNOWN 노드는 PPR 평균값으로 대체.
    fallback_strategy ∈ {"mean", "min", "zero"}.
    """
    if not ppr_scores:
        return 1.0  # PPR을 못 만들면 균등 가중치로 fallback

    if fallback_strategy == "mean":
        fallback = float(np.mean(list(ppr_scores.values())))
    elif fallback_strategy == "min":
        fallback = float(min(ppr_scores.values()))
    else:
        fallback = 0.0

    h = _norm_node(triple.head)
    t = _norm_node(triple.tail)
    p_h = fallback if (triple.head_unknown or h not in ppr_scores) else ppr_scores[h]
    p_t = fallback if (triple.tail_unknown or t not in ppr_scores) else ppr_scores[t]
    p_h = max(p_h, 1e-12)
    p_t = max(p_t, 1e-12)
    return math.sqrt(p_h * p_t)


def compute_triple_weights(
    triples: Sequence[Triple],
    ppr_scores: Dict[str, float],
    normalize: bool = True,
    fallback_strategy: str = "mean",
) -> np.ndarray:
    """각 triple의 weight를 numpy array로 반환 (옵션: 합=1로 정규화)."""
    w = np.array(
        [triple_weight(t, ppr_scores, fallback_strategy) for t in triples],
        dtype=np.float64,
    )
    if normalize and w.sum() > 0:
        w = w / w.sum()
    return w
