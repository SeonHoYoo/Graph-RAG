"""Module 1 단위 테스트 (assert 기반, pytest 없이도 실행 가능)."""
from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import logging
import math

import numpy as np

from graphqa.data.schema import GraphStep, Triple, is_unknown, parse_triple
from graphqa.tasi.align import align_triple, free_matching, pairwise_alignment_matrix
from graphqa.tasi.consistency import propagation_consistency
from graphqa.tasi.core import tasi
from graphqa.tasi.embedding import get_default_encoder
from graphqa.tasi.ppr import (
    build_nx_graph,
    compute_ppr,
    compute_triple_weights,
    triple_weight,
    triples_to_entities,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")


def test_schema_unknown():
    assert is_unknown("(ENT1)")
    assert is_unknown("ENT2")
    assert is_unknown("?S0")
    assert not is_unknown("Slums of Berlin")
    assert is_unknown("")  # empty도 unknown 처리

    t = Triple.from_str("(ENT1) [SEP] is the director of [SEP] film Slums Of Berlin")
    assert t is not None
    assert t.head == "(ENT1)" and t.tail == "film Slums Of Berlin"
    assert t.head_unknown and not t.tail_unknown
    print("[ok] schema_unknown")


def test_align_basic():
    enc = get_default_encoder()
    t1 = Triple.from_str("Slums of Berlin [SEP] is directed by [SEP] Gerhard Lamprecht")
    t2 = Triple.from_str("Gerhard Lamprecht [SEP] directed [SEP] Slums of Berlin")
    s_dir, used_inv_dir = align_triple(t1, t2, enc, allow_inverse=False)
    s_inv, used_inv = align_triple(t1, t2, enc, allow_inverse=True)

    print(f"  align (forward only) = {s_dir:.3f}")
    print(f"  align (allow inverse)= {s_inv:.3f}, used_inverse={used_inv}")
    assert s_inv >= s_dir, "역방향 허용 시 점수가 줄면 안됨"
    assert s_inv > 0.5, "동일한 의미의 triple은 0.5 이상이어야 함"
    assert used_inv, "head/tail 스왑이 더 잘 맞을 거라 기대"
    print("[ok] align_basic")


def test_align_unknown():
    enc = get_default_encoder()
    # Q triple: (ENT1) directs Slums Of Berlin
    q = Triple.from_str("(ENT1) [SEP] is the director of [SEP] film Slums Of Berlin")
    # D triple: Gerhard Lamprecht directed Slums of Berlin
    d = Triple.from_str("Slums of Berlin [SEP] is directed by [SEP] Gerhard Lamprecht")
    s, used_inv = align_triple(q, d, enc, allow_inverse=True)
    print(f"  Q-D align = {s:.3f}, inverse={used_inv}")
    # head가 UNKNOWN이므로 head 부분 제외하고 relation+object만 비교
    # → "is the director of"+"film Slums Of Berlin" vs "is directed by"+"Slums of Berlin"
    assert 0.4 < s <= 1.0
    print("[ok] align_unknown")


def test_pairwise_matrix():
    enc = get_default_encoder()
    A = [
        Triple.from_str("(ENT1) [SEP] is the director of [SEP] film Slums Of Berlin"),
        Triple.from_str("(ENT1) [SEP] is from [SEP] (ENT3)"),
    ]
    B = [
        Triple.from_str("Slums of Berlin [SEP] is directed by [SEP] Gerhard Lamprecht"),
        Triple.from_str("Gerhard Lamprecht [SEP] was [SEP] a German film director"),
        Triple.from_str("Phantom India [SEP] directed by [SEP] Louis Malle"),
    ]
    M = pairwise_alignment_matrix(A, B, enc)
    print("  matrix:\n", np.round(M, 3))
    assert M.shape == (2, 3)
    assert M[0].argmax() == 0, "Slums-related Q triple은 Slums-D 와 가장 잘 매칭되어야 함"
    print("[ok] pairwise_matrix")


def test_free_matching():
    enc = get_default_encoder()
    A = [
        Triple.from_str("(ENT1) [SEP] is the director of [SEP] film Slums Of Berlin"),
        Triple.from_str("(ENT2) [SEP] is the director of [SEP] film Phantom India"),
    ]
    B = [
        Triple.from_str("Slums of Berlin [SEP] is directed by [SEP] Gerhard Lamprecht"),
        Triple.from_str("Phantom India [SEP] directed by [SEP] Louis Malle"),
        Triple.from_str("Gerhard Lamprecht [SEP] was [SEP] a German film director"),
    ]
    fm = free_matching(A, B, enc)
    print(f"  best_idx={fm['best_idx']}, best_score={np.round(fm['best_score'],3)}")
    print(f"  weighted_score={fm['weighted_score']:.3f}")
    assert fm["weighted_score"] > 0.4
    # 1번 (Phantom India) → B[1] (Phantom India directed by Louis Malle) 매칭 기대
    assert fm["best_idx"][1] == 1
    print("[ok] free_matching")


def test_ppr_and_weight():
    triples = [
        Triple.from_str("Slums of Berlin [SEP] is directed by [SEP] Gerhard Lamprecht"),
        Triple.from_str("Gerhard Lamprecht [SEP] is from [SEP] Germany"),
        Triple.from_str("Phantom India [SEP] is directed by [SEP] Louis Malle"),
        Triple.from_str("Louis Malle [SEP] is from [SEP] France"),
        Triple.from_str("Gerhard Lamprecht [SEP] directed [SEP] 63 films"),
    ]
    G = build_nx_graph(triples)
    print(f"  G: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    ppr = compute_ppr(triples, anchors=["Slums of Berlin", "Phantom India"])
    assert sum(ppr.values()) > 0
    print("  PPR top-5:", sorted(ppr.items(), key=lambda x: -x[1])[:5])
    # Slums of Berlin / Phantom India 가 anchor 시드라서 점수가 높아야 함
    sob = ppr["slums of berlin"]
    pi = ppr["phantom india"]
    fr = ppr.get("france", 0.0)
    assert sob > fr and pi > fr

    # weight: edge 양쪽이 anchor 인 첫 triple은 weight 높아야 함
    w_sob = triple_weight(triples[0], ppr)
    w_pi = triple_weight(triples[2], ppr)
    w_fr = triple_weight(triples[3], ppr)
    print(f"  w(Slums-Lamprecht)={w_sob:.4f}, w(Phantom-Malle)={w_pi:.4f}, w(Malle-France)={w_fr:.4f}")
    assert w_sob > w_fr
    print("[ok] ppr_and_weight")


def test_propagation_consistency():
    s0 = GraphStep(0, "step0", [
        Triple.from_str("Slums of Berlin [SEP] directed by [SEP] (ENT1)"),
        Triple.from_str("Phantom India [SEP] directed by [SEP] (ENT2)"),
    ])
    s1 = GraphStep(1, "step1", [
        Triple.from_str("Gerhard Lamprecht [SEP] directed [SEP] Slums of Berlin"),
    ])
    s2 = GraphStep(2, "step2", [
        Triple.from_str("Louis Malle [SEP] directed [SEP] Phantom India"),
        Triple.from_str("Louis Malle [SEP] is from [SEP] France"),
    ])
    s3 = GraphStep(3, "step3", [
        Triple.from_str("Gerhard Lamprecht [SEP] is from [SEP] Germany"),
    ])

    pc = propagation_consistency([s0, s1, s2, s3])
    print("  PC per step:", [round(x, 3) for x in pc["pc_per_step"]])
    print("  PC total   :", round(pc["pc_total"], 3))
    assert 0.0 < pc["pc_total"] <= 1.0

    # step만 1개 / 0개면 PC=1.0
    assert propagation_consistency([])["pc_total"] == 1.0
    assert propagation_consistency([s0])["pc_total"] == 1.0
    print("[ok] propagation_consistency")


def test_tasi_integration():
    enc = get_default_encoder()
    # Q
    Q = [
        Triple.from_str("(ENT1) [SEP] is the director of [SEP] film Slums Of Berlin"),
        Triple.from_str("(ENT2) [SEP] is the director of [SEP] film Phantom India"),
        Triple.from_str("(ENT1) [SEP] is from [SEP] (ENT3)"),
        Triple.from_str("(ENT2) [SEP] is from [SEP] (ENT4)"),
    ]
    # D (관련 doc triples)
    D = [
        Triple.from_str("Slums of Berlin [SEP] is directed by [SEP] Gerhard Lamprecht"),
        Triple.from_str("Gerhard Lamprecht [SEP] was [SEP] a German film director"),
        Triple.from_str("Phantom India [SEP] is directed by [SEP] Louis Malle"),
        Triple.from_str("Louis Malle [SEP] was [SEP] a French film director"),
        Triple.from_str("My Dinner with Andre [SEP] is [SEP] a comedy-drama"),
    ]
    # D2 (전혀 관련 없는 doc)
    D_bad = [
        Triple.from_str("Eiffel Tower [SEP] is in [SEP] Paris"),
        Triple.from_str("Mount Everest [SEP] is in [SEP] Nepal"),
    ]

    r1 = tasi(Q, D, enc)
    r2 = tasi(Q, D_bad, enc)
    print(f"  TASI(Q, D_good) = {r1.tasi:.3f} (wa={r1.wa:.3f}, pc={r1.pc:.3f})")
    print(f"  TASI(Q, D_bad)  = {r2.tasi:.3f} (wa={r2.wa:.3f}, pc={r2.pc:.3f})")
    assert r1.tasi > r2.tasi, "관련 doc이 비관련 doc보다 점수가 높아야 함"

    # steps 적용 시 PC 영향 확인
    steps = [
        GraphStep(0, "", [Q[0], Q[1]]),  # 두 director 슬롯 정의
        GraphStep(1, "", [Q[2], Q[3]]),  # 두 country 슬롯 정의 (UNKNOWN ENT1/ENT2 사용)
    ]
    r3 = tasi(Q, D, enc, steps=steps)
    print(f"  TASI(Q, D_good, steps) = {r3.tasi:.3f} (pc={r3.pc:.3f})")
    print("[ok] tasi_integration")


def main():
    test_schema_unknown()
    test_align_basic()
    test_align_unknown()
    test_pairwise_matrix()
    test_free_matching()
    test_ppr_and_weight()
    test_propagation_consistency()
    test_tasi_integration()
    print("\n=== all tests passed ===")


if __name__ == "__main__":
    main()
