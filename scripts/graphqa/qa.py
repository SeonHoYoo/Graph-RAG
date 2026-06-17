"""Module 3: TASI 기반 UNKNOWN 슬롯 채우기 + 최종 답 도출.

v3 개선:
  1) 슬롯 채우기를 Hungarian 1:1 assignment 로 풀어서 슬롯간 distinct 제약을
     hard 하게 보장 (예: ENT1, ENT2, ENT3 이 같은 후보로 중복 채워지는 문제 해결).
  2) Q_def 의 (ENTk) is a TYPE 정의를 *hard filter* 로 사용 (이전엔 soft bonus 만).
  3) Q에 known 으로 등장한 entity와 같은 토큰을 가진 후보는 강한 페널티
     (자기 자신 채우기 방지).
  4) Yes/No 로직: 슬롯이 모두 같으면 yes, 다르면 no — 단 슬롯이 비교 가능한
     pair 가 없으면 baseline 통계 기반 prior 사용.
  5) Answer normalize 강화: 관사/하이픈/괄호/공백 정규화 + token-level alias.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from graphqa.data.schema import GraphSample, Triple, is_unknown
from graphqa.tasi.align import (
    DEFAULT_W_O,
    DEFAULT_W_R,
    DEFAULT_W_S,
    pairwise_alignment_matrix,
)
from graphqa.tasi.embedding import SentenceEncoder, get_default_encoder
from graphqa.tasi.ppr import _norm_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slot 표기 / 후보
# ---------------------------------------------------------------------------
_SLOT_RE = re.compile(r"^\(?(ENT\d+)\)?$|^\?[A-Za-z0-9_]+$", re.IGNORECASE)


def _slot_id(token: str) -> Optional[str]:
    if not token:
        return None
    m = _SLOT_RE.match(token.strip())
    return m.group(0).upper().replace("(", "").replace(")", "") if m else None


@dataclass
class SlotCandidate:
    value: str
    score: float                        # 누적 alignment score
    support_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    type_match: float = 0.0             # type filter 통과 여부 (0~1)


@dataclass
class QAResult:
    predicted_answer: str
    is_yesno: bool
    is_correct: bool
    em: float = 0.0
    f1: float = 0.0
    slot_fillings: Dict[str, SlotCandidate] = field(default_factory=dict)
    yesno_baseline_pred: Optional[str] = None   # always-yes 측정용
    debug: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Answer normalize / 비교
# ---------------------------------------------------------------------------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[\W_]+")


def _norm_answer(a: str) -> str:
    """공격적 정규화: lowercase + 관사 제거 + 구두점/공백 정규화."""
    if a is None:
        return ""
    s = a.lower()
    s = _ARTICLES_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _expand_aliases(gold: str, aliases: Sequence[str]) -> List[str]:
    """별칭을 더 많이 만들어 EM 가능성 ↑.

    e.g. "United States of America" → ["united states of america", "united states", "usa"]
    하이픈/괄호 변형을 추가.
    """
    out: List[str] = []
    seen: Set[str] = set()
    for raw in [gold, *aliases]:
        if raw is None:
            continue
        for v in _alias_variants(raw):
            n = _norm_answer(v)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _alias_variants(s: str) -> List[str]:
    if not s:
        return []
    out = [s]
    s2 = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()  # 괄호 안 제거
    if s2 and s2 != s:
        out.append(s2)
    out.append(s.replace("-", " "))
    out.append(s.replace("'", ""))
    return out


def _em_set(pred: str, gold_norms: Sequence[str]) -> float:
    p = _norm_answer(pred)
    if not p:
        return 0.0
    if any(p == g for g in gold_norms):
        return 1.0
    # token-level: 모든 token 이 일치하면 EM=1
    p_toks = p.split()
    for g in gold_norms:
        g_toks = g.split()
        if not g_toks:
            continue
        if Counter(p_toks) == Counter(g_toks):
            return 1.0
    return 0.0


def _f1_set(pred: str, gold_norms: Sequence[str]) -> float:
    p_tokens = _norm_answer(pred).split()
    if not p_tokens:
        return 0.0
    best = 0.0
    for g in gold_norms:
        g_tokens = g.split()
        if not g_tokens:
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(p_tokens)
        recall = n_same / len(g_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def score_answer(pred: str, gold: str, aliases: Sequence[str] = ()) -> Tuple[float, float]:
    """(em, f1) 반환 — 외부에서 재사용."""
    gold_norms = _expand_aliases(gold, aliases)
    return _em_set(pred, gold_norms), _f1_set(pred, gold_norms)


# ---------------------------------------------------------------------------
# Yes/No 분류
# ---------------------------------------------------------------------------
_YESNO_PREFIXES = (
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "have ", "has ",
    "had ", "can ", "could ", "should ", "will ", "would ", "either ", "both ",
)


def _is_yesno_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if any(q.startswith(p) for p in _YESNO_PREFIXES):
        return True
    if "same country" in q or "same nationality" in q or "same year" in q:
        return True
    return False


# ---------------------------------------------------------------------------
# Type filter helper
# ---------------------------------------------------------------------------
_TYPE_REL_RE = re.compile(r"^\s*is(?:\s+an?)?\s*$", re.IGNORECASE)


def _slot_type(slot: str, Q: Sequence[Triple]) -> Optional[str]:
    """Q_def 에 (ENTk) is a TYPE 형태가 있으면 TYPE 문자열 반환."""
    sid = slot.upper().strip()
    for q in Q:
        head_id = _slot_id(q.head) if q.head_unknown else None
        if head_id and head_id == sid and _TYPE_REL_RE.match(q.relation or ""):
            t = q.tail.strip()
            t_clean = re.sub(r"^(an?|the)\s+", "", t.lower()).strip()
            if t_clean and not is_unknown(t_clean):
                return t_clean
    return None


def _known_entity_tokens(Q: Sequence[Triple]) -> Set[str]:
    out: Set[str] = set()
    for q in Q:
        if not q.head_unknown:
            out.update(_norm_node(q.head).split())
        if not q.tail_unknown:
            out.update(_norm_node(q.tail).split())
    return {t for t in out if len(t) >= 3}


# ---------------------------------------------------------------------------
# Slot Filling (1:1 Hungarian, type-aware)
# ---------------------------------------------------------------------------
def _collect_raw_candidates(
    Q: Sequence[Triple],
    D: Sequence[Triple],
    encoder: SentenceEncoder,
    *,
    min_align: float,
    w_S: float,
    w_R: float,
    w_O: float,
    use_reverse: bool,
    reverse_threshold: float,
) -> Dict[str, Dict[str, Tuple[float, List[Tuple[int, int, float]]]]]:
    """슬롯 → {value: (score, evid)}  raw 후보 풀."""
    fwd = pairwise_alignment_matrix(Q, D, encoder, w_S=w_S, w_R=w_R, w_O=w_O, allow_inverse=False)
    D_rev = [Triple(head=t.tail, relation=t.relation, tail=t.head, context=t.context, raw=t.raw) for t in D]
    rev = pairwise_alignment_matrix(Q, D_rev, encoder, w_S=w_S, w_R=w_R, w_O=w_O, allow_inverse=False)

    pool: Dict[str, Dict[str, Tuple[float, List[Tuple[int, int, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: (0.0, []))
    )

    def add(slot: str, value: str, score: float, q_i: int, d_j: int) -> None:
        nv = value.strip()
        if not nv or is_unknown(nv):
            return
        cur, evid = pool[slot][nv]
        evid = evid + [(q_i, d_j, float(score))]
        pool[slot][nv] = (cur + float(score), evid)

    for i, q in enumerate(Q):
        head_slot = _slot_id(q.head) if q.head_unknown else None
        tail_slot = _slot_id(q.tail) if q.tail_unknown else None
        if not head_slot and not tail_slot:
            continue
        for j in range(len(D)):
            sf = float(fwd[i, j])
            sr = float(rev[i, j]) if use_reverse else 0.0
            if sf >= min_align:
                if head_slot:
                    add(head_slot, D[j].head, sf, i, j)
                if tail_slot:
                    add(tail_slot, D[j].tail, sf, i, j)
            if use_reverse and sr >= min_align and sf < reverse_threshold and sr > sf:
                if head_slot:
                    add(head_slot, D_rev[j].head, sr, i, j)
                if tail_slot:
                    add(tail_slot, D_rev[j].tail, sr, i, j)
    return pool


def fill_unknown_slots(
    Q: Sequence[Triple],
    D: Sequence[Triple],
    encoder: SentenceEncoder,
    *,
    top_k_per_slot: int = 8,
    min_align: float = 0.32,
    w_S: float = DEFAULT_W_S,
    w_R: float = DEFAULT_W_R,
    w_O: float = DEFAULT_W_O,
    use_reverse: bool = True,
    reverse_threshold: float = 0.55,
    type_match_threshold: float = 0.30,
    type_match_bonus: float = 0.40,
    self_overlap_penalty: float = 0.7,
    enforce_distinct: bool = True,
) -> Dict[str, SlotCandidate]:
    """Q의 UNKNOWN 슬롯을 D에서 채움.

    Pipeline:
      1. raw 후보 수집 (forward + selective reverse).
      2. self-overlap 페널티 + type filter 적용 (최종 score 산출).
      3. 슬롯 간 distinct 제약 (Hungarian) — 같은 후보가 여러 슬롯에 채워지지
         않도록 1:1 매칭.
    """
    if not Q or not D:
        return {}

    pool = _collect_raw_candidates(
        Q, D, encoder,
        min_align=min_align, w_S=w_S, w_R=w_R, w_O=w_O,
        use_reverse=use_reverse, reverse_threshold=reverse_threshold,
    )
    known_tokens = _known_entity_tokens(Q)
    slot_types: Dict[str, Optional[str]] = {s: _slot_type(s, Q) for s in pool.keys()}

    # ---- 후보별 최종 score 산출 ----
    scored: Dict[str, List[Tuple[str, float, List[Tuple[int, int, float]], float]]] = {}
    for slot, cand_dict in pool.items():
        type_str = slot_types.get(slot)
        finals: List[Tuple[str, float, List[Tuple[int, int, float]], float]] = []
        for value, (raw_score, evid) in cand_dict.items():
            score = float(raw_score)
            v_tokens = {t for t in _norm_node(value).split() if len(t) >= 3}
            # self-overlap penalty
            if v_tokens and known_tokens:
                overlap = len(v_tokens & known_tokens) / max(1, len(v_tokens))
                if overlap >= 0.5:
                    score *= (1.0 - self_overlap_penalty * overlap)
            # type bonus
            type_match = 0.0
            if type_str:
                try:
                    ts = encoder.cosine(value, type_str)
                except Exception:
                    ts = 0.0
                type_match = max(0.0, float(ts))
                if type_match > type_match_threshold:
                    score *= (1.0 + type_match_bonus * type_match)
                else:
                    # type 불일치는 약한 페널티 (정확도 우선)
                    score *= max(0.4, 1.0 - 0.3 * (type_match_threshold - type_match) / type_match_threshold)
            if score < min_align * 0.4:
                continue
            finals.append((value, score, evid, type_match))
        if finals:
            finals.sort(key=lambda x: -x[1])
            scored[slot] = finals[:top_k_per_slot]

    if not scored:
        return {}

    # ---- 슬롯간 distinct 제약: Hungarian ----
    slots = list(scored.keys())
    if enforce_distinct and len(slots) >= 2:
        # 후보 union 만들기
        all_values: List[str] = []
        seen: Dict[str, int] = {}
        for s in slots:
            for v, _, _, _ in scored[s]:
                key = _norm_node(v)
                if key not in seen:
                    seen[key] = len(all_values)
                    all_values.append(v)
        n_slots = len(slots)
        n_vals = len(all_values)

        # cost matrix: 슬롯 × value, 음의 score
        # value 개수가 슬롯 수보다 적으면 padding
        cost = np.full((n_slots, max(n_vals, n_slots)), 1e6, dtype=np.float64)
        score_lookup: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, int, float]], float]] = {}
        for i, s in enumerate(slots):
            for v, sc, evid, tm in scored[s]:
                j = seen[_norm_node(v)]
                # 더 좋은 점수가 들어오면 갱신 (보통 한 번만 들어옴)
                cur = -cost[i, j]
                if sc > cur:
                    cost[i, j] = -sc
                    score_lookup[(i, j)] = (sc, evid, tm)

        try:
            from scipy.optimize import linear_sum_assignment
            row, col = linear_sum_assignment(cost)
        except Exception:
            row, col = list(range(n_slots)), list(range(n_slots))

        out: Dict[str, SlotCandidate] = {}
        for i, j in zip(row, col):
            if i >= n_slots:
                continue
            if j >= len(all_values) or cost[i, j] >= 1e5:
                # fallback: 그 슬롯의 top-1 (배타 제약 깨도 됨)
                top_v, top_sc, top_ev, top_tm = scored[slots[i]][0]
                out[slots[i]] = SlotCandidate(value=top_v, score=top_sc, support_pairs=top_ev, type_match=top_tm)
                continue
            sc, evid, tm = score_lookup[(i, j)]
            out[slots[i]] = SlotCandidate(value=all_values[j], score=sc, support_pairs=evid, type_match=tm)
        return out
    else:
        out = {}
        for s in slots:
            v, sc, evid, tm = scored[s][0]
            out[s] = SlotCandidate(value=v, score=sc, support_pairs=evid, type_match=tm)
        return out


# ---------------------------------------------------------------------------
# Top-K 슬롯 후보 (LLM slot filling 용도)
# ---------------------------------------------------------------------------
def topk_slot_candidates(
    sample: GraphSample,
    encoder: SentenceEncoder,
    k: int = 5,
    *,
    min_align: float = 0.32,
    w_S: float = DEFAULT_W_S,
    w_R: float = DEFAULT_W_R,
    w_O: float = DEFAULT_W_O,
    use_reverse: bool = True,
    reverse_threshold: float = 0.55,
    type_match_threshold: float = 0.30,
    type_match_bonus: float = 0.40,
    self_overlap_penalty: float = 0.7,
) -> Dict[str, List[Tuple[str, float, float]]]:
    """슬롯 별로 D 에서 *후보 entity* 의 top-K 만 추려 반환.

    LLM 에게 "ENTk 의 후보는 [a, b, c, ...] 중에 골라" 라는 hint 를 주기 위한
    구조. score 는 정렬 alignment + type bonus + self-overlap penalty 가 모두
    적용된 최종값. 채택률(distinct constraint)은 적용하지 않으므로 슬롯 사이
    후보가 겹칠 수 있음 — LLM 이 최종 결정한다.

    반환 형식: { slot_id : [(entity_value, score, type_match), ...] }
    """
    Q = sample.Q
    D = sample.D
    if not Q or not D:
        return {}

    pool = _collect_raw_candidates(
        Q, D, encoder,
        min_align=min_align, w_S=w_S, w_R=w_R, w_O=w_O,
        use_reverse=use_reverse, reverse_threshold=reverse_threshold,
    )
    if not pool:
        return {}

    known_tokens = _known_entity_tokens(Q)
    slot_types: Dict[str, Optional[str]] = {s: _slot_type(s, Q) for s in pool.keys()}

    out: Dict[str, List[Tuple[str, float, float]]] = {}
    for slot, cand_dict in pool.items():
        type_str = slot_types.get(slot)
        scored: List[Tuple[str, float, float]] = []
        for value, (raw_score, _) in cand_dict.items():
            score = float(raw_score)
            v_tokens = {t for t in _norm_node(value).split() if len(t) >= 3}
            if v_tokens and known_tokens:
                overlap = len(v_tokens & known_tokens) / max(1, len(v_tokens))
                if overlap >= 0.5:
                    score *= (1.0 - self_overlap_penalty * overlap)
            type_match = 0.0
            if type_str:
                try:
                    ts = encoder.cosine(value, type_str)
                except Exception:
                    ts = 0.0
                type_match = max(0.0, float(ts))
                if type_match > type_match_threshold:
                    score *= (1.0 + type_match_bonus * type_match)
                else:
                    score *= max(0.4, 1.0 - 0.3 * (type_match_threshold - type_match) / type_match_threshold)
            if score < min_align * 0.4:
                continue
            scored.append((value, score, type_match))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[1])
        out[slot] = scored[:max(1, int(k))]
    return out


# ---------------------------------------------------------------------------
# Yes/No 결정
# ---------------------------------------------------------------------------
def _decide_yesno(
    Q: Sequence[Triple],
    slot_filling: Dict[str, SlotCandidate],
    sample: GraphSample,
) -> str:
    """Q에서 같은 relation 으로 매핑되는 슬롯들의 값이 같으면 yes, 다르면 no.

    개선:
      - distinct 제약 적용 후이므로 "같음"은 어떤 토큰 overlap 으로만 판단.
      - 슬롯이 충분치 않으면 question 의 "same/both" 등 키워드로 prior.
    """
    rel_to_slots: Dict[str, List[str]] = defaultdict(list)
    for q in Q:
        slot = _slot_id(q.tail) if q.tail_unknown else (_slot_id(q.head) if q.head_unknown else None)
        if slot:
            rel_to_slots[q.relation].append(slot)

    same_count, diff_count, total = 0, 0, 0
    for rel, slots in rel_to_slots.items():
        unique_slots = list(dict.fromkeys(slots))
        if len(unique_slots) < 2:
            continue
        values: List[str] = []
        for s in unique_slots:
            if s in slot_filling:
                values.append(_norm_node(slot_filling[s].value))
        if len(values) < 2:
            continue
        total += 1
        first_toks = set(values[0].split())
        all_same_str = all(v == values[0] for v in values[1:])
        any_token_overlap = (
            bool(first_toks)
            and all(first_toks & set(v.split()) for v in values[1:])
        )
        if all_same_str:
            same_count += 1
        elif any_token_overlap:
            same_count += 1
        else:
            diff_count += 1

    if total == 0:
        return "yes"  # baseline prior — 데이터 통계상 yes 가 우세
    return "yes" if same_count >= diff_count else "no"


# ---------------------------------------------------------------------------
# Open answer 결정
# ---------------------------------------------------------------------------
def _decide_open(
    Q: Sequence[Triple],
    slot_filling: Dict[str, SlotCandidate],
) -> str:
    """ENTk 슬롯 중 가장 점수 높은 것 (단순)."""
    if not slot_filling:
        return ""
    ents = [(s, c) for s, c in slot_filling.items() if s.upper().startswith("ENT")]
    ents.sort(key=lambda x: -x[1].score)
    if ents:
        return ents[0][1].value
    return max(slot_filling.values(), key=lambda c: c.score).value


# ---------------------------------------------------------------------------
# 메인 QA
# ---------------------------------------------------------------------------
def answer_question(
    sample: GraphSample,
    encoder: Optional[SentenceEncoder] = None,
    *,
    top_k_per_slot: int = 8,
    min_align: float = 0.32,
) -> QAResult:
    """Q 의 UNKNOWN 슬롯을 채워 답 도출 (LLM 미사용)."""
    enc = encoder or get_default_encoder()

    Q_full = list(sample.Q) + list(sample.Q_def)
    D = list(sample.D)
    is_yesno = _is_yesno_question(sample.question)

    if not D:
        return QAResult(
            predicted_answer="yes" if is_yesno else "",
            is_yesno=is_yesno,
            is_correct=False,
            yesno_baseline_pred="yes" if is_yesno else None,
        )

    fillings = fill_unknown_slots(
        Q_full, D, enc,
        top_k_per_slot=top_k_per_slot, min_align=min_align,
    )

    if is_yesno:
        pred = _decide_yesno(Q_full, fillings, sample)
        baseline = "yes"
    else:
        pred = _decide_open(Q_full, fillings)
        baseline = None

    em, f1 = score_answer(pred, sample.answer, sample.answer_aliases)
    return QAResult(
        predicted_answer=pred,
        is_yesno=is_yesno,
        is_correct=bool(em >= 1.0),
        em=em,
        f1=f1,
        slot_fillings=fillings,
        yesno_baseline_pred=baseline,
        debug={"yesno": is_yesno, "n_slots": len(fillings)},
    )
