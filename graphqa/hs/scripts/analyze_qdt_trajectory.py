#!/usr/bin/env python
"""Flatten HS online-feedback cases into Q/D/T trajectory verification edges.

This script is analysis-only. It does not rerun SearchR1 and does not inject
feedback. The goal is to make automatic VeriGraph matches easy to inspect:
each output row is one edge such as Q-D, Q-T, D-T, query-Q, or trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EDGE_FIELDS = [
    "dataset",
    "uid",
    "case_index",
    "turn",
    "edge_type",
    "auto_label",
    "score",
    "threshold",
    "question",
    "gold_answer",
    "predicted_answer",
    "query",
    "q_idx",
    "q_text",
    "q_head",
    "q_relation",
    "q_tail",
    "doc_status",
    "think_status",
    "doc_candidate",
    "think_candidate",
    "doc_slot_value",
    "think_slot_value",
    "raw_evidence",
    "think_text",
    "feedback",
    "manual_label",
    "manual_note",
]


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    return [x for x in data if isinstance(x, dict)]


def _norm(value: Any) -> str:
    return " ".join(str(value or "").replace('"', "").split())


def _norm_l(value: Any) -> str:
    return _norm(value).lower()


def _is_unknown(value: Any) -> bool:
    s = _norm_l(value)
    return (not s) or bool(re.fullmatch(r"\(?ent\d+\)?", s)) or s in {"unknown", "none", "n/a"}


def _strip_known_prefix(value: Any) -> str:
    text = _norm(value)
    low = text.lower()
    for prefix in ("film ", "movie "):
        if low.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _values_match(a: Any, b: Any) -> bool:
    na, nb = _norm_l(a), _norm_l(b)
    return bool(na and nb and (na == nb or na in nb or nb in na))


def _passes(match: Dict[str, Any], threshold: float) -> bool:
    if not match or not match.get("candidate_text"):
        return False
    val = match.get("min_field_cosine")
    try:
        return float(val) >= float(threshold)
    except Exception:
        return False


def _score(match: Dict[str, Any]) -> Optional[float]:
    try:
        v = match.get("min_field_cosine")
        return None if v is None else float(v)
    except Exception:
        return None


def _match_status(match: Dict[str, Any], threshold: float) -> str:
    if not match or not match.get("candidate_text"):
        return "no_candidate"
    return "pass" if _passes(match, threshold) else "fail"


def _answer_for_requirement(row: Dict[str, Any], match: Dict[str, Any]) -> str:
    if not match:
        return ""
    q_head = row.get("q_head") or ""
    q_tail = row.get("q_tail") or ""
    c_head = match.get("candidate_head") or ""
    c_tail = match.get("candidate_tail") or ""
    c_rel = _norm_l(match.get("candidate_relation") or "")

    if _is_unknown(q_head) and q_tail and not _is_unknown(q_tail):
        known = _strip_known_prefix(q_tail)
        if _values_match(known, c_tail):
            return _norm(c_head)
        if _values_match(known, c_head):
            return _norm(c_tail)
        if "directed by" in c_rel:
            return _norm(c_tail)
    if _is_unknown(q_tail) and q_head and not _is_unknown(q_head):
        known = _strip_known_prefix(q_head)
        if _values_match(known, c_head):
            return _norm(c_tail)
        if _values_match(known, c_tail):
            return _norm(c_head)
    return ""


def _answer_from_think_text(row: Dict[str, Any], think_text: str) -> str:
    if not think_text or not _is_unknown(row.get("q_head")):
        return ""
    q_tail = row.get("q_tail") or ""
    q_rel = _norm_l(row.get("q_relation"))
    if not q_tail or _is_unknown(q_tail) or "director" not in q_rel:
        return ""
    title = _strip_known_prefix(q_tail)
    if not title:
        return ""
    title_pat = re.escape(title)
    person_pat = r"([A-Z][A-Za-z .'-]+?)(?=,|\.| and the director| who | Now|$)"
    patterns = [
        rf"director of (?:the film |film )?{title_pat}\s+is\s+{person_pat}",
        rf"{person_pat}\s+is\s+the director of (?:the film |film )?{title_pat}",
    ]
    for pat in patterns:
        m = re.search(pat, think_text, flags=re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g]
            if groups:
                ans = _norm(groups[-1])
                if ans and not _is_unknown(ans) and not _values_match(ans, title):
                    return ans
    return ""


def _anchor_terms(row: Dict[str, Any]) -> List[str]:
    anchor = row.get("q_tail") if not _is_unknown(row.get("q_tail")) else row.get("q_head")
    anchor = _strip_known_prefix(anchor or "")
    return [w for w in re.findall(r"[A-Za-z0-9]+", anchor.lower()) if len(w) > 2]


def _query_label(row: Dict[str, Any], query: str, threshold: float) -> Tuple[str, str]:
    d = row.get("doc_match") or {}
    if _passes(d, threshold):
        return "already_supported", ""
    terms = _anchor_terms(row)
    q_l = _norm_l(query)
    if terms and all(t in q_l for t in terms):
        return "targets_unresolved_requirement", ""
    if terms and any(t in q_l for t in terms):
        return "partially_targets_requirement", ""
    return "query_may_miss_requirement", ""


def _base_row(
    *,
    case: Dict[str, Any],
    case_index: int,
    turn: Dict[str, Any],
    edge_type: str,
    threshold: float,
    align_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r = align_row or {}
    return {
        "dataset": case.get("dataset", ""),
        "uid": case.get("uid", ""),
        "case_index": case_index,
        "turn": turn.get("turn", ""),
        "edge_type": edge_type,
        "auto_label": "",
        "score": "",
        "threshold": threshold,
        "question": case.get("question", ""),
        "gold_answer": case.get("answer", ""),
        "predicted_answer": case.get("predicted_answer", ""),
        "query": turn.get("query", ""),
        "q_idx": r.get("q_idx", ""),
        "q_text": r.get("q_text", ""),
        "q_head": r.get("q_head", ""),
        "q_relation": r.get("q_relation", ""),
        "q_tail": r.get("q_tail", ""),
        "doc_status": "",
        "think_status": "",
        "doc_candidate": "",
        "think_candidate": "",
        "doc_slot_value": "",
        "think_slot_value": "",
        "raw_evidence": "",
        "think_text": turn.get("think_text", ""),
        "feedback": turn.get("injected_feedback", ""),
        "manual_label": "",
        "manual_note": "",
    }


def _alignment_edges(
    *,
    case: Dict[str, Any],
    case_index: int,
    turn: Dict[str, Any],
    threshold: float,
) -> Iterable[Dict[str, Any]]:
    for ar in turn.get("alignment", []) or []:
        d = ar.get("doc_match") or {}
        t = ar.get("think_match") or {}
        doc_pass = _passes(d, threshold)
        think_pass = _passes(t, threshold)
        doc_value = _answer_for_requirement(ar, d)
        think_value = _answer_for_requirement(ar, t) or _answer_from_think_text(ar, str(turn.get("think_text") or ""))

        qd = _base_row(case=case, case_index=case_index, turn=turn, edge_type="Q-D", threshold=threshold, align_row=ar)
        qd.update({
            "auto_label": "support" if doc_pass else "no_support",
            "score": _score(d) if _score(d) is not None else "",
            "doc_status": _match_status(d, threshold),
            "doc_candidate": d.get("candidate_text", ""),
            "doc_slot_value": doc_value,
            "raw_evidence": d.get("candidate_text", ""),
        })
        yield qd

        qt = _base_row(case=case, case_index=case_index, turn=turn, edge_type="Q-T", threshold=threshold, align_row=ar)
        qt.update({
            "auto_label": "relevant_or_asserted" if think_pass or think_value else "no_relevant_think",
            "score": _score(t) if _score(t) is not None else "",
            "think_status": _match_status(t, threshold),
            "think_candidate": t.get("candidate_text", ""),
            "think_slot_value": think_value,
        })
        yield qt

        dt_label = "missing_both"
        if think_value and not doc_pass:
            dt_label = "unsupported_think_claim"
        elif doc_pass and think_value and doc_value and not _values_match(doc_value, think_value):
            dt_label = "conflict"
        elif doc_pass and think_pass:
            dt_label = "aligned"
        elif doc_pass and not think_pass:
            dt_label = "evidence_not_used"
        elif think_pass and not doc_pass:
            dt_label = "think_only"

        dt = _base_row(case=case, case_index=case_index, turn=turn, edge_type="D-T", threshold=threshold, align_row=ar)
        dt.update({
            "auto_label": dt_label,
            "score": min(x for x in [_score(d), _score(t)] if x is not None) if (_score(d) is not None or _score(t) is not None) else "",
            "doc_status": _match_status(d, threshold),
            "think_status": _match_status(t, threshold),
            "doc_candidate": d.get("candidate_text", ""),
            "think_candidate": t.get("candidate_text", ""),
            "doc_slot_value": doc_value,
            "think_slot_value": think_value,
            "raw_evidence": d.get("candidate_text", ""),
        })
        yield dt

        q_label, _ = _query_label(ar, str(turn.get("query") or ""), threshold)
        qq = _base_row(case=case, case_index=case_index, turn=turn, edge_type="query-Q", threshold=threshold, align_row=ar)
        qq.update({
            "auto_label": q_label,
            "doc_status": _match_status(d, threshold),
            "think_status": _match_status(t, threshold),
            "doc_candidate": d.get("candidate_text", ""),
            "think_candidate": t.get("candidate_text", ""),
            "doc_slot_value": doc_value,
            "think_slot_value": think_value,
        })
        yield qq


def _trajectory_edges(case: Dict[str, Any], case_index: int, threshold: float) -> Iterable[Dict[str, Any]]:
    unsupported: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for turn in case.get("turn_records", []) or []:
        current_text = f"{turn.get('query','')} {turn.get('think_text','')}"
        for key, entry in list(unsupported.items()):
            claim = key[1]
            if claim and claim.lower() in _norm_l(current_text):
                row = _base_row(case=case, case_index=case_index, turn=turn, edge_type="trajectory", threshold=threshold)
                row.update({
                    "auto_label": "repeated_unsupported_claim",
                    "q_idx": entry.get("q_idx", ""),
                    "q_text": entry.get("q_text", ""),
                    "q_head": entry.get("q_head", ""),
                    "q_relation": entry.get("q_relation", ""),
                    "q_tail": entry.get("q_tail", ""),
                    "think_slot_value": claim,
                    "manual_label": "",
                    "manual_note": "",
                })
                yield row

        for ar in turn.get("alignment", []) or []:
            d = ar.get("doc_match") or {}
            t = ar.get("think_match") or {}
            doc_pass = _passes(d, threshold)
            think_value = _answer_for_requirement(ar, t) or _answer_from_think_text(ar, str(turn.get("think_text") or ""))
            doc_value = _answer_for_requirement(ar, d)
            qkey = str(ar.get("q_idx", ""))
            if doc_pass and doc_value:
                unsupported.pop((qkey, _norm_l(doc_value)), None)
                continue
            if think_value and not doc_pass:
                unsupported[(qkey, _norm_l(think_value))] = {
                    "q_idx": ar.get("q_idx", ""),
                    "q_text": ar.get("q_text", ""),
                    "q_head": ar.get("q_head", ""),
                    "q_relation": ar.get("q_relation", ""),
                    "q_tail": ar.get("q_tail", ""),
                }


def build_edges(cases: Sequence[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for turn in case.get("turn_records", []) or []:
            rows.extend(_alignment_edges(case=case, case_index=case_index, turn=turn, threshold=threshold))
        rows.extend(_trajectory_edges(case, case_index, threshold))
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_edge: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    for r in rows:
        e = str(r.get("edge_type") or "")
        l = str(r.get("auto_label") or "")
        by_edge[e] = by_edge.get(e, 0) + 1
        by_label[f"{e}:{l}"] = by_label.get(f"{e}:{l}", 0) + 1
    return {
        "n_edges": len(rows),
        "by_edge_type": by_edge,
        "by_edge_label": by_label,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=0.50)
    args = p.parse_args()

    cases = _load_cases(args.cases)
    rows = build_edges(cases, float(args.threshold))
    _write_csv(args.out_csv, rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(_summary(rows), f, indent=2, ensure_ascii=False)
    print(f"wrote {len(rows)} edges to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
