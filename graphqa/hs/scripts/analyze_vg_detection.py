#!/usr/bin/env python
"""Join raw case-judge results with saved VG labels/hints.

This is analysis-only. It answers a different question from the LLM raw judge:

- raw case judge: does the final trajectory look evidence-supported?
- this script: did the saved VeriGraph labels/hints flag an issue in that case?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


STRICT_DT_ISSUES = {"unsupported_think_claim", "conflict"}
STRICT_ACTION_ISSUES = {"do_not_carry_forward", "revise_conflict"}
RETRIEVAL_QD_ISSUES = {"no_support", "weak_or_irrelevant_doc"}
RETRIEVAL_ACTION_ISSUES = {"search_exact_title_relation"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True, help="VG case JSON with turn_records.")
    p.add_argument("--case-judge", required=True, help="Raw case judge JSON output.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    return p.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _labels(case: Dict[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for turn in case.get("turn_records", []) or []:
        if not isinstance(turn, dict):
            continue
        turn_no = int(turn.get("turn", 0))
        for lab in turn.get("verification_labels", []) or []:
            if isinstance(lab, dict):
                yield turn_no, lab


def _feedback_turns(case: Dict[str, Any]) -> List[int]:
    out = []
    for turn in case.get("turn_records", []) or []:
        if isinstance(turn, dict) and str(turn.get("injected_feedback") or "").strip():
            out.append(int(turn.get("turn", 0)))
    return out


def _turn_numbers(case: Dict[str, Any]) -> List[int]:
    turns: List[int] = []
    for i, turn in enumerate(case.get("turn_records", []) or []):
        if isinstance(turn, dict):
            turns.append(int(turn.get("turn", i)))
    return turns


def detect_vg(case: Dict[str, Any]) -> Dict[str, Any]:
    strict_labels: List[Dict[str, Any]] = []
    retrieval_labels: List[Dict[str, Any]] = []
    all_issue_labels: List[Dict[str, Any]] = []

    issue_turns: List[int] = []
    strict_issue_turns: List[int] = []
    retrieval_issue_turns: List[int] = []

    for turn_no, lab in _labels(case):
        d_t = str(lab.get("d_t") or "")
        action = str(lab.get("action") or "")
        q_d = str(lab.get("q_d") or "")
        q_t = str(lab.get("q_t") or "")

        row = {
            "turn": turn_no,
            "rid": lab.get("rid"),
            "requirement": lab.get("requirement"),
            "q_d": q_d,
            "q_t": q_t,
            "d_t": d_t,
            "action": action,
            "doc_value": lab.get("doc_value"),
            "think_value": lab.get("think_value"),
            "suggested_query": lab.get("suggested_query"),
        }

        is_strict = d_t in STRICT_DT_ISSUES or action in STRICT_ACTION_ISSUES
        is_retrieval = q_d in RETRIEVAL_QD_ISSUES or action in RETRIEVAL_ACTION_ISSUES
        if is_strict:
            strict_labels.append(row)
            all_issue_labels.append(row)
            issue_turns.append(turn_no)
            strict_issue_turns.append(turn_no)
        elif is_retrieval:
            retrieval_labels.append(row)
            all_issue_labels.append(row)
            issue_turns.append(turn_no)
            retrieval_issue_turns.append(turn_no)
        elif q_d in RETRIEVAL_QD_ISSUES and q_t == "claims_value":
            retrieval_labels.append(row)
            all_issue_labels.append(row)
            issue_turns.append(turn_no)
            retrieval_issue_turns.append(turn_no)

    feedback_turns = _feedback_turns(case)
    turns = _turn_numbers(case)
    final_turn = max(turns) if turns else None
    final_window_start = max(0, final_turn - 1) if final_turn is not None else None
    issue_turn_set = set(issue_turns)
    strict_turn_set = set(strict_issue_turns)
    retrieval_turn_set = set(retrieval_issue_turns)
    if final_turn is None:
        early_issue = False
        final_unresolved = False
        final_strict_unresolved = False
        final_retrieval_unresolved = False
    else:
        early_issue = any(t < final_turn for t in issue_turn_set)
        final_unresolved = any(t >= final_window_start for t in issue_turn_set)
        final_strict_unresolved = any(t >= final_window_start for t in strict_turn_set)
        final_retrieval_unresolved = any(t >= final_window_start for t in retrieval_turn_set)

    return {
        "vg_detected_strict": bool(strict_labels),
        "vg_detected_retrieval": bool(retrieval_labels),
        "vg_detected_any_label_issue": bool(all_issue_labels),
        "vg_early_issue": early_issue,
        "vg_final_unresolved": final_unresolved,
        "vg_final_strict_unresolved": final_strict_unresolved,
        "vg_final_retrieval_unresolved": final_retrieval_unresolved,
        "first_issue_turn": min(issue_turn_set) if issue_turn_set else None,
        "last_issue_turn": max(issue_turn_set) if issue_turn_set else None,
        "final_turn": final_turn,
        "n_issue_turns": len(issue_turn_set),
        "issue_turns": sorted(issue_turn_set),
        "strict_issue_turns": sorted(strict_turn_set),
        "retrieval_issue_turns": sorted(retrieval_turn_set),
        "vg_feedback_injected": bool(feedback_turns),
        "n_feedback_turns": len(feedback_turns),
        "feedback_turns": feedback_turns,
        "strict_issue_labels": strict_labels[:20],
        "retrieval_issue_labels": retrieval_labels[:20],
        "n_strict_issue_labels": len(strict_labels),
        "n_retrieval_issue_labels": len(retrieval_labels),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        cat = str(row.get("category") or "unknown")
        by_category[cat]["n"] += 1
        for key in [
            "vg_detected_strict",
            "vg_detected_retrieval",
            "vg_detected_any_label_issue",
            "vg_early_issue",
            "vg_final_unresolved",
            "vg_final_strict_unresolved",
            "vg_final_retrieval_unresolved",
            "vg_recovered_success_candidate",
            "vg_feedback_injected",
        ]:
            if row.get(key):
                by_category[cat][key] += 1

    def pack(counter: Counter) -> Dict[str, Any]:
        n = counter.get("n", 0)
        out = dict(counter)
        for key in [
            "vg_detected_strict",
            "vg_detected_retrieval",
            "vg_detected_any_label_issue",
            "vg_early_issue",
            "vg_final_unresolved",
            "vg_final_strict_unresolved",
            "vg_final_retrieval_unresolved",
            "vg_recovered_success_candidate",
            "vg_feedback_injected",
        ]:
            out[f"{key}_rate"] = counter.get(key, 0) / n if n else 0.0
        return out

    totals = Counter()
    for row in rows:
        totals["n"] += 1
        for key in [
            "vg_detected_strict",
            "vg_detected_retrieval",
            "vg_detected_any_label_issue",
            "vg_early_issue",
            "vg_final_unresolved",
            "vg_final_strict_unresolved",
            "vg_final_retrieval_unresolved",
            "vg_recovered_success_candidate",
            "vg_feedback_injected",
        ]:
            if row.get(key):
                totals[key] += 1
    return {
        "n_cases": len(rows),
        "by_category": {cat: pack(cnt) for cat, cnt in sorted(by_category.items())},
        "overall": pack(totals),
    }


def main() -> None:
    args = parse_args()
    cases = load_json(args.cases)
    judge_rows = load_json(args.case_judge)
    if not isinstance(cases, list) or not isinstance(judge_rows, list):
        raise ValueError("--cases and --case-judge must both be JSON lists")

    case_by_uid = {str(c.get("uid")): c for c in cases if isinstance(c, dict)}
    output_rows: List[Dict[str, Any]] = []
    missing = 0
    for judge in judge_rows:
        if not isinstance(judge, dict):
            continue
        uid = str(judge.get("uid"))
        case = case_by_uid.get(uid)
        if case is None:
            missing += 1
            continue
        det = detect_vg(case)
        recovered_success = (
            judge.get("category") == "true_success"
            and det.get("vg_early_issue") is True
            and det.get("vg_final_unresolved") is False
        )
        output_rows.append({
            "dataset": judge.get("dataset") or case.get("dataset"),
            "uid": uid,
            "question": judge.get("question") or case.get("question"),
            "predicted_answer": judge.get("predicted_answer") or case.get("predicted_answer"),
            "gold_answer": judge.get("gold_answer") or case.get("answer"),
            "answer_correct": judge.get("answer_correct"),
            "category": judge.get("category"),
            "final_requirements_supported": judge.get("final_requirements_supported"),
            "evidence_issue_found": judge.get("evidence_issue_found"),
            "vg_recovered_success_candidate": recovered_success,
            **det,
        })

    result = {
        "cases_path": str(args.cases),
        "case_judge_path": str(args.case_judge),
        "n_judge_rows": len(judge_rows),
        "n_matched_rows": len(output_rows),
        "n_missing_case_rows": missing,
        "summary": summarize(output_rows),
        "cases": output_rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"[vg_detection] wrote {out}")


if __name__ == "__main__":
    main()
