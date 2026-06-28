#!/usr/bin/env python
"""Compare selective online-feedback configs without an external LLM judge."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


STRICT_DT = {"unsupported_think_claim", "conflict"}
STRICT_ACTION = {"do_not_carry_forward", "revise_conflict"}
RETRIEVAL_QD = {"no_support", "weak_or_irrelevant_doc"}


def _is_strict_issue(label: Dict[str, Any]) -> bool:
    return str(label.get("d_t") or "") in STRICT_DT or str(label.get("action") or "") in STRICT_ACTION


def _is_retrieval_issue(label: Dict[str, Any]) -> bool:
    return str(label.get("q_d") or "") in RETRIEVAL_QD or str(label.get("action") or "") == "search_exact_title_relation"


def _case_metrics(case: Dict[str, Any]) -> Dict[str, Any]:
    turns = list(case.get("turn_records") or [])
    labels = [lab for t in turns for lab in (t.get("verification_labels") or [])]
    feedback_labels = [lab for t in turns for lab in (t.get("feedback_labels") or [])]
    final_labels = list(turns[-1].get("verification_labels") or []) if turns else []
    final_feedback_labels = list(turns[-1].get("feedback_labels") or []) if turns else []
    csv_row = case.get("csv_row") if isinstance(case.get("csv_row"), dict) else {}
    if "answer_matches_gold" in case:
        answer_correct = bool(case.get("answer_matches_gold"))
    elif "em" in csv_row:
        answer_correct = float(csv_row.get("em") or 0.0) >= 1.0
    else:
        answer_correct = str(case.get("predicted_answer") or "").strip().lower() == str(case.get("answer") or "").strip().lower()
    final_strict = any(_is_strict_issue(lab) for lab in final_labels)
    final_retrieval = any(_is_retrieval_issue(lab) for lab in final_labels)
    final_any = final_strict or final_retrieval
    feedback_turns = sum(1 for t in turns if t.get("injected_feedback"))
    suppressed_turns = sum(1 for t in turns if t.get("feedback_suppressed"))
    return {
        "uid": case.get("uid"),
        "question": case.get("question"),
        "gold_answer": case.get("answer"),
        "predicted_answer": case.get("predicted_answer"),
        "answer_correct": answer_correct,
        "num_turns": int(case.get("num_turns") or len(turns)),
        "feedback_turns": feedback_turns,
        "suppressed_turns": suppressed_turns,
        "n_labels": len(labels),
        "n_feedback_labels": len(feedback_labels),
        "final_strict_unresolved": final_strict,
        "final_retrieval_unresolved": final_retrieval,
        "final_any_unresolved": final_any,
        "verified_success_proxy_strict": answer_correct and not final_strict,
        "verified_success_proxy_any": answer_correct and not final_any,
        "suspicious_proxy_strict": answer_correct and final_strict,
        "suspicious_proxy_any": answer_correct and final_any,
        "final_label_summary": Counter(
            f"{lab.get('q_d')}:{lab.get('d_t')}:{lab.get('action')}" for lab in final_labels
        ),
        "final_feedback_label_summary": Counter(
            f"{lab.get('q_d')}:{lab.get('d_t')}:{lab.get('action')}" for lab in final_feedback_labels
        ),
    }


def _rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _summarize(path: pathlib.Path) -> Dict[str, Any]:
    cases = json.load(path.open())
    rows = [_case_metrics(c) for c in cases]
    n = len(rows)
    out: Dict[str, Any] = {
        "path": str(path),
        "n": n,
        "answer_correct": sum(r["answer_correct"] for r in rows),
        "verified_success_proxy_strict": sum(r["verified_success_proxy_strict"] for r in rows),
        "verified_success_proxy_any": sum(r["verified_success_proxy_any"] for r in rows),
        "suspicious_proxy_strict": sum(r["suspicious_proxy_strict"] for r in rows),
        "suspicious_proxy_any": sum(r["suspicious_proxy_any"] for r in rows),
        "final_strict_unresolved": sum(r["final_strict_unresolved"] for r in rows),
        "final_any_unresolved": sum(r["final_any_unresolved"] for r in rows),
        "feedback_turns_mean": round(sum(r["feedback_turns"] for r in rows) / n, 4) if n else 0.0,
        "feedback_labels_mean": round(sum(r["n_feedback_labels"] for r in rows) / n, 4) if n else 0.0,
        "turns_mean": round(sum(r["num_turns"] for r in rows) / n, 4) if n else 0.0,
    }
    for key in [
        "answer_correct",
        "verified_success_proxy_strict",
        "verified_success_proxy_any",
        "suspicious_proxy_strict",
        "suspicious_proxy_any",
        "final_strict_unresolved",
        "final_any_unresolved",
    ]:
        out[f"{key}_rate"] = _rate(int(out[key]), n)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    args = p.parse_args()

    summaries = [_summarize(pathlib.Path(x)) for x in args.cases]
    pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    keys = [
        "path",
        "n",
        "answer_correct",
        "answer_correct_rate",
        "verified_success_proxy_strict",
        "verified_success_proxy_strict_rate",
        "verified_success_proxy_any",
        "verified_success_proxy_any_rate",
        "suspicious_proxy_strict",
        "suspicious_proxy_strict_rate",
        "suspicious_proxy_any",
        "suspicious_proxy_any_rate",
        "final_strict_unresolved_rate",
        "final_any_unresolved_rate",
        "feedback_turns_mean",
        "feedback_labels_mean",
        "turns_mean",
    ]
    with pathlib.Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k) for k in keys})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
