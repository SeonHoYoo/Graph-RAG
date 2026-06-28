#!/usr/bin/env python
"""Convert Seonho online_corrector JSONL files into HS judge case JSON lists."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


MODES = [
    "vanilla",
    "control",
    "corrector",
    "corrector_v2",
    "coach_g",
    "coach_e",
    "coach_f",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Seonho online_corrector_*.jsonl")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--output", required=True, help="Output JSON list path for judge scripts")
    p.add_argument("--skip-missing", action="store_true", help="Skip rows where selected mode is missing")
    p.add_argument("--max-cases", type=int, default=0)
    return p.parse_args()


def _normalize_answer(text: Any) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _answer_correct(row: Dict[str, Any], mode_obj: Dict[str, Any]) -> bool:
    if mode_obj.get("em") not in (None, ""):
        try:
            return float(mode_obj.get("em")) >= 1.0
        except Exception:
            pass
    return _normalize_answer(mode_obj.get("predicted_answer")) == _normalize_answer(row.get("answer"))


def _question_graph_raw(row: Dict[str, Any]) -> str:
    qg = row.get("question_graph") or {}
    if isinstance(qg, dict):
        return str(qg.get("raw_graph") or "\n".join(qg.get("raw_triples") or qg.get("kept_triples") or []))
    return ""


def _question_triples(row: Dict[str, Any]) -> List[str]:
    qg = row.get("question_graph") or {}
    if isinstance(qg, dict):
        triples = qg.get("kept_triples") or qg.get("raw_triples") or []
        if isinstance(triples, list):
            return [str(t) for t in triples if str(t).strip()]
    return []


def _parse_response_turns(full_response: Any) -> List[Dict[str, str]]:
    text = str(full_response or "")
    pat = re.compile(
        r"<think>(?P<think>.*?)</think>\s*"
        r"<search>(?P<search>.*?)</search>\s*"
        r"<information>(?P<info>.*?)</information>",
        flags=re.S | re.I,
    )
    turns: List[Dict[str, str]] = []
    for m in pat.finditer(text):
        turns.append({
            "think_text": m.group("think").strip(),
            "query": m.group("search").strip(),
            "information": m.group("info").strip(),
        })
    return turns


def _split_docs(info: str) -> List[str]:
    info = str(info or "").strip()
    if not info:
        return []
    parts = re.split(r"(?=Doc\s+\d+\s*\()", info)
    docs = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        s = re.sub(r"^Doc\s+\d+\s*", "", s).strip()
        docs.append(s)
    return docs or [info]


def _docs_from_retrieval_turn(turn: Dict[str, Any], parsed_turn: Optional[Dict[str, str]]) -> List[str]:
    if parsed_turn:
        docs = _split_docs(parsed_turn.get("information") or "")
        if docs:
            return docs
    docs = turn.get("search_results")
    if docs is None:
        docs = turn.get("retrieved_docs")
    if isinstance(docs, list):
        return [str(d) for d in docs]
    if docs:
        return [str(docs)]
    return []


def _simplify_alignment_label(item: Dict[str, Any]) -> Dict[str, Any]:
    doc_match = item.get("doc_match") or {}
    think_match = item.get("think_match") or {}
    doc_pass = bool(doc_match.get("passes_threshold"))
    think_pass = bool(think_match.get("passes_threshold"))
    return {
        "requirement": item.get("q_text"),
        "q_d": "support" if doc_pass else "weak_or_irrelevant_doc",
        "q_t": "relevant_think" if think_pass else "weak_relevant_think",
        "d_t": "aligned" if doc_pass and think_pass else "open_or_unverified",
        "action": "can_use" if doc_pass else "search_exact_title_relation",
        "query_q": "already_supported" if doc_pass else "targets_unresolved_requirement",
        "doc_value": "",
        "think_value": "",
        "doc_candidate": doc_match.get("candidate_text"),
        "think_candidate": think_match.get("candidate_text"),
    }


def _turn_records(mode_obj: Dict[str, Any], parsed: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    records = mode_obj.get("turn_records")
    if not isinstance(records, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        turn_no = int(rec.get("turn", i))
        new = dict(rec)
        if not new.get("think_text") and turn_no < len(parsed):
            new["think_text"] = parsed[turn_no].get("think_text", "")
        if not new.get("query") and turn_no < len(parsed):
            new["query"] = parsed[turn_no].get("query", "")
        if "verification_labels" not in new:
            alignment = new.get("alignment")
            if isinstance(alignment, list):
                new["verification_labels"] = [
                    _simplify_alignment_label(x) for x in alignment if isinstance(x, dict)
                ]
            else:
                new["verification_labels"] = []
        out.append(new)
    return out


def _retrieval_turns(mode_obj: Dict[str, Any], parsed: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    turns = mode_obj.get("retrieval_turns") or []
    if not isinstance(turns, list):
        turns = []
    n = max(len(turns), len(parsed))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        src = turns[i] if i < len(turns) and isinstance(turns[i], dict) else {}
        parsed_turn = parsed[i] if i < len(parsed) else None
        think = parsed_turn.get("think_text", "") if parsed_turn else ""
        query = src.get("query") or (parsed_turn.get("query", "") if parsed_turn else "")
        out.append({
            "turn": int(src.get("turn", i)),
            "query": query,
            "model_output": f"<think>{think}</think>" if think else "",
            "search_results": _docs_from_retrieval_turn(src, parsed_turn),
        })
    return out


def convert_row(row: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    mode_obj = row.get(mode)
    if not isinstance(mode_obj, dict):
        return None
    parsed = _parse_response_turns(mode_obj.get("full_response"))
    case = {
        "dataset": row.get("dataset"),
        "uid": row.get("uid"),
        "index": row.get("index"),
        "source_mode": mode,
        "question": row.get("question") or "",
        "answer": row.get("answer") or "",
        "answer_aliases": row.get("answer_aliases") or [],
        "csv_row": row.get("csv_row") or {},
        "question_graph_raw": _question_graph_raw(row),
        "question_triples": _question_triples(row),
        "predicted_answer": mode_obj.get("predicted_answer") or "",
        "answer_matches_gold": _answer_correct(row, mode_obj),
        "retrieval_turns": _retrieval_turns(mode_obj, parsed),
        "num_turns": mode_obj.get("num_turns"),
        "em": mode_obj.get("em"),
        "f1": mode_obj.get("f1"),
    }
    turn_records = _turn_records(mode_obj, parsed)
    if turn_records:
        case["turn_records"] = turn_records
    return case


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    n_missing = 0
    for row in iter_jsonl(Path(args.input)):
        converted = convert_row(row, args.mode)
        if converted is None:
            n_missing += 1
            if args.skip_missing:
                continue
            continue
        rows.append(converted)
        if args.max_cases and len(rows) >= args.max_cases:
            break
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[convert_seonho_cases] wrote {len(rows)} cases to {out}")
    if n_missing:
        print(f"[convert_seonho_cases] skipped missing mode rows: {n_missing}")


if __name__ == "__main__":
    main()
