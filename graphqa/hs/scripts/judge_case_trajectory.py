#!/usr/bin/env python
"""Case-level LLM judge for SearchR1 + VeriGraph trajectories.

This script is analysis-only. It reads saved HS online-feedback case JSON files
and asks an OpenAI-compatible judge model to classify each full trajectory into
true_success, suspicious_correct, explained_failure, or unexplained_failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from judge_prompt import (
    ASSISTED_CASE_SYSTEM_PROMPT,
    ASSISTED_CASE_USER_TEMPLATE,
    JUDGE_PROMPT_VERSION,
    RAW_CASE_SYSTEM_PROMPT,
    RAW_CASE_USER_TEMPLATE,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal envs
    tqdm = None

try:
    from model_library.openai_client import create_openai_client
except Exception:  # pragma: no cover - handled at runtime
    create_openai_client = None


Category = Literal[
    "true_success",
    "suspicious_correct",
    "explained_failure",
    "unexplained_failure",
]


class TrajectoryJudgeOutput(BaseModel):
    final_requirements_supported: bool = Field(
        ...,
        description=(
            "Whether the final answer's necessary requirements are supported by retrieved "
            "documents, not merely asserted in thoughts."
        ),
    )
    evidence_issue_found: bool = Field(
        ...,
        description=(
            "Whether the trajectory shows a missing, unsupported, wrong-entity, "
            "guessed, or conflicting requirement on the final-answer path."
        ),
    )


def _canonical_category(
    answer_correct: Optional[bool],
    final_requirements_supported: Optional[bool],
    evidence_issue_found: Optional[bool],
) -> Optional[Category]:
    if answer_correct is True and final_requirements_supported is True:
        return "true_success"
    if answer_correct is True and final_requirements_supported is False:
        return "suspicious_correct"
    if answer_correct is False and evidence_issue_found is True:
        return "explained_failure"
    if answer_correct is False and evidence_issue_found is False:
        return "unexplained_failure"
    return None


def _pydantic_schema(model_cls: Any) -> Dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def _pydantic_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _normalize_answer(text: Any) -> str:
    s = str(text or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _deterministic_answer_correct(case: Dict[str, Any]) -> bool:
    row = case.get("csv_row") or {}
    if row.get("em") not in (None, ""):
        try:
            return float(row.get("em")) >= 1.0
        except Exception:
            pass
    if case.get("answer_matches_gold") is not None:
        return bool(case.get("answer_matches_gold"))
    return _normalize_answer(case.get("predicted_answer")) == _normalize_answer(case.get("answer"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True, help="Path to online_feedback_*_cases_*.json")
    p.add_argument("--output-dir", default="", help="Default: same directory as --cases")
    p.add_argument("--output-prefix", default="", help="Default: case_judge_<cases stem>")
    p.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini-2025-04-14"))
    p.add_argument("--base-url", default=os.environ.get("SKIML_API_BASE", os.environ.get("JUDGE_BASE_URL", "")))
    p.add_argument("--api-key", default="", help="Override SKIML API key.")
    p.add_argument(
        "--judge-view",
        choices=["raw", "assisted"],
        default=os.environ.get("JUDGE_VIEW", "assisted"),
        help="raw: judge only question/think/query/docs. assisted: include question graph, verifier labels, and vg_hint.",
    )
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--case-indices", type=int, nargs="*", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-doc-chars-per-turn", type=int, default=2200)
    p.add_argument("--max-think-chars", type=int, default=900)
    p.add_argument("--max-feedback-chars", type=int, default=1400)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=int(os.environ.get("JUDGE_MAX_RETRIES", "3")))
    p.add_argument("--retry-backoff-sec", type=float, default=float(os.environ.get("JUDGE_RETRY_BACKOFF_SEC", "8")))
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")
    p.add_argument("--dry-run", action="store_true", help="Write judge inputs without calling the model")
    return p.parse_args()


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [x for x in data if isinstance(x, dict)]


def _clip(text: Any, limit: int) -> str:
    s = str(text or "").strip()
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit].rstrip() + "\n...[truncated]"


def _extract_information(text: str) -> str:
    matches = re.findall(r"<information>(.*?)</information>", text or "", flags=re.S | re.I)
    if matches:
        return "\n\n".join(m.strip() for m in matches if m.strip())
    return ""


def _safe_json_loads(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("judge response is not a JSON object")
    return obj


def _compact_labels(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    keep = []
    for lab in turn.get("verification_labels", []) or []:
        if not isinstance(lab, dict):
            continue
        keep.append({
            "requirement": lab.get("requirement"),
            "q_d": lab.get("q_d"),
            "q_t": lab.get("q_t"),
            "d_t": lab.get("d_t"),
            "action": lab.get("action"),
            "query_q": lab.get("query_q"),
            "doc_value": lab.get("doc_value"),
            "think_value": lab.get("think_value"),
            "doc_candidate": _clip(lab.get("doc_candidate"), 260),
            "think_candidate": _clip(lab.get("think_candidate"), 260),
        })
    return keep


def _critical_label_summary(auto_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    critical: List[Dict[str, Any]] = []
    support_claims: List[Dict[str, Any]] = []
    for turn_entry in auto_labels:
        turn_no = turn_entry.get("turn")
        for lab in turn_entry.get("labels", []) or []:
            if not isinstance(lab, dict):
                continue
            d_t = str(lab.get("d_t") or "")
            action = str(lab.get("action") or "")
            q_d = str(lab.get("q_d") or "")
            q_t = str(lab.get("q_t") or "")
            if q_d == "support":
                support_claims.append({
                    "turn": turn_no,
                    "requirement": lab.get("requirement"),
                    "doc_value": lab.get("doc_value"),
                    "doc_candidate": _clip(lab.get("doc_candidate"), 160),
                    "think_value": lab.get("think_value"),
                    "think_candidate": _clip(lab.get("think_candidate"), 160),
                })
            issue = ""
            if d_t in {"unsupported_think_claim", "conflict"}:
                issue = d_t
            elif action == "do_not_carry_forward":
                issue = action
            elif q_d == "weak_or_irrelevant_doc" and q_t == "claims_value":
                issue = "claimed_value_without_strong_doc"
            if not issue:
                continue
            critical.append({
                "turn": turn_no,
                "issue": issue,
                "requirement": lab.get("requirement"),
                "doc_value": lab.get("doc_value"),
                "think_value": lab.get("think_value"),
                "doc_candidate": _clip(lab.get("doc_candidate"), 160),
                "think_candidate": _clip(lab.get("think_candidate"), 160),
            })
    return {
        "n_critical_labels": len(critical),
        "critical_labels": critical[:20],
        "n_support_claims_to_check": len(support_claims),
        "support_claims_to_check": support_claims[:20],
    }


def _extract_think(text: Any) -> str:
    matches = re.findall(r"<think>(.*?)</think>", str(text or ""), flags=re.S | re.I)
    if matches:
        return matches[-1].strip()
    return str(text or "").strip()


def _format_search_results(search_results: Any) -> str:
    if not isinstance(search_results, list):
        return str(search_results or "")
    return "\n".join(f"Doc {i + 1}{doc}" for i, doc in enumerate(search_results))


def _retrieval_info_by_turn(case: Dict[str, Any]) -> Dict[int, str]:
    retrieval_info = case.get("retrieval_info") or {}
    turns = case.get("retrieval_turns") or retrieval_info.get("retrieval_turns") or []
    out: Dict[int, str] = {}
    if not isinstance(turns, list):
        return out
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        turn_no = int(turn.get("turn", i))
        docs = turn.get("search_results")
        if docs is None:
            docs = turn.get("retrieved_docs")
        out[turn_no] = _format_search_results(docs)
    return out


def _iter_vanilla_turns(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    retrieval_info = case.get("retrieval_info") or {}
    turns = case.get("retrieval_turns") or retrieval_info.get("retrieval_turns") or []
    if not isinstance(turns, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        normalized.append({
            "turn": int(turn.get("turn", i)),
            "think_text": _extract_think(turn.get("model_output")),
            "query": turn.get("query") or "",
            "retrieved_information": _format_search_results(turn.get("search_results")),
            "injected_feedback": "",
            "verification_labels": [],
            "source": "vanilla_searchr1",
        })
    return normalized


def _iter_common_turns(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    online_turns = case.get("turn_records") or []
    if online_turns:
        observer_by_turn = {
            int(ev.get("turn", i)): ev
            for i, ev in enumerate(case.get("observer_events", []) or [])
            if isinstance(ev, dict)
        }
        retrieved_by_turn = _retrieval_info_by_turn(case)
        normalized: List[Dict[str, Any]] = []
        for i, turn in enumerate(online_turns):
            if not isinstance(turn, dict):
                continue
            turn_no = int(turn.get("turn", i))
            ev = observer_by_turn.get(turn_no, {})
            info = _extract_information(str(ev.get("search_text") or ev.get("full_response") or ""))
            if not info:
                info = retrieved_by_turn.get(turn_no, "")
            normalized.append({
                "turn": turn_no,
                "think_text": turn.get("think_text") or "",
                "query": turn.get("query") or "",
                "retrieved_information": info,
                "injected_feedback": turn.get("injected_feedback") or "",
                "verification_labels": turn.get("verification_labels") or [],
                "source": "online_feedback",
            })
        return normalized
    return _iter_vanilla_turns(case)


def _build_case_payload(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    trajectory_parts: List[str] = []
    auto_labels: List[Dict[str, Any]] = []
    judge_view = str(args.judge_view or "assisted")
    turns = _iter_common_turns(case)
    for turn in turns:
        turn_no = int(turn.get("turn", len(trajectory_parts)))
        labels = _compact_labels(turn)
        auto_labels.append({"turn": turn_no + 1, "labels": labels})
        feedback = turn.get("injected_feedback") or ""
        if not feedback and turn.get("source") == "vanilla_searchr1":
            feedback = "(vanilla SearchR1: no verifier hint was injected)"
        if judge_view == "raw":
            trajectory_lines = [
                f"TURN {turn_no + 1}",
                f"think: {_clip(turn.get('think_text'), args.max_think_chars)}",
                f"query: {turn.get('query') or ''}",
                f"retrieved_information: {_clip(turn.get('retrieved_information'), args.max_doc_chars_per_turn)}",
            ]
        else:
            trajectory_lines = [
                f"TURN {turn_no + 1}",
                "turn_evidence_check: Check any concrete claims in think against this turn's retrieved_information. If a later turn reuses a value, that value must be supported by retrieved documents from this or another turn.",
                f"think: {_clip(turn.get('think_text'), args.max_think_chars)}",
                f"query: {turn.get('query') or ''}",
                f"retrieved_information: {_clip(turn.get('retrieved_information'), args.max_doc_chars_per_turn)}",
                f"vg_hint_given_to_next_turn: {_clip(feedback, args.max_feedback_chars)}",
            ]
        trajectory_parts.append("\n".join(trajectory_lines))
    if judge_view == "raw":
        question_graph = ""
        auto_labels_text = "[]"
        critical_summary_text = "{}"
    else:
        question_graph = case.get("question_graph_raw") or "\n".join(case.get("question_triples", []) or [])
        auto_labels_text = json.dumps(auto_labels, ensure_ascii=False, indent=2)
        critical_summary_text = json.dumps(_critical_label_summary(auto_labels), ensure_ascii=False, indent=2)
    return {
        "judge_view": judge_view,
        "question": case.get("question") or "",
        "gold_answer": case.get("answer") or "",
        "predicted_answer": case.get("predicted_answer") or "",
        "answer_correct": str(_deterministic_answer_correct(case)).lower(),
        "question_graph": question_graph,
        "trajectory": "\n\n".join(trajectory_parts),
        "auto_labels": auto_labels_text,
        "critical_auto_label_summary": critical_summary_text,
    }


def _call_judge(client: Any, args: argparse.Namespace, payload: Dict[str, Any]) -> Dict[str, Any]:
    if str(args.judge_view or "assisted") == "raw":
        system_prompt = RAW_CASE_SYSTEM_PROMPT
        user_template = RAW_CASE_USER_TEMPLATE
    else:
        system_prompt = ASSISTED_CASE_SYSTEM_PROMPT
        user_template = ASSISTED_CASE_USER_TEMPLATE
    user_prompt = user_template.format(
        **payload,
        schema=json.dumps(_pydantic_schema(TrajectoryJudgeOutput), ensure_ascii=False, indent=2),
    )
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=args.temperature,
        top_p=1.0,
        seed=42,
    )
    content = resp.choices[0].message.content or ""
    raw_result = _safe_json_loads(content)
    result = _pydantic_to_dict(TrajectoryJudgeOutput.model_validate(raw_result))
    answer_correct = str(payload.get("answer_correct", "")).lower() == "true"
    result["answer_correct"] = answer_correct
    result["judge_view"] = str(args.judge_view or "assisted")
    result["category"] = _canonical_category(
        answer_correct,
        result.get("final_requirements_supported"),
        result.get("evidence_issue_found"),
    )
    result["_judge_raw"] = content
    return result


def _make_client(args: argparse.Namespace) -> Any:
    if args.base_url:
        os.environ["SKIML_API_BASE"] = args.base_url
    if create_openai_client is None:
        raise RuntimeError("model_library.openai_client.create_openai_client is required unless --dry-run is used")
    return create_openai_client(api_key=args.api_key or os.environ.get("SKIML_API_KEY") or None)


def _call_judge_with_retries(client: Any, args: argparse.Namespace, payload: Dict[str, Any]) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(max(args.max_retries, 0) + 1):
        try:
            return _call_judge(client, args, payload)
        except Exception as exc:
            last_exc = exc
            retryable = "429" in repr(exc) or "RateLimit" in repr(exc) or "No deployments available" in repr(exc)
            if not retryable or attempt >= args.max_retries:
                raise
            time.sleep(args.retry_backoff_sec * (attempt + 1))
    raise last_exc or RuntimeError("judge call failed")


def _select_cases(cases: List[Dict[str, Any]], args: argparse.Namespace) -> List[tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(cases))
    if args.case_indices:
        wanted = set(args.case_indices)
        indexed = [(i, c) for i, c in indexed if i in wanted]
    if args.max_cases and args.max_cases > 0:
        indexed = indexed[: args.max_cases]
    return indexed


def _summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(results)
    categories = Counter(str(r.get("category") or "uncategorized") for r in rows)
    answer_correct = sum(1 for r in rows if r.get("answer_correct") is True)
    final_supported = sum(1 for r in rows if r.get("final_requirements_supported") is True)
    issue_found = sum(1 for r in rows if r.get("evidence_issue_found") is True)
    return {
        "n_cases": len(rows),
        "answer_correct_rate": answer_correct / len(rows) if rows else 0.0,
        "final_requirements_supported_rate": final_supported / len(rows) if rows else 0.0,
        "evidence_issue_found_rate": issue_found / len(rows) if rows else 0.0,
        "by_category": dict(categories),
    }


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    cases = _load_cases(cases_path)
    selected = _select_cases(cases, args)

    out_dir = Path(args.output_dir) if args.output_dir else cases_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"case_judge_{cases_path.stem}"
    results_path = out_dir / f"{prefix}.json"
    summary_path = out_dir / f"{prefix}_summary.json"
    inputs_path = out_dir / f"{prefix}_inputs.json"

    print(f"[case_judge] judge_prompt_version = {JUDGE_PROMPT_VERSION}")

    client = None
    if not args.dry_run:
        client = _make_client(args)

    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    iterator = selected
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(selected, desc="case_judge", unit="case", dynamic_ncols=True)
    for case_index, case in iterator:
        payload = _build_case_payload(case, args)
        input_row = {
            "case_index": case_index,
            "dataset": case.get("dataset"),
            "uid": case.get("uid"),
            "question": case.get("question"),
            "gold_answer": case.get("answer"),
            "predicted_answer": case.get("predicted_answer"),
            "judge_payload": payload,
        }
        inputs.append(input_row)
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        try:
            judged = _call_judge_with_retries(client, args, payload)
            judged["_judge_error"] = ""
        except (ValidationError, Exception) as exc:
            answer_correct = _deterministic_answer_correct(case)
            judged = {
                "judge_view": str(args.judge_view or "assisted"),
                "answer_correct": answer_correct,
                "final_requirements_supported": None,
                "evidence_issue_found": None,
                "category": _canonical_category(answer_correct, None, None),
                "_judge_error": repr(exc),
            }
        judged.update({
            "case_index": case_index,
            "dataset": case.get("dataset"),
            "uid": case.get("uid"),
            "question": case.get("question"),
            "gold_answer": case.get("answer"),
            "predicted_answer": case.get("predicted_answer"),
            "judge_model": args.model,
            "judge_view": str(args.judge_view or "assisted"),
            "judge_latency_sec": round(time.perf_counter() - t0, 6),
        })
        outputs.append(judged)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_summarize(outputs), f, indent=2, ensure_ascii=False)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    with inputs_path.open("w", encoding="utf-8") as f:
        json.dump(inputs, f, indent=2, ensure_ascii=False)
    if args.dry_run:
        print(f"[case_judge] dry-run wrote inputs: {inputs_path}")
        return

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    summary = _summarize(outputs)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[case_judge] wrote results: {results_path}")
    print(f"[case_judge] wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
