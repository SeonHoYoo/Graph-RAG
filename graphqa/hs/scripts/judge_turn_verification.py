#!/usr/bin/env python
"""Turn-level judge for VeriGraph verification and hint quality.

This script evaluates whether each turn's Q-D-T verification and injected
vg_hint were appropriate for the local state at that turn. It is separate from
case-level trajectory judging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from judge_prompt import (
    RAW_TURN_SYSTEM_PROMPT,
    RAW_TURN_USER_TEMPLATE,
    TURN_SYSTEM_PROMPT,
    TURN_USER_TEMPLATE,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal envs
    tqdm = None

try:
    from model_library.openai_client import create_openai_client
except Exception:  # pragma: no cover - handled at runtime
    create_openai_client = None


ErrorType = Literal[
    "none",
    "unsupported_claim",
    "conflict",
    "wrong_requirement_match",
    "overzealous_hint",
    "underpowered_hint",
    "retrieval_missing",
]


class TurnJudgeOutput(BaseModel):
    hint_needed: bool = Field(
        ...,
        description="Whether some verifier hint was appropriate at this turn.",
    )
    label_correct: bool = Field(
        ...,
        description="Whether the automatic verifier labels match the Q/D/T situation.",
    )
    hint_correct: bool = Field(
        ...,
        description="Whether the injected vg_hint was appropriate in content and strength.",
    )
    error_type: ErrorType = Field(
        ...,
        description="Main verification/hint quality issue for this turn.",
    )


VanillaErrorType = Literal[
    "none",
    "unsupported_claim",
    "conflict",
    "wrong_requirement_match",
    "retrieval_missing",
]


class VanillaTurnJudgeOutput(BaseModel):
    hint_needed: bool = Field(
        ...,
        description="Whether verifier feedback would have been useful at this vanilla turn.",
    )
    error_type: VanillaErrorType = Field(
        ...,
        description="Main local issue in the vanilla turn.",
    )


RawTurnJudgeOutput = VanillaTurnJudgeOutput


def _pydantic_schema(model_cls: Any) -> Dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def _pydantic_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True, help="Path to online_feedback_*_cases_*.json")
    p.add_argument("--output-dir", default="", help="Default: same directory as --cases")
    p.add_argument("--output-prefix", default="", help="Default: turn_judge_<cases stem>")
    p.add_argument("--judge-view", choices=["raw", "assisted"], default="raw")
    p.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini-2025-04-14"))
    p.add_argument("--base-url", default=os.environ.get("SKIML_API_BASE", os.environ.get("JUDGE_BASE_URL", "")))
    p.add_argument("--api-key", default="", help="Override SKIML API key.")
    p.add_argument("--max-cases", type=int, default=0, help="Maximum number of selected samples/cases to judge")
    p.add_argument("--max-turns", type=int, default=0, help="Optional safety cap on total turn rows")
    p.add_argument("--case-indices", type=int, nargs="*", default=None)
    p.add_argument("--turn-indices", type=int, nargs="*", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-doc-chars", type=int, default=2600)
    p.add_argument("--max-memory-chars", type=int, default=3600)
    p.add_argument("--max-think-chars", type=int, default=900)
    p.add_argument("--max-feedback-chars", type=int, default=1400)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=int(os.environ.get("JUDGE_MAX_RETRIES", "3")))
    p.add_argument("--retry-backoff-sec", type=float, default=float(os.environ.get("JUDGE_RETRY_BACKOFF_SEC", "8")))
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")
    p.add_argument("--dry-run", action="store_true")
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
    keep: List[Dict[str, Any]] = []
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


def _question_graph(case: Dict[str, Any]) -> str:
    return str(case.get("question_graph_raw") or "\n".join(case.get("question_triples", []) or []))


def _observer_by_turn(case: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(ev.get("turn", i)): ev
        for i, ev in enumerate(case.get("observer_events", []) or [])
        if isinstance(ev, dict)
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
        observers = _observer_by_turn(case)
        retrieved_by_turn = _retrieval_info_by_turn(case)
        normalized: List[Dict[str, Any]] = []
        for i, turn in enumerate(online_turns):
            if not isinstance(turn, dict):
                continue
            turn_no = int(turn.get("turn", i))
            ev = observers.get(turn_no, {})
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


def _build_turn_payloads(cases: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected_cases = set(args.case_indices or [])
    selected_turns = set(args.turn_indices or [])
    payloads: List[Dict[str, Any]] = []
    processed_cases = 0
    for case_index, case in enumerate(cases):
        if selected_cases and case_index not in selected_cases:
            continue
        if args.max_cases and processed_cases >= args.max_cases:
            break
        processed_cases += 1
        prior_evidence: List[str] = []
        for turn in _iter_common_turns(case):
            turn_no = int(turn.get("turn", len(payloads)))
            current_retrieved = str(turn.get("retrieved_information") or "")
            if selected_turns and turn_no not in selected_turns:
                if current_retrieved.strip():
                    prior_evidence.append(f"[turn {turn_no}]\n{current_retrieved.strip()}")
                continue
            feedback = turn.get("injected_feedback") or ""
            if not feedback and turn.get("source") == "vanilla_searchr1":
                feedback = "(vanilla SearchR1: no verifier hint was injected)"
            auto_labels = json.dumps(_compact_labels(turn), ensure_ascii=False, indent=2)
            if args.judge_view == "raw":
                auto_labels = "[]"
                feedback = ""
            payloads.append({
                "source": turn.get("source") or "unknown",
                "judge_view": args.judge_view,
                "case_index": case_index,
                "turn_number": turn_no,
                "dataset": case.get("dataset"),
                "uid": case.get("uid"),
                "question": case.get("question") or "",
                "gold_answer": case.get("answer") or "",
                "predicted_answer": case.get("predicted_answer") or "",
                "question_graph": _question_graph(case),
                "think_text": _clip(turn.get("think_text"), args.max_think_chars),
                "query": turn.get("query") or "",
                "previous_evidence_memory": _clip("\n\n".join(prior_evidence), args.max_memory_chars),
                "retrieved_information": _clip(current_retrieved, args.max_doc_chars),
                "auto_labels": auto_labels,
                "vg_hint": _clip(feedback, args.max_feedback_chars),
            })
            if current_retrieved.strip():
                prior_evidence.append(f"[turn {turn_no}]\n{current_retrieved.strip()}")
            if args.max_turns and len(payloads) >= args.max_turns:
                return payloads
    return payloads


def _call_judge(client: Any, args: argparse.Namespace, payload: Dict[str, Any]) -> Dict[str, Any]:
    is_assisted = args.judge_view == "assisted" and payload.get("source") != "vanilla_searchr1"
    if not is_assisted:
        user_prompt = RAW_TURN_USER_TEMPLATE.format(
            **payload,
            schema=json.dumps(_pydantic_schema(RawTurnJudgeOutput), ensure_ascii=False, indent=2),
        )
        system_prompt = RAW_TURN_SYSTEM_PROMPT
    else:
        user_prompt = TURN_USER_TEMPLATE.format(
            **payload,
            schema=json.dumps(_pydantic_schema(TurnJudgeOutput), ensure_ascii=False, indent=2),
        )
        system_prompt = TURN_SYSTEM_PROMPT
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
    if is_assisted:
        result = _pydantic_to_dict(TurnJudgeOutput.model_validate(raw_result))
    else:
        result = _pydantic_to_dict(RawTurnJudgeOutput.model_validate(raw_result))
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
            err = repr(exc)
            retryable = (
                "429" in err
                or "RateLimit" in err
                or "No deployments available" in err
                or "JSONDecodeError" in err
                or "judge response is not a JSON object" in err
            )
            if not retryable or attempt >= args.max_retries:
                raise
            time.sleep(args.retry_backoff_sec * (attempt + 1))
    raise last_exc or RuntimeError("judge call failed")


def _summarize(rows: Iterable[Dict[str, Any]], judge_view: str = "") -> Dict[str, Any]:
    items = list(rows)
    errors = Counter(str(r.get("error_type") or "uncategorized") for r in items)
    out = {
        "judge_view": judge_view,
        "n_turns": len(items),
        "hint_needed_rate": sum(1 for r in items if r.get("hint_needed") is True) / len(items) if items else 0.0,
        "by_error_type": dict(errors),
        "n_hint_true_error_none": sum(1 for r in items if r.get("hint_needed") is True and r.get("error_type") == "none"),
        "n_hint_false_error_non_none": sum(1 for r in items if r.get("hint_needed") is False and r.get("error_type") not in (None, "none")),
    }
    label_rows = [r for r in items if "label_correct" in r]
    hint_rows = [r for r in items if "hint_correct" in r]
    if label_rows:
        out["label_correct_rate"] = sum(1 for r in label_rows if r.get("label_correct") is True) / len(label_rows)
    if hint_rows:
        out["hint_correct_rate"] = sum(1 for r in hint_rows if r.get("hint_correct") is True) / len(hint_rows)
    return out


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    cases = _load_cases(cases_path)
    payloads = _build_turn_payloads(cases, args)

    out_dir = Path(args.output_dir) if args.output_dir else cases_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"turn_judge_{cases_path.stem}"
    results_path = out_dir / f"{prefix}.json"
    summary_path = out_dir / f"{prefix}_summary.json"
    inputs_path = out_dir / f"{prefix}_inputs.json"

    client = None
    if not args.dry_run:
        client = _make_client(args)

    with inputs_path.open("w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2, ensure_ascii=False)
    if args.dry_run:
        print(f"[turn_judge] dry-run wrote inputs: {inputs_path}")
        return

    outputs: List[Dict[str, Any]] = []
    iterator = payloads
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(payloads, desc="turn_judge", unit="turn", dynamic_ncols=True)
    for payload in iterator:
        t0 = time.perf_counter()
        try:
            judged = _call_judge_with_retries(client, args, payload)
            judged["_judge_error"] = ""
        except (ValidationError, Exception) as exc:
            judged = {
                "hint_needed": False,
                "error_type": None,
                "_judge_error": repr(exc),
            }
            if args.judge_view == "assisted" and payload.get("source") != "vanilla_searchr1":
                judged.update({
                    "label_correct": False,
                    "hint_correct": False,
                })
        judged.update({
            "source": payload.get("source"),
            "judge_view": args.judge_view,
            "case_index": payload["case_index"],
            "turn_number": payload["turn_number"],
            "dataset": payload.get("dataset"),
            "uid": payload.get("uid"),
            "question": payload.get("question"),
            "gold_answer": payload.get("gold_answer"),
            "predicted_answer": payload.get("predicted_answer"),
            "query": payload.get("query"),
            "judge_model": args.model,
            "judge_latency_sec": round(time.perf_counter() - t0, 6),
        })
        outputs.append(judged)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_summarize(outputs, args.judge_view), f, indent=2, ensure_ascii=False)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    summary = _summarize(outputs, args.judge_view)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[turn_judge] wrote results: {results_path}")
    print(f"[turn_judge] wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
