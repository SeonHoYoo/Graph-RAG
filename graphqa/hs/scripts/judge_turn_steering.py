#!/usr/bin/env python
"""Judge whether each VeriGraph hint steers the next SearchR1 turn.

This script is separate from turn-level verification. Verification asks whether
the hint/labels are locally appropriate at turn t. Steering asks whether the
next turn t+1 actually responds to the hint injected after turn t.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from judge_prompt import STEERING_SYSTEM_PROMPT, STEERING_USER_TEMPLATE
from judge_turn_verification import (
    _clip,
    _compact_labels,
    _iter_common_turns,
    _load_cases,
    _make_client,
    _pydantic_schema,
    _pydantic_to_dict,
    _question_graph,
    _safe_json_loads,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SteeringErrorType = Literal[
    "none",
    "hint_not_actionable",
    "query_ignored_hint",
    "think_repeated_error",
    "retrieval_not_improved",
    "oversteered_wrong_direction",
]


class SteeringJudgeOutput(BaseModel):
    hint_was_actionable: bool = Field(
        ...,
        description="Whether the vg_hint gave a useful next-step direction.",
    )
    next_query_followed_hint: bool = Field(
        ...,
        description="Whether the next search query followed the hint.",
    )
    next_think_revised_error: bool = Field(
        ...,
        description="Whether the next think avoided or corrected the flagged issue.",
    )
    same_error_repeated: bool = Field(
        ...,
        description="Whether the next turn repeated the same flagged error.",
    )
    steering_success: bool = Field(
        ...,
        description="Whether the hint successfully steered the next turn.",
    )
    steering_error_type: SteeringErrorType = Field(
        ...,
        description="Main reason steering failed, or none if successful.",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True, help="Path to VG online_feedback cases JSON")
    p.add_argument("--output-dir", default="", help="Default: same directory as --cases")
    p.add_argument("--output-prefix", default="", help="Default: steering_judge_<cases stem>")
    p.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini-2025-04-14"))
    p.add_argument("--base-url", default=os.environ.get("SKIML_API_BASE", os.environ.get("JUDGE_BASE_URL", "")))
    p.add_argument("--api-key", default="", help="Override SKIML API key.")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=0, help="Optional safety cap on t->t+1 pairs")
    p.add_argument("--case-indices", type=int, nargs="*", default=None)
    p.add_argument("--turn-indices", type=int, nargs="*", default=None, help="Evaluate only hints after these turn numbers")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-doc-chars", type=int, default=2200)
    p.add_argument("--max-memory-chars", type=int, default=3200)
    p.add_argument("--max-think-chars", type=int, default=800)
    p.add_argument("--max-feedback-chars", type=int, default=1400)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=int(os.environ.get("JUDGE_MAX_RETRIES", "3")))
    p.add_argument("--retry-backoff-sec", type=float, default=float(os.environ.get("JUDGE_RETRY_BACKOFF_SEC", "8")))
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _build_pair_payloads(cases: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
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

        turns = _iter_common_turns(case)
        if not turns or all(t.get("source") == "vanilla_searchr1" for t in turns):
            continue

        prior_evidence: List[str] = []
        for i, turn in enumerate(turns[:-1]):
            turn_no = int(turn.get("turn", i))
            current_retrieved = str(turn.get("retrieved_information") or "")
            if selected_turns and turn_no not in selected_turns:
                if current_retrieved.strip():
                    prior_evidence.append(f"[turn {turn_no}]\n{current_retrieved.strip()}")
                continue

            feedback = str(turn.get("injected_feedback") or "").strip()
            if not feedback:
                if current_retrieved.strip():
                    prior_evidence.append(f"[turn {turn_no}]\n{current_retrieved.strip()}")
                continue

            next_turn = turns[i + 1]
            next_turn_no = int(next_turn.get("turn", turn_no + 1))
            payloads.append({
                "source": turn.get("source") or "online_feedback",
                "case_index": case_index,
                "turn_number": turn_no,
                "next_turn_number": next_turn_no,
                "dataset": case.get("dataset"),
                "uid": case.get("uid"),
                "question": case.get("question") or "",
                "gold_answer": case.get("answer") or "",
                "predicted_answer": case.get("predicted_answer") or "",
                "question_graph": _question_graph(case),
                "previous_evidence_memory": _clip("\n\n".join(prior_evidence), args.max_memory_chars),
                "think_text": _clip(turn.get("think_text"), args.max_think_chars),
                "query": turn.get("query") or "",
                "retrieved_information": _clip(current_retrieved, args.max_doc_chars),
                "auto_labels": json.dumps(_compact_labels(turn), ensure_ascii=False, indent=2),
                "vg_hint": _clip(feedback, args.max_feedback_chars),
                "next_think_text": _clip(next_turn.get("think_text"), args.max_think_chars),
                "next_query": next_turn.get("query") or "",
                "next_retrieved_information": _clip(next_turn.get("retrieved_information"), args.max_doc_chars),
            })
            if current_retrieved.strip():
                prior_evidence.append(f"[turn {turn_no}]\n{current_retrieved.strip()}")
            if args.max_pairs and len(payloads) >= args.max_pairs:
                return payloads

        last_turn = turns[-1]
        last_docs = str(last_turn.get("retrieved_information") or "")
        if last_docs.strip():
            prior_evidence.append(f"[turn {last_turn.get('turn', len(turns) - 1)}]\n{last_docs.strip()}")

    return payloads


def _call_judge(client: Any, args: argparse.Namespace, payload: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = STEERING_USER_TEMPLATE.format(
        **payload,
        schema=json.dumps(_pydantic_schema(SteeringJudgeOutput), ensure_ascii=False, indent=2),
    )
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": STEERING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=args.temperature,
        top_p=1.0,
        seed=42,
    )
    content = resp.choices[0].message.content or ""
    raw_result = _safe_json_loads(content)
    result = _pydantic_to_dict(SteeringJudgeOutput.model_validate(raw_result))
    result["_judge_raw"] = content
    return result


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


def _summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows)
    by_error = Counter(str(r.get("steering_error_type") or "uncategorized") for r in items)
    out = {
        "n_pairs": len(items),
        "hint_actionable_rate": sum(1 for r in items if r.get("hint_was_actionable") is True) / len(items) if items else 0.0,
        "next_query_followed_rate": sum(1 for r in items if r.get("next_query_followed_hint") is True) / len(items) if items else 0.0,
        "next_think_revised_rate": sum(1 for r in items if r.get("next_think_revised_error") is True) / len(items) if items else 0.0,
        "same_error_repeated_rate": sum(1 for r in items if r.get("same_error_repeated") is True) / len(items) if items else 0.0,
        "steering_success_rate": sum(1 for r in items if r.get("steering_success") is True) / len(items) if items else 0.0,
        "by_steering_error_type": dict(by_error),
        "n_success_error_non_none": sum(1 for r in items if r.get("steering_success") is True and r.get("steering_error_type") != "none"),
        "n_failure_error_none": sum(1 for r in items if r.get("steering_success") is False and r.get("steering_error_type") == "none"),
    }
    actionable = [r for r in items if r.get("hint_was_actionable") is True]
    if actionable:
        out["steering_success_rate_when_actionable"] = (
            sum(1 for r in actionable if r.get("steering_success") is True) / len(actionable)
        )
    return out


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    cases = _load_cases(cases_path)
    payloads = _build_pair_payloads(cases, args)

    out_dir = Path(args.output_dir) if args.output_dir else cases_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"steering_judge_{cases_path.stem}"
    results_path = out_dir / f"{prefix}.json"
    summary_path = out_dir / f"{prefix}_summary.json"
    inputs_path = out_dir / f"{prefix}_inputs.json"

    with inputs_path.open("w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2, ensure_ascii=False)
    if args.dry_run:
        print(f"[steering_judge] dry-run wrote inputs: {inputs_path}")
        return

    client = _make_client(args)
    outputs: List[Dict[str, Any]] = []
    iterator = payloads
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(payloads, desc="steering_judge", unit="pair", dynamic_ncols=True)

    for payload in iterator:
        t0 = time.perf_counter()
        try:
            judged = _call_judge_with_retries(client, args, payload)
            judged["_judge_error"] = ""
        except (ValidationError, Exception) as exc:
            judged = {
                "hint_was_actionable": False,
                "next_query_followed_hint": False,
                "next_think_revised_error": False,
                "same_error_repeated": False,
                "steering_success": False,
                "steering_error_type": None,
                "_judge_error": repr(exc),
            }
        judged.update({
            "source": payload.get("source"),
            "case_index": payload["case_index"],
            "turn_number": payload["turn_number"],
            "next_turn_number": payload["next_turn_number"],
            "dataset": payload.get("dataset"),
            "uid": payload.get("uid"),
            "question": payload.get("question"),
            "gold_answer": payload.get("gold_answer"),
            "predicted_answer": payload.get("predicted_answer"),
            "query": payload.get("query"),
            "next_query": payload.get("next_query"),
            "judge_model": args.model,
            "judge_latency_sec": round(time.perf_counter() - t0, 6),
        })
        outputs.append(judged)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_summarize(outputs), f, indent=2, ensure_ascii=False)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    summary = _summarize(outputs)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[steering_judge] wrote results: {results_path}")
    print(f"[steering_judge] wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
