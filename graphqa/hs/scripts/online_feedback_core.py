"""HS-local online feedback loop and prompt formatter.

This file is intentionally separate from ``graphqa/scripts/run_online_corrector.py``
so prompt experiments can change quickly without modifying the original
fallback corrector baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graphqa.data import Triple
from graphqa.data.schema import is_unknown
from graphqa.scripts.run_online_corrector import (
    CorrectorOutputs,
    _expected_answer_type,
    _extract_latest_think,
    _fmt_cosine,
    _is_think_self_paraphrase_fields,
    _per_q_alignment,
    _relation_equivalent,
    _short_triple_summary,
    _strip_vg_hint_from_query,
)
from graphqa.scripts.run_online_eval import (
    GraphGenResult,
    ExtractorPool,
    _normalize_text,
    _split_question_triples,
    _truncate_words,
    make_graph_input,
    triples_to_raw,
)
from search_r1 import SearchR1Inference


def _compact_statement(text: Any, *, max_words: int = 18) -> str:
    s = " ".join(str(text or "").replace("[SEP]", " ").split())
    if not s:
        return ""
    return _truncate_words(s, max_words)


def _suggest_search_query(q: Triple) -> str:
    head = (q.head or "").strip()
    rel = " ".join((q.relation or "").lower().split())
    tail = (q.tail or "").strip()
    known = tail if tail and not is_unknown(tail) else head if head and not is_unknown(head) else ""
    known = known.replace("film ", "").replace("movie ", "").strip()
    if not known:
        return "exact title/entity plus the missing relation"
    if "director" in rel:
        return f'"{known}" film director'
    if "from" in rel or "country" in rel or "nationality" in rel:
        return f'"{known}" nationality country'
    return f'"{known}" {q.relation}'.strip()


def _strip_known_prefix(value: Any) -> str:
    text = " ".join(str(value or "").replace('"', "").split())
    lowered = text.lower()
    for prefix in ("film ", "movie "):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _values_match(a: Any, b: Any) -> bool:
    na = _normalize_text(str(a or ""))
    nb = _normalize_text(str(b or ""))
    return bool(na and nb and (na == nb or na in nb or nb in na))


def _answer_for_requirement(q: Triple, cand: Dict[str, Any]) -> str:
    """Return the candidate value that fills the unknown side of a requirement."""
    if not cand:
        return ""
    q_head = q.head or ""
    q_tail = q.tail or ""
    c_head = cand.get("candidate_head") or ""
    c_tail = cand.get("candidate_tail") or ""
    c_rel = " ".join(str(cand.get("candidate_relation") or "").lower().split())

    if q.head_unknown and q_tail and not is_unknown(q_tail):
        known = _strip_known_prefix(q_tail)
        if _values_match(known, c_tail):
            return str(c_head).strip()
        if _values_match(known, c_head):
            return str(c_tail).strip()
        if "directed by" in c_rel:
            return str(c_tail).strip()
    if q.tail_unknown and q_head and not is_unknown(q_head):
        known = _strip_known_prefix(q_head)
        if _values_match(known, c_head):
            return str(c_tail).strip()
        if _values_match(known, c_tail):
            return str(c_head).strip()
    return ""


def _answer_from_think_text(q: Triple, think_text: str) -> str:
    if not think_text or not q.head_unknown or not q.tail or is_unknown(q.tail):
        return ""
    if "director" not in " ".join((q.relation or "").lower().split()):
        return ""
    title = _strip_known_prefix(q.tail)
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
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        if groups:
            ans = " ".join(groups[-1].split())
            if ans and not is_unknown(ans) and not _values_match(ans, title):
                return ans
    return ""


def _unresolved_requirement_labels(
    q_triples: Sequence[Triple],
    alignment_rows: Sequence[Dict[str, Any]],
    *,
    threshold: float,
) -> List[str]:
    rows = {row.get("q_idx"): row for row in alignment_rows}
    labels: List[str] = []
    for q_idx, q in enumerate(q_triples):
        if not ((not q.head_unknown and (q.head or "").strip()) or (not q.tail_unknown and (q.tail or "").strip())):
            continue
        row = rows.get(q_idx) or {}
        d = row.get("doc_match") or {}
        d_cos = d.get("min_field_cosine")
        d_val = float(d_cos) if d_cos is not None else None
        doc_pass = bool(d and d.get("candidate_text") and d_val is not None and d_val >= threshold)
        if doc_pass:
            continue
        anchor = q.tail if q.tail and not is_unknown(q.tail) else q.head if q.head and not is_unknown(q.head) else ""
        anchor_label = _strip_known_prefix(anchor)
        if anchor_label and "director" in (q.relation or "").lower():
            labels.append(f"{anchor_label} director")
        else:
            labels.append(_short_triple_summary(q))
    return labels


def format_hs_feedback_block(
    turn: int,
    alignment_rows: Sequence[Dict[str, Any]],
    *,
    threshold: float,
    expected_answer_type: str,
    q_triples_in_order: Sequence[Triple],
    last_block_summary: Optional[str] = None,
    latest_think_text: str = "",
    style: str = "score",
) -> Tuple[str, str]:
    """Render the HS online feedback block injected into the next SearchR1 turn."""
    if not alignment_rows:
        block = (
            "\n<vg_hint turn=\"{t}\" style=\"natural\" expects=\"{at}\">\n"
            "Evidence check. Retrieved evidence is not clear yet, so keep any missing facts tentative.\n"
            "</vg_hint>\n\n"
        ).format(t=turn, at=expected_answer_type)
        return block, "empty"

    lines: List[str] = []
    natural_styles = {
        "natural",
        "natural_brief",
        "natural_check",
        "natural_guard",
        "natural_mismatch",
        "natural_mismatch_query",
        "natural_revision",
        "revision_brief",
        "correction_brief",
        "repair_brief",
    }
    style = style if style in (natural_styles | {"soft", "score", "strict", "checklist", "next_query"}) else "natural_mismatch"
    if style in natural_styles:
        lines.append(f'<vg_hint turn="{turn}" style="{style}" expects="{expected_answer_type}">')
    else:
        lines.append(f'<vg_hint turn="{turn}" style="{style}" expects="{expected_answer_type}" threshold={threshold:.2f}>')
    if style == "natural":
        lines.append(
            "Evidence check, not an instruction. Use your own reasoning, but treat facts as tentative unless retrieved text supports them."
        )
        if expected_answer_type == "yes or no":
            lines.append(
                "For a yes/no comparison, verify each compared entity and its country/nationality from retrieved evidence before finalizing."
            )
    elif style == "natural_brief":
        lines.append(
            "Evidence check only: keep unsupported facts tentative, and continue with your own reasoning."
        )
    elif style == "natural_check":
        lines.append(
            "Grounding check. Use these as evidence states, not as answer hints."
        )
        if expected_answer_type == "yes or no":
            lines.append("For yes/no, both sides of the comparison need grounded evidence.")
    elif style == "natural_guard":
        lines.append(
            "Reasoning guardrail: if a fact was only guessed or inferred from an unrelated passage, do not carry it forward as found."
        )
        lines.append("Prefer one more exact verification step over finalizing with an ungrounded entity.")
    elif style == "revision_brief":
        lines.append(
            "Revise unsupported facts. If docs do not prove a claim, say NOT FOUND YET; do not say it was found."
        )
    elif style == "correction_brief":
        lines.append(
            "Evidence correction only: unsupported facts should stay tentative; do not carry them forward as found."
        )
    elif style == "repair_brief":
        lines.append("Evidence repair only: revise unsupported or contradicted facts before continuing.")
    elif style in {"natural_mismatch", "natural_mismatch_query", "natural_revision"}:
        lines.append(
            "Evidence check, not an answer hint. Keep your own reasoning, but do not reuse names from a retrieved passage unless the passage connects them to the requested fact."
        )
        if expected_answer_type == "yes or no":
            lines.append("For a yes/no comparison, both compared entities must be grounded before comparing countries/nationalities.")
        if style == "natural_revision":
            lines.append(
                "If your previous thought said a fact was found but the retrieved text does not support it, explicitly revise that thought: say it is not found yet, then verify before using it."
            )
    elif style == "soft":
        lines.append(
            "Evidence status only. Use your own reasoning; if an important requirement has weak doc support, consider verifying it before relying on it."
        )
    elif style == "strict":
        lines.append(
            "RULE: Treat doc below threshold as NOT FOUND. Do not write 'I found out' for a requirement until doc is strong. "
            "If doc is weak, your next step must be a targeted search, not an answer."
        )
    elif style == "next_query":
        lines.append(
            "RULE: Follow NEXT_SEARCH_QUERY exactly for the next search turn. Do not answer and do not assert the missing fact first."
        )
    elif style == "checklist":
        lines.append(
            "Before answering, complete every unchecked requirement below with retrieved evidence. "
            "A model claim alone does not complete a requirement."
        )
    elif turn == 0:
        lines.append(
            "[How to read] doc is retrieved-evidence support; think is only the model's current claim. "
            "Do not assert facts when doc is weak. Use this hint to choose the next search."
        )

    rid_to_row = {row["q_idx"]: row for row in alignment_rows}
    summary_parts: List[str] = []
    next_query: Optional[str] = None
    next_query_reason: str = ""

    for q_idx, q in enumerate(q_triples_in_order):
        if not ((not q.head_unknown and (q.head or "").strip()) or (not q.tail_unknown and (q.tail or "").strip())):
            continue
        row = rid_to_row.get(q_idx)
        if row is None:
            continue
        d = row.get("doc_match") or {}
        t = row.get("think_match") or {}

        d_cos = d.get("min_field_cosine")
        t_cos = t.get("min_field_cosine")
        d_val = float(d_cos) if d_cos is not None else None
        doc_has_candidate = bool(d and d.get("candidate_text") and d_val is not None)
        doc_pass = bool(doc_has_candidate and d_val >= threshold)

        if t and t.get("candidate_text"):
            if not doc_pass:
                think_label = "claim-only(untrusted)"
            elif _is_think_self_paraphrase_fields(
                q.head or "", q.relation or "", q.tail or "",
                t.get("candidate_head") or "",
                t.get("candidate_relation") or "",
                t.get("candidate_tail") or "",
            ):
                think_label = "N/A(paraphrase)"
            else:
                think_label = _fmt_cosine(t_cos)
        else:
            think_label = "N/A(no-triple)"

        doc_extra = ""
        if doc_has_candidate:
            q_rel = q.relation or ""
            c_rel = d.get("candidate_relation", "") or ""
            if (d_val < threshold and _relation_equivalent(q_rel, c_rel)
                    and d_val >= max(threshold - 0.15, 0.30)):
                doc_extra = " (paraphrase-OK but still weak)"

        rid = f"R{q_idx + 1}"
        req_text = _short_triple_summary(q)
        doc_part = _fmt_cosine(d_val) + doc_extra if d_val is not None else "N/A"
        suggested_query = _suggest_search_query(q)
        anchor = ""
        if q.tail and not is_unknown(q.tail):
            anchor = q.tail
        elif q.head and not is_unknown(q.head):
            anchor = q.head
        anchor_clean = " ".join(anchor.replace('"', "").split())
        anchor_label = _strip_known_prefix(anchor_clean)
        req_label = req_text
        if anchor_label and "director" in (q.relation or "").lower():
            req_label = f"{anchor_label} director"
        if not doc_pass and next_query is None:
            next_query = suggested_query
            next_query_reason = f"{rid} {req_text}"

        if style == "natural":
            if doc_pass:
                if think_label.startswith("N/A"):
                    status = "retrieved evidence looks relevant; connect it explicitly to your reasoning before relying on it"
                else:
                    status = "retrieved evidence looks relevant; use it only if the document text matches the fact"
            elif t and t.get("candidate_text"):
                status = "your reasoning mentions this, but retrieved evidence has not grounded it yet; keep it tentative"
            elif doc_has_candidate:
                status = "retrieved text is only loosely related; verify with a more exact title/entity if this fact matters"
            else:
                status = "still needs retrieved evidence"
            lines.append(f'{rid} {req_text}: {status}')
        elif style == "natural_brief":
            if doc_pass:
                status = "grounded enough to use carefully"
            elif t and t.get("candidate_text"):
                status = "mentioned in reasoning but not grounded yet"
            else:
                status = "needs evidence"
            lines.append(f'{rid}: {status}')
        elif style == "natural_check":
            if doc_pass:
                status = "Grounded"
            elif t and t.get("candidate_text"):
                status = "Tentative claim"
            elif doc_has_candidate:
                status = "Related document only"
            else:
                status = "Missing evidence"
            lines.append(f'{rid} {status}: {req_text}')
        elif style == "natural_guard":
            if doc_pass:
                status = "safe to use if the retrieved wording matches"
            elif t and t.get("candidate_text"):
                status = "do not treat this as found yet"
            else:
                status = "verify before using"
            lines.append(f'{rid} {req_text}: {status}')
        elif style == "revision_brief":
            if doc_pass:
                doc_answer = _answer_for_requirement(q, d)
                think_answer = _answer_for_requirement(q, t) or _answer_from_think_text(q, latest_think_text)
                if (
                    doc_answer
                    and think_answer
                    and not _values_match(doc_answer, think_answer)
                    and not is_unknown(doc_answer)
                    and not is_unknown(think_answer)
                ):
                    lines.append(f'{rid} CONFLICT: {req_label}; docs support {doc_answer}, current thought says {think_answer}')
                else:
                    lines.append(f'{rid} FOUND: {req_label}')
            elif t and t.get("candidate_text"):
                lines.append(f'{rid} NOT FOUND YET: {req_label}; do not use a person name for this slot yet')
            else:
                lines.append(f'{rid} NOT FOUND YET: {req_label}')
        elif style == "correction_brief":
            if doc_pass:
                doc_answer = _answer_for_requirement(q, d)
                think_answer = _answer_for_requirement(q, t) or _answer_from_think_text(q, latest_think_text)
                if (
                    doc_answer
                    and think_answer
                    and not _values_match(doc_answer, think_answer)
                    and not is_unknown(doc_answer)
                    and not is_unknown(think_answer)
                ):
                    lines.append(
                        f'{rid} CONFLICT: docs support {doc_answer}; current thought says {think_answer}'
                    )
                else:
                    lines.append(f'{rid} FOUND: {req_text}')
            else:
                suffix = "; current thought unsupported" if t and t.get("candidate_text") else "; retrieved evidence does not ground it"
                lines.append(f'{rid} NOT FOUND YET: {req_label}{suffix}')
        elif style == "repair_brief":
            doc_answer = _answer_for_requirement(q, d)
            think_answer = _answer_for_requirement(q, t) or _answer_from_think_text(q, latest_think_text)
            if (
                doc_pass
                and doc_answer
                and think_answer
                and not _values_match(doc_answer, think_answer)
                and not is_unknown(doc_answer)
                and not is_unknown(think_answer)
            ):
                lines.append(f'{rid} CONFLICT: {req_label}; docs support {doc_answer}, current thought says {think_answer}')
            elif not doc_pass and think_answer and not is_unknown(think_answer):
                lines.append(f'{rid} UNSUPPORTED: current thought says {think_answer}; docs do not support {req_label}')
            elif not doc_pass:
                lines.append(f'{rid} NOT FOUND YET: {req_label}')
            else:
                lines.append(f'{rid} FOUND: {req_label}')
        elif style in {"natural_mismatch", "natural_mismatch_query", "natural_revision"}:
            if doc_pass:
                cand = _compact_statement(d.get("candidate_text"))
                status = "grounded by retrieved evidence"
                if cand:
                    status += f"; retrieved statement: {cand}"
            elif doc_has_candidate:
                cand = _compact_statement(d.get("candidate_text"))
                cand_l = str(d.get("candidate_text") or "").lower()
                status = "nearest retrieved statement does not establish the requested fact"
                if cand:
                    status += f"; nearest statement: {cand}"
                if anchor_clean and anchor_clean.lower() not in cand_l:
                    status += f"; it does not mention {anchor_clean}"
                status += "; do not use names from that statement for this requirement"
                if "director" in (q.relation or "").lower():
                    status += "; verify the exact film director before moving to country/nationality"
                    if style == "natural_mismatch_query":
                        status += f"; a safer verification query would include the exact title, e.g. {suggested_query}"
                if t and t.get("candidate_text"):
                    status += "; treat the current reasoning claim as ungrounded"
                    if style == "natural_revision":
                        claim = _compact_statement(t.get("candidate_text"))
                        status += "; if your last thought said this was found, revise it to not found yet"
                        if claim:
                            status += f"; unsupported thought claim: {claim}"
            elif t and t.get("candidate_text"):
                claim = _compact_statement(t.get("candidate_text"))
                status = "current reasoning mentions this, but no retrieved statement grounds it"
                if claim:
                    status += f"; current claim: {claim}"
                if style == "natural_mismatch_query" and "director" in (q.relation or "").lower():
                    status += f"; a safer verification query would include the exact title, e.g. {suggested_query}"
                if style == "natural_revision":
                    status += "; if your last thought said this was found, revise it to not found yet before searching again"
            else:
                status = "no retrieved statement grounds this yet"
            lines.append(f'{rid} {req_text}: {status}')
        elif style == "soft":
            status = "supported" if doc_pass else "weak-doc"
            lines.append(f'{rid} {req_text}: {status}; doc={doc_part}; think={think_label}')
        elif style == "next_query":
            status = "VERIFIED" if doc_pass else "MISSING"
            lines.append(f'{rid} {status}: {req_text}; doc={doc_part}; think={think_label}')
        elif style == "strict":
            if not doc_pass:
                lines.append(
                    f'{rid} INVALID/UNVERIFIED: {req_text}. doc={doc_part}; think={think_label}. '
                    f'Do NOT assert this fact. Next search: {suggested_query}'
                )
            elif think_label.startswith("N/A"):
                lines.append(
                    f'{rid} EVIDENCE FOUND: {req_text}. doc={doc_part}; restate the evidence precisely before answering.'
                )
            else:
                lines.append(f'{rid} VERIFIED: {req_text}. doc={doc_part}; think={think_label}')
        elif style == "checklist":
            mark = "[x]" if doc_pass and not think_label.startswith("N/A") else "[ ]"
            if mark == "[ ]":
                lines.append(
                    f'{mark} {rid} {req_text}: doc={doc_part}; think={think_label}; required next search: {suggested_query}'
                )
            else:
                lines.append(f'{mark} {rid} {req_text}: doc={doc_part}; think={think_label}')
        else:
            action = ""
            if not doc_pass:
                action = " -> do not assert yet; search exact title/entity + relation"
            elif think_label.startswith("N/A"):
                action = " -> evidence exists; restate it precisely before answering"
            lines.append(f'{rid} {req_text}: doc={doc_part}  think={think_label}{action}')
        summary_parts.append(f'{rid}:doc={doc_part};think={think_label}')

    block_summary = "|".join(summary_parts)
    if style == "next_query":
        if next_query:
            lines.append(f'NEXT_SEARCH_QUERY: {next_query}')
            lines.append(f'REASON: {next_query_reason} is still missing retrieved evidence.')
        else:
            lines.append('NEXT_ACTION: all listed requirements have evidence; verify remaining country/nationality facts before answering.')
    if style not in natural_styles and last_block_summary and last_block_summary == block_summary:
        lines.append(
            "[Note] Scores did not improve. Try a different query: exact title, original title, year, or country qualifier."
        )
    lines.append("</vg_hint>")
    return "\n\n" + "\n".join(lines) + "\n\n", block_summary


def _match_passes(match: Dict[str, Any], threshold: float) -> bool:
    if not match or not match.get("candidate_text"):
        return False
    val = match.get("min_field_cosine")
    try:
        return float(val) >= float(threshold)
    except Exception:
        return False


def _requirement_label(q: Triple) -> str:
    req_text = _short_triple_summary(q)
    anchor = q.tail if q.tail and not is_unknown(q.tail) else q.head if q.head and not is_unknown(q.head) else ""
    anchor_label = _strip_known_prefix(anchor)
    if anchor_label and "director" in (q.relation or "").lower():
        return f"{anchor_label} director"
    return req_text


def _query_targets_requirement(q: Triple, query: str) -> str:
    anchor = q.tail if q.tail and not is_unknown(q.tail) else q.head if q.head and not is_unknown(q.head) else ""
    anchor = _strip_known_prefix(anchor)
    terms = [w for w in re.findall(r"[A-Za-z0-9]+", anchor.lower()) if len(w) > 2]
    q_l = " ".join((query or "").lower().split())
    if terms and all(t in q_l for t in terms):
        return "targets_unresolved_requirement"
    if terms and any(t in q_l for t in terms):
        return "partially_targets_requirement"
    return "query_may_miss_requirement"


def build_question_only_labels(q_triples: Sequence[Triple]) -> List[Dict[str, Any]]:
    """Represent decomposed question requirements without D/T verification."""
    labels: List[Dict[str, Any]] = []
    for q_idx, q in enumerate(q_triples):
        if not ((not q.head_unknown and (q.head or "").strip()) or (not q.tail_unknown and (q.tail or "").strip())):
            continue
        labels.append({
            "q_idx": int(q_idx),
            "rid": f"R{q_idx + 1}",
            "requirement": _requirement_label(q),
            "q_text": _short_triple_summary(q),
            "q_d": "not_checked",
            "q_t": "not_checked",
            "d_t": "question_only",
            "action": "use_as_question_checklist",
            "query_q": "not_checked",
            "doc_value": "",
            "think_value": "",
            "doc_score": None,
            "think_score": None,
            "doc_candidate": "",
            "think_candidate": "",
            "suggested_query": _suggest_search_query(q),
        })
    return labels


def build_verification_labels(
    alignment_rows: Sequence[Dict[str, Any]],
    *,
    threshold: float,
    q_triples_in_order: Sequence[Triple],
    latest_think_text: str = "",
    query: str = "",
) -> List[Dict[str, Any]]:
    """Convert raw Q-D/Q-T alignment rows into stable verifier labels.

    The labels are an internal representation: they make the verifier decision
    explicit before any prompt text is chosen.
    """
    rid_to_row = {row.get("q_idx"): row for row in alignment_rows}
    labels: List[Dict[str, Any]] = []
    for q_idx, q in enumerate(q_triples_in_order):
        if not ((not q.head_unknown and (q.head or "").strip()) or (not q.tail_unknown and (q.tail or "").strip())):
            continue
        row = rid_to_row.get(q_idx)
        if row is None:
            continue
        d = row.get("doc_match") or {}
        t = row.get("think_match") or {}
        doc_pass = _match_passes(d, threshold)
        think_pass = _match_passes(t, threshold)
        doc_answer = _answer_for_requirement(q, d)
        think_answer = _answer_for_requirement(q, t) or _answer_from_think_text(q, latest_think_text)

        q_d = "support" if doc_pass else "no_support"
        if (not doc_pass) and d and d.get("candidate_text"):
            q_d = "weak_or_irrelevant_doc"

        if think_answer and not is_unknown(think_answer):
            q_t = "claims_value"
        elif t and t.get("candidate_text") and think_pass:
            if _is_think_self_paraphrase_fields(
                q.head or "", q.relation or "", q.tail or "",
                t.get("candidate_head") or "",
                t.get("candidate_relation") or "",
                t.get("candidate_tail") or "",
            ):
                q_t = "self_paraphrase"
            else:
                q_t = "relevant_think"
        elif t and t.get("candidate_text"):
            q_t = "weak_relevant_think"
        else:
            q_t = "no_relevant_think"

        if (
            doc_pass
            and doc_answer
            and think_answer
            and not is_unknown(doc_answer)
            and not is_unknown(think_answer)
            and not _values_match(doc_answer, think_answer)
        ):
            d_t = "conflict"
        elif (not doc_pass) and think_answer and not is_unknown(think_answer):
            d_t = "unsupported_think_claim"
        elif doc_pass and (think_pass or think_answer):
            d_t = "aligned"
        elif doc_pass:
            d_t = "evidence_not_used"
        elif think_pass or think_answer:
            d_t = "think_only"
        else:
            d_t = "missing_both"

        suggested_query = _suggest_search_query(q)
        if d_t == "conflict":
            action = "revise_conflict"
        elif d_t == "unsupported_think_claim":
            action = "do_not_carry_forward"
        elif q_d != "support":
            action = "search_exact_title_relation"
        elif d_t == "evidence_not_used":
            action = "use_grounded_evidence"
        else:
            action = "can_use"

        labels.append({
            "q_idx": int(q_idx),
            "rid": f"R{q_idx + 1}",
            "requirement": _requirement_label(q),
            "q_text": _short_triple_summary(q),
            "q_d": q_d,
            "q_t": q_t,
            "d_t": d_t,
            "action": action,
            "query_q": "already_supported" if doc_pass else _query_targets_requirement(q, query),
            "doc_value": doc_answer,
            "think_value": think_answer,
            "doc_score": d.get("min_field_cosine"),
            "think_score": t.get("min_field_cosine"),
            "doc_candidate": d.get("candidate_text") or "",
            "think_candidate": t.get("candidate_text") or "",
            "suggested_query": suggested_query,
        })
    return labels


def format_feedback_from_labels(
    *,
    turn: int,
    labels: Sequence[Dict[str, Any]],
    expected_answer_type: str,
    style: str,
) -> Tuple[str, str]:
    """Render vg_hint from precomputed verification labels."""
    style = style or "repair_brief"
    lines = [f'<vg_hint turn="{turn}" style="labels:{style}" expects="{expected_answer_type}">']
    explicit_styles = {"explicit_requirements", "explicit_requirements_short", "explicit_requirements_soft"}
    query_styles = {"natural_mismatch_query", "next_query", "natural_mismatch_query_norid", "natural_mismatch_query_short"}

    if style == "explicit_requirements":
        lines.append(
            "Use this as a checklist for the original question. "
            "Each item is one required fact from the question, not a new question. "
            "Do not rely on a value unless retrieved evidence directly supports that exact item."
        )
    elif style == "explicit_requirements_short":
        lines.append(
            "Checklist for the original question. A value is usable only when retrieved documents support that exact requirement."
        )
    elif style == "explicit_requirements_soft":
        lines.append(
            "Evidence notes for the original question. Keep unsupported values tentative and verify important missing links."
        )
    elif style == "requirement_brief":
        lines.append(
            "Evidence checklist for the original question. "
            "Use it as a reminder, not as a new question. "
            "Only use a value when the retrieved documents directly support that requirement."
        )
    elif style in {"soft", "natural", "natural_brief", "natural_mismatch"}:
        lines.append("Evidence check only. Keep unsupported facts tentative and continue with your own reasoning.")
    elif style in query_styles:
        lines.append("Evidence check only. Prefer an exact verification search when an important link is not grounded.")
    else:
        lines.append("Evidence repair only. Revise unsupported or contradicted facts before continuing.")

    summary_parts: List[str] = []
    for lab in labels:
        rid = lab.get("rid") or f"R{int(lab.get('q_idx', 0)) + 1}"
        req = lab.get("requirement") or lab.get("q_text") or "requirement"
        d_t = lab.get("d_t")
        action = lab.get("action")
        think_value = str(lab.get("think_value") or "").strip()
        doc_value = str(lab.get("doc_value") or "").strip()
        suggested_query = str(lab.get("suggested_query") or "").strip()

        item_name = f"Question requirement {int(lab.get('q_idx', 0)) + 1}"

        if style in explicit_styles:
            prefix = item_name if style != "explicit_requirements_short" else "Requirement"
            lines.append(f"{prefix}: {req}.")
            if d_t == "question_only":
                lines.append("Status: requirement from the original question. Use it as a checklist while searching.")
            elif d_t == "conflict":
                if style == "explicit_requirements_soft":
                    lines.append(
                        f"Status: conflict. Docs support {doc_value or 'a different value'}; current reasoning says {think_value or 'another value'}. Verify before relying on it."
                    )
                else:
                    lines.append(
                        f"Status: conflict. Retrieved evidence supports {doc_value or 'a different value'}, "
                        f"but current reasoning says {think_value or 'another value'}."
                    )
                    lines.append(
                        f"What to do next: do not use {think_value or 'the current value'} for this requirement. "
                        "Re-check the exact title/relation before using this fact."
                    )
            elif d_t == "unsupported_think_claim":
                if style == "explicit_requirements_soft":
                    lines.append(
                        f"Status: unsupported. Current reasoning says {think_value}, but retrieved evidence does not support this requirement yet."
                    )
                else:
                    lines.append(
                        f"Status: unsupported. Current reasoning says {think_value}, "
                        "but retrieved evidence does not support this requirement."
                    )
                    lines.append(
                        f"What to do next: do not carry forward {think_value}; "
                        "search for direct evidence for this requirement."
                    )
            elif lab.get("q_d") == "support":
                lines.append("Status: found in retrieved evidence.")
                if doc_value:
                    lines.append(f"Evidence value: {doc_value}.")
            else:
                lines.append("Status: not found yet in retrieved evidence.")
                if style != "explicit_requirements_soft":
                    lines.append("What to do next: search this exact requirement before using any guessed value.")
            if action in {"search_exact_title_relation", "do_not_carry_forward", "revise_conflict"} and suggested_query:
                if style == "explicit_requirements_short":
                    lines.append(f"Search: {suggested_query}.")
                else:
                    lines.append(f"Suggested next search: {suggested_query}.")
        elif style == "requirement_brief":
            if d_t == "question_only":
                msg = f"- {req}: requirement from the original question. Keep this slot in view while searching."
            elif d_t == "conflict":
                msg = (
                    f"- {req}: CONFLICT. Retrieved docs support {doc_value or 'a different value'}, "
                    f"but current reasoning uses {think_value or 'another value'}. "
                    f"Do not rely on {think_value or 'that value'} unless an exact document supports it."
                )
            elif d_t == "unsupported_think_claim":
                msg = (
                    f"- {req}: NOT VERIFIED. Current reasoning uses {think_value}, "
                    "but retrieved docs do not directly support it. Treat it as tentative."
                )
            elif lab.get("q_d") == "support":
                msg = f"- {req}: VERIFIED in retrieved docs"
                if doc_value:
                    msg += f" as {doc_value}"
                msg += "."
            else:
                msg = f"- {req}: NOT FOUND YET in retrieved docs. Do not fill this slot with a guessed value."
            if action in {"search_exact_title_relation", "do_not_carry_forward", "revise_conflict"} and suggested_query:
                msg += f" If this fact is still needed, search exactly: {suggested_query}."
        elif style in {"soft", "natural", "natural_brief", "natural_mismatch"}:
            if d_t == "question_only":
                msg = f"{rid} {req}: requirement from the original question."
            elif d_t == "conflict":
                msg = f"{rid} {req}: retrieved evidence and current reasoning disagree; verify this link before using it."
            elif d_t == "unsupported_think_claim":
                msg = f"{rid} {req}: current reasoning mentions {think_value}, but retrieved evidence has not grounded that link yet."
            elif lab.get("q_d") == "support":
                msg = f"{rid} {req}: retrieved evidence appears available; use it only if the wording matches the fact."
            else:
                msg = f"{rid} {req}: not found yet in retrieved evidence."
        elif style in query_styles:
            if d_t == "question_only":
                msg = f"{req}: requirement from the original question."
            elif d_t == "conflict":
                msg = f"{req}: docs support {doc_value}, but current reasoning says {think_value}; verify before moving on."
            elif d_t == "unsupported_think_claim":
                msg = f"{req}: current reasoning says {think_value}, but docs do not support that link."
            elif lab.get("q_d") == "support":
                msg = f"{req}: found in retrieved evidence."
            else:
                msg = f"{req}: not found yet."
            if style == "natural_mismatch_query":
                msg = f"{rid} {msg}"
            if action in {"search_exact_title_relation", "do_not_carry_forward", "revise_conflict"} and suggested_query:
                if style == "natural_mismatch_query_short":
                    msg += f" Search: {suggested_query}."
                else:
                    msg += f" A safer search would use the exact title and relation, e.g. {suggested_query}."
        else:
            if d_t == "question_only":
                msg = f"{rid} REQUIREMENT: {req}."
            elif d_t == "conflict":
                msg = f"{rid} CONFLICT: {req}; docs support {doc_value or 'a different value'}, current thought says {think_value or 'another value'}."
            elif d_t == "unsupported_think_claim":
                msg = f"{rid} UNSUPPORTED: current thought says {think_value}; docs do not support {req}."
            elif lab.get("q_d") == "support":
                msg = f"{rid} FOUND: {req}."
            else:
                msg = f"{rid} NOT FOUND YET: {req}."
        if style not in explicit_styles:
            lines.append(msg)
        summary_parts.append(f"{rid}:{lab.get('q_d')}:{lab.get('q_t')}:{lab.get('d_t')}:{lab.get('action')}")

    lines.append("</vg_hint>")
    return "\n\n" + "\n".join(lines) + "\n\n", "|".join(summary_parts)


def select_feedback_labels(
    labels: Sequence[Dict[str, Any]],
    *,
    trigger: str,
) -> List[Dict[str, Any]]:
    """Filter verifier labels before turning them into model-facing feedback.

    The full label list is still recorded for analysis. This function only
    decides what is strong enough to inject into the next SearchR1 turn.
    """
    trigger = (trigger or "all").strip().lower()
    selected: List[Dict[str, Any]] = []
    for lab in labels:
        q_d = str(lab.get("q_d") or "")
        d_t = str(lab.get("d_t") or "")
        action = str(lab.get("action") or "")
        query_q = str(lab.get("query_q") or "")
        has_value = bool(str(lab.get("think_value") or "").strip())

        is_strict = d_t in {"unsupported_think_claim", "conflict"} or action in {"do_not_carry_forward", "revise_conflict"}
        is_missing = q_d in {"no_support", "weak_or_irrelevant_doc"} or action == "search_exact_title_relation"
        is_actionable = action not in {"can_use", "use_grounded_evidence"}
        query_already_targets = query_q in {"targets_unresolved_requirement", "partially_targets_requirement"}

        keep = False
        if trigger in {"all", "always"}:
            keep = True
        elif trigger == "non_support":
            keep = q_d != "support" or is_strict
        elif trigger == "actionable":
            keep = is_actionable
        elif trigger == "strict":
            keep = is_strict
        elif trigger == "strict_query":
            keep = is_strict and not query_already_targets
        elif trigger == "strict_or_missing_query":
            keep = is_strict or (is_missing and not query_already_targets)
        elif trigger == "strict_query_or_missing_query":
            keep = (is_strict or is_missing) and not query_already_targets
        elif trigger == "unsupported_only":
            keep = d_t == "unsupported_think_claim"
        elif trigger == "conflict_only":
            keep = d_t == "conflict"
        elif trigger == "claim_only":
            keep = is_strict or (is_missing and has_value)
        elif trigger == "missing_query":
            keep = is_missing and not query_already_targets
        elif trigger in {"off", "none", "never"}:
            keep = False
        else:
            keep = True
        if keep:
            selected.append(dict(lab))
    return selected


def run_hs_online_feedback(
    *,
    question: str,
    q_triples: Optional[List[Triple]] = None,
    q_future: Optional[Future] = None,
    pool: ExtractorPool,
    searchr1: SearchR1Inference,
    encoder,
    args: argparse.Namespace,
) -> CorrectorOutputs:
    """Run SearchR1 once, injecting HS-local feedback after every search turn."""
    feedback_source = str(getattr(args, "feedback_source", "full") or "full")
    question_only = feedback_source == "question_only"
    d_extractor = None if question_only else pool.get(args.document_model)
    t_extractor = None if question_only else pool.get(args.think_model)

    doc_cache: Dict[str, GraphGenResult] = {}
    accumulated_doc_triples: List[Triple] = []
    seen_doc_keys: set = set()
    turn_records: List[Dict[str, Any]] = []
    last_block_summary: Optional[str] = None
    expected_answer_type = _expected_answer_type(question)
    q_triples_resolved: Optional[List[Triple]] = list(q_triples or []) if q_triples is not None else None

    def _get_q_triples() -> List[Triple]:
        nonlocal q_triples_resolved
        if q_triples_resolved is None:
            if q_future is None:
                q_triples_resolved = []
            else:
                q_result = q_future.result()
                if isinstance(q_result, tuple):
                    q_result = q_result[0]
                q_triples_resolved = _split_question_triples(list(getattr(q_result, "triples", []) or []))[0]
        return q_triples_resolved

    def _record_doc_result(doc: str, result: GraphGenResult) -> None:
        doc_cache[doc] = result
        for tri in result.triples:
            key = (
                " ".join((tri.head or "").split()),
                " ".join((tri.relation or "").split()),
                " ".join((tri.tail or "").split()),
            )
            if key in seen_doc_keys:
                continue
            seen_doc_keys.add(key)
            accumulated_doc_triples.append(tri)

    def _generate_doc_batch(docs: List[str]) -> List[GraphGenResult]:
        if d_extractor is None:
            return []
        inputs = [
            make_graph_input(
                "document",
                document=_truncate_words(doc, int(args.doc_max_words)),
            )
            for doc in docs
        ]
        generate_many = getattr(d_extractor, "generate_many", None)
        if callable(generate_many):
            return list(generate_many("document", inputs))
        return [d_extractor.generate("document", user_content) for user_content in inputs]

    def _add_docs(docs: List[str]) -> List[Triple]:
        missing_docs: List[str] = []
        seen_missing = set()
        for doc in docs:
            if doc in doc_cache or doc in seen_missing:
                continue
            seen_missing.add(doc)
            missing_docs.append(doc)

        if missing_docs:
            for doc, result in zip(missing_docs, _generate_doc_batch(missing_docs)):
                _record_doc_result(doc, result)

        current: List[Triple] = []
        for doc in docs:
            if doc in doc_cache:
                current.extend(doc_cache[doc].triples)
        return current

    def on_turn(event: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal last_block_summary
        observer_t0 = time.perf_counter()
        turn = int(event.get("turn", len(turn_records)))
        raw_query = _normalize_text(str(event.get("query", "") or ""))
        clean_query = _strip_vg_hint_from_query(raw_query)
        docs = list(event.get("search_results", []) or [])[: int(args.max_docs_per_sample)]
        think_text = _extract_latest_think(str(event.get("output_text", "") or ""))

        overlap_executor = getattr(args, "_graph_overlap_executor", None)
        before = len(accumulated_doc_triples)

        def _run_doc_graph() -> Tuple[List[Triple], float]:
            if question_only:
                return [], 0.0
            t0 = time.perf_counter()
            return _add_docs(docs), time.perf_counter() - t0

        def _run_think_graph() -> Tuple[GraphGenResult, float]:
            if question_only or not think_text:
                return GraphGenResult(text="", triples=[]), 0.0
            t0 = time.perf_counter()
            if t_extractor is None:
                return GraphGenResult(text="", triples=[]), 0.0
            result = t_extractor.generate(
                args.think_task,
                make_graph_input(args.think_task, think=think_text, search_query=clean_query),
            )
            return result, time.perf_counter() - t0

        if overlap_executor is not None:
            doc_future = overlap_executor.submit(_run_doc_graph)
            think_future = overlap_executor.submit(_run_think_graph)
            current_doc_triples, doc_graph_sec = doc_future.result()
            think_result, think_graph_sec = think_future.result()
        else:
            current_doc_triples, doc_graph_sec = _run_doc_graph()
            think_result, think_graph_sec = _run_think_graph()

        n_new_doc_triples = len(accumulated_doc_triples) - before

        doc_pool_for_check = accumulated_doc_triples

        alignment_t0 = time.perf_counter()
        if question_only:
            alignment_rows = []
        else:
            alignment_rows = _per_q_alignment(
                _get_q_triples(),
                doc_pool_for_check,
                think_result.triples,
                encoder,
                top_k=int(args.cosine_doc_top_k),
                threshold=float(args.corrector_v2_threshold),
            )
        alignment_sec = time.perf_counter() - alignment_t0

        label_t0 = time.perf_counter()
        if question_only:
            verification_labels = build_question_only_labels(_get_q_triples())
        else:
            verification_labels = build_verification_labels(
                alignment_rows,
                threshold=float(args.corrector_v2_threshold),
                q_triples_in_order=_get_q_triples(),
                latest_think_text=think_text,
                query=clean_query,
            )
        label_sec = time.perf_counter() - label_t0
        feedback_engine = str(getattr(args, "feedback_engine", "legacy") or "legacy")
        feedback_t0 = time.perf_counter()
        feedback_trigger = str(getattr(args, "feedback_trigger", "all") or "all")
        feedback_labels = verification_labels
        feedback_suppressed = False
        if feedback_engine == "labels":
            feedback_labels = (
                list(verification_labels)
                if question_only and feedback_trigger not in {"off", "none", "never"}
                else select_feedback_labels(verification_labels, trigger=feedback_trigger)
            )
            feedback_block, block_summary = format_feedback_from_labels(
                turn=turn,
                labels=feedback_labels,
                expected_answer_type=expected_answer_type,
                style=str(getattr(args, "feedback_style", "repair_brief") or "repair_brief"),
            ) if feedback_labels else ("", "")
            feedback_suppressed = bool(not feedback_labels and verification_labels)
        else:
            feedback_block, block_summary = format_hs_feedback_block(
                turn=turn,
                alignment_rows=alignment_rows,
                threshold=float(args.corrector_v2_threshold),
                expected_answer_type=expected_answer_type,
                q_triples_in_order=_get_q_triples(),
                last_block_summary=last_block_summary,
                latest_think_text=think_text,
                style=str(getattr(args, "feedback_style", "natural") or "natural"),
            )
            if feedback_trigger in {"off", "none", "never"}:
                feedback_block, block_summary = "", ""
                feedback_suppressed = True
        feedback_sec = time.perf_counter() - feedback_t0
        last_block_summary = block_summary
        observer_total_sec = time.perf_counter() - observer_t0

        turn_records.append({
            "turn": turn,
            "query": clean_query,
            "raw_query_pre_sanitize": raw_query,
            "query_was_sanitized": bool(clean_query != raw_query),
            "think_text": think_text,
            "n_docs_this_turn": len(docs),
            "n_new_doc_triples": int(n_new_doc_triples),
            "n_total_doc_triples": len(accumulated_doc_triples),
            "n_current_doc_triples": len(current_doc_triples),
            "n_doc_pool_for_check": len(doc_pool_for_check),
            "current_doc_triples": triples_to_raw(current_doc_triples),
            "doc_pool_for_check_triples": triples_to_raw(doc_pool_for_check),
            "n_think_triples": len(think_result.triples),
            "think_raw_graph": think_result.text,
            "think_triples": triples_to_raw(think_result.triples),
            "alignment": alignment_rows,
            "feedback_engine": feedback_engine,
            "feedback_trigger": feedback_trigger,
            "verification_labels": verification_labels,
            "feedback_labels": feedback_labels,
            "feedback_suppressed": feedback_suppressed,
            "expected_answer_type": expected_answer_type,
            "block_summary": block_summary,
            "injected_feedback": feedback_block,
            "latency": {
                "document_graph_sec": round(doc_graph_sec, 6),
                "think_graph_sec": round(think_graph_sec, 6),
                "alignment_sec": round(alignment_sec, 6),
                "verification_label_sec": round(label_sec, 6),
                "feedback_format_sec": round(feedback_sec, 6),
                "observer_total_sec": round(observer_total_sec, 6),
            },
        })

        action = {
            "prompt_injection": feedback_block,
            "prompt_injection_position": str(getattr(args, "feedback_position", "after_info") or "after_info"),
        }
        if bool(getattr(args, "abstain_on_unresolved", False)) and turn >= int(getattr(args, "abstain_min_turn", 3)):
            unresolved = _unresolved_requirement_labels(
                _get_q_triples(),
                alignment_rows,
                threshold=float(args.corrector_v2_threshold),
            )
            if unresolved:
                missing = "; ".join(unresolved[:2])
                action.update({
                    "stop": True,
                    "abstain": True,
                    "reason": "unresolved_verigraph_requirements",
                    "final_response_append": (
                        f"\n\n<think>Retrieved evidence still does not support: {missing}. "
                        "I should not answer using unsupported names.</think>\n\n"
                        f"<answer>I cannot answer yet because retrieved evidence has not found: {missing}.</answer>\n"
                    ),
                })
        return action

    if args.searchr1_verbose:
        result = searchr1.infer_with_observer(
            question,
            on_turn=on_turn,
            verbose=True,
            max_turns_override=int(args.corrector_v2_max_turns),
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = searchr1.infer_with_observer(
                question,
                on_turn=on_turn,
                verbose=False,
                max_turns_override=int(args.corrector_v2_max_turns),
            )

    return CorrectorOutputs(
        full_response=str(result.get("full_response", "")),
        predicted_answer=str(result.get("predicted_answer", "") or ""),
        num_turns=int(result.get("num_turns", 0) or 0),
        turn_records=turn_records,
        retrieval_turns=list(result.get("retrieval_turns", []) or []),
        observer_events=list(result.get("observer_events", []) or []),
    )
