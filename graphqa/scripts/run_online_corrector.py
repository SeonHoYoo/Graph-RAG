"""Vanilla SearchR1 + selective VeriGraph reasoning corrector.

Per-sample pipeline
-------------------
1. Run vanilla SearchR1 (no Veri-Graph) with the project's default search
   budget.  Capture trajectory length ``n_searches`` and the predicted answer.
2. If ``n_searches <= --vanilla-trigger-threshold`` (default 3) AND the
   vanilla pass produced an ``<answer>``, accept it.  This is the regime
   where SearchR1 is empirically reliable.
3. Otherwise re-run SearchR1 from scratch with an enlarged thinking budget
   (``--corrector-max-turns``) and a callback that, after every reasoning/
   search turn, runs Veri-Graph on the latest think+docs and injects a
   per-triple alignment report (head/relation/tail cosines) back into the
   prompt.  The injected block is reasoning guidance only — it never
   contains a guessed answer.
4. If even the extended run produces no ``<answer>``, the sample is
   *abstained*; in that case EM/F1 are scored against the empty string and
   selected (non-abstain) accuracy is reported separately.

Outputs (per dataset directory):
  - ``online_corrector_<dataset>.csv``           one row per sample
  - ``online_corrector_<dataset>_summary.json``  aggregate metrics
  - ``online_corrector_<dataset>_cases.jsonl``   full trajectories + VG feedback
  - combined ``online_corrector_all.csv`` / ``online_corrector_all_summary.json``
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import pathlib
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the project root importable so we can reuse the existing extractors.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from tqdm import tqdm

from graphqa.data import Triple
from graphqa.data.schema import is_unknown
from graphqa.llm_qa import _cosine_field, _topk_whole_triple_matches
from graphqa.qa import _is_yesno_question, score_answer
from graphqa.tasi.embedding import get_default_encoder
from graphqa.scripts.run_online_eval import (
    ExtractorPool,
    GraphGenResult,
    _normalize_text,
    _split_question_triples,
    _truncate_words,
    make_graph_input,
    triples_to_raw,
)
from search_r1 import SearchR1Inference


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_latest_think(output_text: str) -> str:
    """Pull the most recent <think>...</think> chunk from a generated span."""
    matches = list(re.finditer(r"<think>(.*?)</think>", output_text or "", flags=re.DOTALL))
    if not matches:
        return ""
    return _normalize_text(matches[-1].group(1))


def _has_known_field(t: Triple) -> bool:
    """A query triple is comparable iff at least one of head/tail is concrete."""
    return (not t.head_unknown and bool((t.head or "").strip())) or (
        not t.tail_unknown and bool((t.tail or "").strip())
    )


def _triple_to_text(t: Triple) -> str:
    parts = [t.head or "", t.relation or "", t.tail or ""]
    if t.context:
        return f"{parts[0]} [SEP] {parts[1]} [SEP] {parts[2]} [PREP] {t.context}"
    return f"{parts[0]} [SEP] {parts[1]} [SEP] {parts[2]}"


# ---------------------------------------------------------------------------
# System-D helpers: hint-style verigraph injection (lessons from the
# B_fail/C_succeed case study). Goals per failure mode:
#   - Mode A (text leak into <search>): short block, distinctive sentinel
#   - Mode B (THINK self-paraphrase PASS): drop PASS/FAIL labels entirely;
#       mark think as N/A(paraphrase) when the think triple just repeats the
#       question triple
#   - Mode C (distractor entity surfaced at FAIL): hide candidate entity
#       names when doc cosine < threshold
#   - Mode D (passive-voice rejected at 0.59 vs 0.60): threshold dropped to
#       0.50 AND a small lexicon treats "directed by" ≡ "is the director of"
#   - Mode E (slot-fill collapse → wrong answer type): prepend an
#       "expects: a film title / yes or no" hint derived from a simple
#       rule-based question-type classifier
# ---------------------------------------------------------------------------


_PASSIVE_RELATION_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("is the director of", "was directed by", "directed", "director of",
     "is directed by", "has directed", "was the director of"),
    ("is the composer of", "was composed by", "composed", "composer of"),
    ("is the writer of", "was written by", "wrote", "writer of"),
    ("is the producer of", "was produced by", "produced", "producer of"),
    ("is the performer of", "was performed by", "performed", "performer of",
     "sang", "was sung by", "sings"),
    ("was born in", "is born in", "born in", "birthplace is"),
    ("died in", "passed away in", "death occurred in"),
)


def _normalize_relation(rel: str) -> str:
    return " ".join((rel or "").lower().strip().split())


def _relation_equivalent(a: str, b: str) -> bool:
    na, nb = _normalize_relation(a), _normalize_relation(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    for group in _PASSIVE_RELATION_GROUPS:
        if na in group and nb in group:
            return True
    return False


def _is_think_self_paraphrase_fields(
    q_head: str, q_rel: str, q_tail: str,
    t_head: str, t_rel: str, t_tail: str,
) -> bool:
    """Detect when a 'best think candidate' is just a paraphrase of the
    question triple — this is what gave cosine ≈1.0 false PASSes (Mode B).

    Heuristic: relation is equivalent under our active/passive lexicon AND
    tail tokens overlap ≥0.6 AND the head field is still an unbound
    (ENT\\d+) slot on either side (a true grounding would have a concrete
    head).
    """
    if not _relation_equivalent(q_rel, t_rel):
        return False
    def _tok(s: str) -> set:
        return set(w for w in (s or "").lower().split() if len(w) > 1)
    a, b = _tok(q_tail), _tok(t_tail)
    tail_overlap = (len(a & b) / max(len(a | b), 1)) if (a and b) else 0.0
    head_q_slot = is_unknown(q_head or "")
    head_t_slot = is_unknown(t_head or "") or not (t_head or "").strip()
    # AND not OR: a paraphrase only "self-confirms" when BOTH heads are
    # placeholders (the model just restated the question). A concrete head
    # on the think side ("Roman Polanski directed …") is a substantive
    # claim that should keep its cosine value, not be masked.
    return bool(tail_overlap >= 0.6 and head_q_slot and head_t_slot)


_ANSWER_TYPE_RULES: List[Tuple[str, str]] = [
    (r"\bwhich film\b|\bwhich movie\b", "a film title"),
    (r"\bwhich song\b|\bwhich album\b", "a song or album title"),
    (r"\bwhich book\b|\bwhich novel\b", "a book title"),
    (r"\bwhich (country|nation|state|province|region|city|town|village)\b", "a place name"),
    (r"\bwhich (person|director|writer|actor|actress|composer|performer|musician)\b", "a person name"),
    (r"\bwhich\b", "a short noun phrase matching the question's category"),
    (r"^\s*(are|were|do|does|did|is|was|has|have|had|can|could|will|would)\b.*\bsame (country|nationality|place)\b", "yes or no"),
    (r"^\s*(are|were|do|does|did|is|was|has|have|had|can|could|will|would|are\s+the|do\s+both|are\s+both|did\s+both|do\s+the|are\s+director|are\s+directors|did\s+the)\b", "yes or no"),
    (r"\bwho is (younger|older|taller|shorter|elder)\b", "a person name"),
    (r"\bwho is\b|\bwho was\b|\bwho are\b|\bwhose\b", "a person name"),
    (r"\bwhat (year|date)\b", "a date or year"),
    (r"\bwhen (did|was|is|will|does|do)\b", "a date or year"),
    (r"\bwhere (was|is|did|do|does)\b", "a place name"),
    (r"\bwhat (place|location|city|country|state|town|nation|region)\b", "a place name"),
    (r"\bwhat nationality\b", "a nationality"),
    (r"\bwhat (occupation|profession|job)\b", "a profession"),
    (r"\bwhat (name|title)\b", "a short name or title"),
    (r"\bhow (old|tall|long|much|many)\b", "a number or measurement"),
]


def _expected_answer_type(question: str) -> str:
    """Lightweight rule-based question-type classifier for the answer-shape
    hint in System-D blocks. Order matters; first match wins."""
    q = " ".join((question or "").lower().split())
    if not q:
        return "a short phrase"
    for pat, label in _ANSWER_TYPE_RULES:
        if re.search(pat, q):
            return label
    return "a short phrase"


def _strip_vg_hint_from_query(query: str) -> str:
    """Remove any leaked vg_hint content from a search query (Mode A fix).

    SearchR1 sometimes copies parts of the verigraph block into the next
    ``<search>`` tag. We aggressively strip anything that looks like it
    came from the block before BM25 ever sees it.
    """
    if not query:
        return query
    # Full block <vg_hint>...</vg_hint>
    s = re.sub(r"<vg_hint[^>]*>.*?</vg_hint>", " ", query, flags=re.DOTALL | re.IGNORECASE)
    # Stray standalone tags (when only one side leaked)
    s = re.sub(r"</?vg_hint[^>]*>", " ", s, flags=re.IGNORECASE)
    # Bracketed meta lines
    s = re.sub(r"\[How to read[^\]]*\]", " ", s)
    s = re.sub(r"\[Hint\][^\n]*", " ", s)
    s = re.sub(r"\[Note\][^\n]*", " ", s)
    # Per-requirement lines (start with R<digit> and contain doc=)
    s = re.sub(r"\bR\d+\s+[^\n]*?doc=[\d.]+[^\n]*", " ", s)
    # Trailer sentences that frequently get echoed
    s = re.sub(r"do\s+NOT\s+copy[^.]*\.?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"do\s+NOT\s+use[^.]*\.?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"Use this only[^.]*\.?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"aim for both ≥[^.]*\.?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s




# ---------------------------------------------------------------------------
# Verigraph feedback formatter
# ---------------------------------------------------------------------------


def _fmt_cosine(v: Optional[float]) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{float(v):.2f}"


def _per_field_max_against_candidate(
    q: Triple,
    cand: Triple,
    encoder,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Standard Veri-Graph per-field alignment of one (Q, candidate) pair.

    For each KNOWN field of the question:
      - ``head`` is scored against BOTH ``cand.head`` and ``cand.tail``; the
        higher cosine is the selected match (cross is the passive-voice
        signal).
      - ``tail`` is scored against BOTH ``cand.tail`` and ``cand.head``; the
        higher cosine wins.
      - ``relation`` is scored only against ``cand.relation`` (no swap).

    Returns ``(fields, min_field_cosine)`` where ``fields`` is a list of
    per-field rows describing the options compared, the selected option
    (with a ``cross`` flag), and the cosines.  ``min_field_cosine`` is the
    aggregate score across used fields, NaN if no field could be used.
    """
    fields: List[Dict[str, Any]] = []
    used: List[float] = []

    # HEAD : q.head vs max(cand.head, cand.tail)
    if not q.head_unknown and (q.head or "").strip():
        ch_cos = _cosine_field(q.head, cand.head, encoder)
        ct_cos = _cosine_field(q.head, cand.tail, encoder)
        options = [
            {"side": "cand.head", "candidate_value": cand.head or "",
             "cosine": (float(ch_cos) if ch_cos == ch_cos else None)},
            {"side": "cand.tail", "candidate_value": cand.tail or "",
             "cosine": (float(ct_cos) if ct_cos == ct_cos else None)},
        ]
        # pick the higher (treat NaN as -inf)
        def _key(opt: Dict[str, Any]) -> float:
            v = opt["cosine"]
            return float(v) if v is not None else float("-inf")
        sel = max(options, key=_key)
        sel_score = sel["cosine"]
        is_cross = bool(sel["side"] == "cand.tail")
        fields.append({
            "field": "head",
            "query_value": q.head,
            "used": True,
            "options": options,
            "selected": {**sel, "is_cross": is_cross},
        })
        if sel_score is not None:
            used.append(float(sel_score))
    else:
        fields.append({
            "field": "head", "query_value": q.head, "used": False,
            "skip_reason": "query_field_unknown",
            "options": [], "selected": None,
        })

    # RELATION : locked to relation↔relation
    if (q.relation or "").strip() and (cand.relation or "").strip():
        r_cos = _cosine_field(q.relation, cand.relation, encoder)
        rel_val = float(r_cos) if r_cos == r_cos else None
        fields.append({
            "field": "relation",
            "query_value": q.relation,
            "used": True,
            "options": [
                {"side": "cand.relation", "candidate_value": cand.relation, "cosine": rel_val},
            ],
            "selected": {
                "side": "cand.relation", "candidate_value": cand.relation,
                "cosine": rel_val, "is_cross": False,
            },
        })
        if rel_val is not None:
            used.append(rel_val)
    else:
        fields.append({
            "field": "relation", "query_value": q.relation, "used": False,
            "skip_reason": "missing_relation",
            "options": [], "selected": None,
        })

    # TAIL : q.tail vs max(cand.tail, cand.head)
    if not q.tail_unknown and (q.tail or "").strip():
        tt_cos = _cosine_field(q.tail, cand.tail, encoder)
        th_cos = _cosine_field(q.tail, cand.head, encoder)
        options = [
            {"side": "cand.tail", "candidate_value": cand.tail or "",
             "cosine": (float(tt_cos) if tt_cos == tt_cos else None)},
            {"side": "cand.head", "candidate_value": cand.head or "",
             "cosine": (float(th_cos) if th_cos == th_cos else None)},
        ]
        def _key2(opt: Dict[str, Any]) -> float:
            v = opt["cosine"]
            return float(v) if v is not None else float("-inf")
        sel = max(options, key=_key2)
        sel_score = sel["cosine"]
        is_cross = bool(sel["side"] == "cand.head")
        fields.append({
            "field": "tail",
            "query_value": q.tail,
            "used": True,
            "options": options,
            "selected": {**sel, "is_cross": is_cross},
        })
        if sel_score is not None:
            used.append(float(sel_score))
    else:
        fields.append({
            "field": "tail", "query_value": q.tail, "used": False,
            "skip_reason": "query_field_unknown",
            "options": [], "selected": None,
        })

    min_field = (min(used) if used else float("nan"))
    return fields, (float(min_field) if min_field == min_field else None)


def _best_per_field_match(
    q: Triple,
    pool: Sequence[Triple],
    encoder,
    *,
    top_k: int,
    threshold: float,
) -> Dict[str, Any]:
    """Pick the top-K pool candidates by whole-triple cosine, then for each
    candidate compute the per-field-max alignment and return the candidate
    whose aggregate (min over used fields) is highest.
    """
    topk = _topk_whole_triple_matches(q, list(pool), encoder, top_k=max(1, int(top_k)))
    if not topk:
        return {
            "candidate_index": -1,
            "candidate_text": "",
            "candidate_head": "", "candidate_relation": "", "candidate_tail": "",
            "whole_cosine": None,
            "min_field_cosine": None,
            "passes_threshold": False,
            "fields": [],
            "rank_by_whole": -1,
            "all_topk_candidates": [],
        }

    all_topk_records: List[Dict[str, Any]] = []
    best: Optional[Tuple[Tuple[float, float, int], Dict[str, Any]]] = None
    for rank, (idx, whole) in enumerate(topk, start=1):
        cand = pool[idx]
        fields, min_field = _per_field_max_against_candidate(q, cand, encoder)
        record = {
            "rank_by_whole": int(rank),
            "candidate_index": int(idx),
            "candidate_text": _triple_to_text(cand),
            "candidate_head": cand.head or "",
            "candidate_relation": cand.relation or "",
            "candidate_tail": cand.tail or "",
            "whole_cosine": (float(whole) if whole == whole else None),
            "min_field_cosine": (float(min_field) if min_field is not None else None),
            "passes_threshold": bool(min_field is not None and min_field >= threshold),
            "fields": fields,
        }
        all_topk_records.append(record)
        # Key: per-field-min first, then whole cosine, then earlier rank.
        score_key = (
            (float(min_field) if min_field is not None else float("-inf")),
            (float(whole) if whole == whole else float("-inf")),
            -rank,
        )
        if best is None or score_key > best[0]:
            best = (score_key, record)

    chosen = dict(best[1]) if best else all_topk_records[0]
    chosen["all_topk_candidates"] = all_topk_records
    return chosen


def _per_q_alignment(
    q_triples: Sequence[Triple],
    doc_pool: Sequence[Triple],
    think_pool: Sequence[Triple],
    encoder,
    *,
    top_k: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    """For each comparable Q triple, return the best DOC and THINK matches.

    Uses the standard Veri-Graph per-field-max alignment: head and tail can
    each independently pick the higher cosine against cand.head or cand.tail;
    relation is locked to relation.  Searches among the top-K candidates by
    whole-triple cosine.
    """
    rows: List[Dict[str, Any]] = []
    for q_idx, q in enumerate(q_triples):
        if not _has_known_field(q):
            continue

        doc_match = _best_per_field_match(
            q, doc_pool, encoder, top_k=top_k, threshold=threshold,
        )
        think_match = _best_per_field_match(
            q, think_pool, encoder, top_k=top_k, threshold=threshold,
        )

        rows.append({
            "q_idx": int(q_idx),
            "q_text": _triple_to_text(q),
            "q_head": q.head, "q_relation": q.relation, "q_tail": q.tail,
            "doc_match": doc_match,
            "think_match": think_match,
        })
    return rows


def _display_value(val: str) -> str:
    """Render a triple field for the LLM, calling out empty slots and placeholders.

    The graph extractor emits placeholders like ``(ENT1)`` for entities the
    question has not yet bound to a concrete value.  SearchR1 cannot tell
    those apart from literal strings without help, so we annotate them
    explicitly.  Empty strings are shown as ``(empty)``.
    """
    s = (val or "").strip()
    if not s:
        return "(empty)"
    if is_unknown(s):
        return f"[placeholder {s} — undetermined slot, fill from evidence]"
    return f"\"{s}\""


def _render_per_field_match(match: Dict[str, Any]) -> List[str]:
    """Render a per-field-max candidate match for one side (DOC or THINK).

    For HEAD: shows q.head vs whichever of (cand.head, cand.tail) scored
    higher; flags it as CROSS if it landed on cand.tail.  Same shape for
    TAIL with cand.tail/cand.head swapped.  RELATION is always
    relation↔relation.  The losing side is shown in parentheses for
    transparency so SearchR1 can see why the winner won.
    """
    out: List[str] = []
    fields = match.get("fields", []) or []
    by_field = {f.get("field", ""): f for f in fields}

    for fname in ("head", "relation", "tail"):
        f = by_field.get(fname)
        if f is None:
            continue
        if not f.get("used"):
            reason = f.get("skip_reason") or "skipped"
            out.append(
                "          {fn:>8}: {q}  ({r})".format(
                    fn=fname, q=_display_value(f.get("query_value") or ""), r=reason,
                )
            )
            continue
        sel = f.get("selected") or {}
        sel_side = sel.get("side", "")
        sel_val = sel.get("candidate_value", "")
        sel_cos = sel.get("cosine")
        is_cross = bool(sel.get("is_cross", False))
        # Compact side label for the winner
        if fname == "head":
            winner_label = "↔cand.head" if sel_side == "cand.head" else "↔cand.tail [CROSS]"
        elif fname == "tail":
            winner_label = "↔cand.tail" if sel_side == "cand.tail" else "↔cand.head [CROSS]"
        else:
            winner_label = "↔cand.relation"
        out.append(
            "          {fn:>8} {wl:<20}: {q} vs {c}  cos={s}{xnote}".format(
                fn=fname, wl=winner_label,
                q=_display_value(f.get("query_value") or ""),
                c=_display_value(sel_val),
                s=_fmt_cosine(sel_cos),
                xnote=(" — passive-voice / swapped subject" if is_cross else ""),
            )
        )
        # For head/tail, also surface the alternative option in parens so the
        # LLM sees that we actually compared both sides.
        if fname in ("head", "tail"):
            for opt in f.get("options", []) or []:
                if opt.get("side") == sel_side:
                    continue
                out.append(
                    "                   alt {os:<13}: {q} vs {c}  cos={s}".format(
                        os=opt.get("side", ""),
                        q=_display_value(f.get("query_value") or ""),
                        c=_display_value(opt.get("candidate_value", "")),
                        s=_fmt_cosine(opt.get("cosine")),
                    )
                )
    return out


def _format_feedback_block(
    turn: int,
    alignment_rows: Sequence[Dict[str, Any]],
    *,
    threshold: float,
    top_k: int,
) -> str:
    """Render the Veri-Graph alignment report for the LLM.

    Standard Veri-Graph philosophy:
      - For each question triple, score against the top-K candidate triples
        (by whole-triple cosine) and pick the candidate that maximises the
        per-field-min after each field has independently maxed over the
        relevant sides:
            HEAD     : max(cos(q.head, cand.head),  cos(q.head, cand.tail))
            RELATION : cos(q.relation, cand.relation)             (locked)
            TAIL     : max(cos(q.tail, cand.tail),  cos(q.tail, cand.head))
      - When HEAD picks cand.tail (or TAIL picks cand.head) we flag the
        match as CROSS — that orientation flip is the passive-voice signal.

    Placeholders like ``(ENT1)`` are surfaced as undetermined slots so
    SearchR1 does not mistake them for literal strings.  The report is
    explicitly reasoning-guidance, not an answer hint.
    """
    if not alignment_rows:
        return (
            "\n<verigraph_check turn=\"{turn}\">\n"
            "No comparable question triples to verify yet (your latest reasoning has\n"
            "no concrete entities to align). Refine your search query and try again.\n"
            "</verigraph_check>\n\n"
        ).format(turn=turn)

    lines: List[str] = []
    lines.append(f"<verigraph_check turn=\"{turn}\">")
    lines.append("[How this report was produced]")
    lines.append(
        "  1. I parsed your most recent <think> and the documents you have retrieved "
        "so far into (subject, relation, object) triples."
    )
    lines.append(
        "  2. For each requirement of the question (one triple per requirement) I picked "
        f"the top-{top_k} candidate triples by whole-triple cosine, then for each candidate "
        "scored fields independently:"
    )
    lines.append(
        "       head     : higher of cos(q.head, cand.head) and cos(q.head, cand.tail)"
    )
    lines.append(
        "       relation : cos(q.relation, cand.relation)             (no swap)"
    )
    lines.append(
        "       tail     : higher of cos(q.tail, cand.tail) and cos(q.tail, cand.head)"
    )
    lines.append(
        "     The candidate whose min-of-used-fields is highest is reported below."
    )
    lines.append(
        f"  3. Gate threshold = {threshold:.2f}. A field landing on the *swapped* "
        "side (head→cand.tail or tail→cand.head) is marked [CROSS] — that signals "
        "a passive-voice / swapped-subject paraphrase."
    )
    lines.append(
        "  4. Placeholders like (ENT1), (ENT2) are SLOTS the question leaves open; "
        "treat them as empty unknowns to be filled from evidence, not as literal strings."
    )
    lines.append(
        "Use this report only to refine your next <think>/<search>. "
        "Do NOT use it to guess an answer; it does not contain one."
    )

    for row in alignment_rows:
        q_text = row["q_text"]
        d = row["doc_match"]
        t = row["think_match"]

        lines.append("")
        lines.append(f"Requirement: {q_text}")

        # DOC side
        if d and d.get("candidate_text"):
            lines.append(
                "  [DOC]   best evidence triple (rank={r} of top-{k} by whole-cos): "
                "{ct}  (whole-cos={w}, min-field-cos={mf}, gate={pf})".format(
                    r=d.get("rank_by_whole", "?"), k=top_k,
                    ct=d["candidate_text"],
                    w=_fmt_cosine(d.get("whole_cosine")),
                    mf=_fmt_cosine(d.get("min_field_cosine")),
                    pf=("PASS" if d.get("passes_threshold") else "FAIL"),
                )
            )
            lines.extend(_render_per_field_match(d))
        else:
            lines.append("  [DOC]   no candidate document triple found yet.")

        # THINK side
        if t and t.get("candidate_text"):
            lines.append(
                "  [THINK] best reasoning triple (rank={r} of top-{k} by whole-cos): "
                "{ct}  (whole-cos={w}, min-field-cos={mf}, gate={pf})".format(
                    r=t.get("rank_by_whole", "?"), k=top_k,
                    ct=t["candidate_text"],
                    w=_fmt_cosine(t.get("whole_cosine")),
                    mf=_fmt_cosine(t.get("min_field_cosine")),
                    pf=("PASS" if t.get("passes_threshold") else "FAIL"),
                )
            )
            lines.extend(_render_per_field_match(t))
        else:
            lines.append(
                "  [THINK] your reasoning has not yet stated a triple comparable "
                "to this requirement."
            )

        # Per-requirement diagnosis
        d_pass = bool(d and d.get("passes_threshold"))
        t_pass = bool(t and t.get("passes_threshold"))
        if d_pass and t_pass:
            diag = "OK on both sides."
        elif d_pass and not t_pass:
            diag = "Evidence is present but your reasoning has not used it correctly; restate the fact more precisely in your next <think>."
        elif (not d_pass) and t_pass:
            diag = "Your reasoning asserts this but no retrieved document supports it; issue a <search> for direct evidence."
        else:
            diag = "Neither documents nor reasoning support this requirement; issue a more targeted <search>."
        lines.append(f"  → {diag}")

    lines.append("</verigraph_check>")
    return "\n\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# System-D feedback block (hint-style, threshold-relaxed, format-sanitized)
# ---------------------------------------------------------------------------


def _short_triple_summary(q: Triple, max_words: int = 8) -> str:
    """Render a question requirement as a compact one-liner for the hint
    block. Keeps placeholders as 'ENT1' (not the verbose '[placeholder ...]')
    and clips long tails so the block stays short."""
    def _clip(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return "?"
        if is_unknown(s):
            return re.sub(r"[()]", "", s).strip() or "ENT?"
        parts = s.split()
        if len(parts) > max_words:
            return " ".join(parts[:max_words]) + "…"
        return s
    head = _clip(q.head)
    rel = _clip(q.relation)
    tail = _clip(q.tail)
    return f'{head} {rel} "{tail}"'


def _format_feedback_block_v2(
    turn: int,
    alignment_rows: Sequence[Dict[str, Any]],
    *,
    threshold: float,
    expected_answer_type: str,
    q_triples_in_order: Sequence[Triple],
    last_block_summary: Optional[str] = None,
) -> Tuple[str, str]:
    """Hint-style verigraph block (System D).

    Design choices vs. System B:
      - No preamble (saved every turn).
      - No PASS/FAIL labels; raw cosine values only.
      - One line per requirement (R1, R2, …) instead of multi-line per-field.
      - Candidate entity name hidden when doc cosine < threshold (Mode C).
      - Think marked N/A(paraphrase) when it just restates the question
        triple (Mode B).
      - Distinctive ``<vg_hint>`` sentinel (Mode A).
      - Expected answer type prepended (Mode E).

    Returns (block_text, summary_for_stagnation_check).
    """
    if not alignment_rows:
        block = (
            "\n<vg_hint turn=\"{t}\" expects=\"{at}\">\n"
            "[Hint] No comparable requirements scored yet — refine your search query.\n"
            "</vg_hint>\n\n"
        ).format(t=turn, at=expected_answer_type)
        return block, "empty"

    lines: List[str] = []
    lines.append(f'<vg_hint turn="{turn}" expects="{expected_answer_type}" threshold={threshold:.2f}>')
    if turn == 0:
        lines.append(
            "[How to read] doc/think are 0-1 similarity scores; "
            "aim for both ≥ {th:.2f} to trust a fact. Use this only as a hint; do NOT copy this block into <search>.".format(
                th=threshold,
            )
        )

    rid_to_row = {row["q_idx"]: row for row in alignment_rows}
    summary_parts: List[str] = []

    for q_idx, q in enumerate(q_triples_in_order):
        if not _has_known_field(q):
            continue
        row = rid_to_row.get(q_idx)
        if row is None:
            continue
        d = row.get("doc_match") or {}
        t = row.get("think_match") or {}

        d_cos = d.get("min_field_cosine")
        t_cos = t.get("min_field_cosine")
        d_val = float(d_cos) if d_cos is not None else None
        t_val = float(t_cos) if t_cos is not None else None

        # Mode B: detect think paraphrase of the question triple.
        think_label = _fmt_cosine(t_cos)
        if t and t.get("candidate_text"):
            if _is_think_self_paraphrase_fields(
                q.head or "", q.relation or "", q.tail or "",
                t.get("candidate_head") or "",
                t.get("candidate_relation") or "",
                t.get("candidate_tail") or "",
            ):
                think_label = "N/A(paraphrase)"
        else:
            think_label = "N/A(no-triple)"

        # Mode D extra: if relation is in the active/passive lexicon AND
        # doc cosine is in [threshold-0.10, threshold), nudge the doc value up
        # by treating the relation as equivalent — surface it as
        # "doc=0.XX (paraphrase OK)".
        doc_extra = ""
        if d and d.get("candidate_text") and d_val is not None:
            q_rel = q.relation or ""
            c_rel = d.get("candidate_relation", "") or ""
            if (d_val < threshold and _relation_equivalent(q_rel, c_rel)
                    and d_val >= max(threshold - 0.15, 0.30)):
                doc_extra = " (paraphrase-OK)"

        # Mode C: hide candidate entity name unless doc is clearly strong.
        # We use a tighter "show name" cutoff (max(threshold+0.20, 0.70))
        # because Mode C examples like "Renny Harlin (cos=0.59)" misled the
        # model — surface only when we're confident enough that this is
        # likely the right entity, not a distractor.
        SHOW_NAME_THRESHOLD = max(threshold + 0.20, 0.70)
        cand_disp = ""
        if d and d.get("candidate_text") and d_val is not None:
            if d_val >= SHOW_NAME_THRESHOLD:
                cand_h = (d.get("candidate_head") or "").strip()
                cand_t = (d.get("candidate_tail") or "").strip()
                cand_disp_bits = []
                if cand_h and not is_unknown(cand_h):
                    cand_disp_bits.append(cand_h[:40])
                if cand_t and not is_unknown(cand_t):
                    cand_disp_bits.append(cand_t[:40])
                if cand_disp_bits:
                    cand_disp = f' [candidate: {" — ".join(cand_disp_bits)}]'

        rid = f"R{q_idx + 1}"
        req_text = _short_triple_summary(q)
        doc_part = _fmt_cosine(d_val) + doc_extra if d_val is not None else "N/A"
        lines.append(f'{rid} {req_text}: doc={doc_part}  think={think_label}{cand_disp}')
        summary_parts.append(f'{rid}:doc={doc_part};think={think_label}')

    block_summary = "|".join(summary_parts)

    # Stagnation hint — if exactly the same summary as last turn, escalate.
    if last_block_summary and last_block_summary == block_summary:
        lines.append(
            "[Note] These scores have not changed since the last turn. "
            "Try a different search-query strategy (year, country qualifier, alternate title) "
            "or answer from the strongest existing evidence."
        )

    lines.append("</vg_hint>")
    return "\n\n" + "\n".join(lines) + "\n\n", block_summary


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


@dataclass
class CorrectorOutputs:
    full_response: str
    predicted_answer: str
    num_turns: int
    turn_records: List[Dict[str, Any]]
    retrieval_turns: List[Dict[str, Any]]
    observer_events: List[Dict[str, Any]]


def _run_corrector(
    *,
    question: str,
    q_triples: List[Triple],
    pool: ExtractorPool,
    searchr1: SearchR1Inference,
    encoder,
    args: argparse.Namespace,
) -> CorrectorOutputs:
    d_extractor = pool.get(args.document_model)
    t_extractor = pool.get(args.think_model)

    doc_cache: Dict[str, GraphGenResult] = {}
    accumulated_doc_triples: List[Triple] = []
    seen_doc_keys: set = set()
    turn_records: List[Dict[str, Any]] = []

    def _add_doc(doc: str) -> List[Triple]:
        if doc in doc_cache:
            return doc_cache[doc].triples
        truncated = _truncate_words(doc, int(args.doc_max_words))
        result = d_extractor.generate(
            "document",
            make_graph_input("document", document=truncated),
        )
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
        return result.triples

    def on_turn(event: Dict[str, Any]) -> Dict[str, Any]:
        turn = int(event.get("turn", len(turn_records)))
        query = _normalize_text(str(event.get("query", "") or ""))
        docs = list(event.get("search_results", []) or [])[: int(args.max_docs_per_sample)]
        think_text = _extract_latest_think(str(event.get("output_text", "") or ""))

        # Extract doc graph for any docs we haven't seen yet
        new_doc_triples_count_before = len(accumulated_doc_triples)
        for d in docs:
            _add_doc(d)
        n_new_doc_triples = len(accumulated_doc_triples) - new_doc_triples_count_before

        # Extract think graph for the latest think
        think_result = GraphGenResult(text="", triples=[])
        if think_text:
            think_result = t_extractor.generate(
                args.think_task,
                make_graph_input(args.think_task, think=think_text, search_query=query),
            )

        alignment_rows = _per_q_alignment(
            q_triples,
            accumulated_doc_triples,
            think_result.triples,
            encoder,
            top_k=int(args.cosine_doc_top_k),
            threshold=float(args.cosine_threshold),
        )

        feedback_block = _format_feedback_block(
            turn=turn,
            alignment_rows=alignment_rows,
            threshold=float(args.cosine_threshold),
            top_k=int(args.cosine_doc_top_k),
        )

        turn_records.append({
            "turn": turn,
            "query": query,
            "think_text": think_text,
            "n_docs_this_turn": len(docs),
            "n_new_doc_triples": int(n_new_doc_triples),
            "n_total_doc_triples": len(accumulated_doc_triples),
            "n_think_triples": len(think_result.triples),
            "think_raw_graph": think_result.text,
            "think_triples": triples_to_raw(think_result.triples),
            "alignment": alignment_rows,
            "injected_feedback": feedback_block,
        })

        return {"prompt_injection": feedback_block}

    if args.searchr1_verbose:
        result = searchr1.infer_with_observer(
            question,
            on_turn=on_turn,
            verbose=True,
            max_turns_override=int(args.corrector_max_turns),
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = searchr1.infer_with_observer(
                question,
                on_turn=on_turn,
                verbose=False,
                max_turns_override=int(args.corrector_max_turns),
            )

    return CorrectorOutputs(
        full_response=str(result.get("full_response", "")),
        predicted_answer=str(result.get("predicted_answer", "") or ""),
        num_turns=int(result.get("num_turns", 0) or 0),
        turn_records=turn_records,
        retrieval_turns=list(result.get("retrieval_turns", []) or []),
        observer_events=list(result.get("observer_events", []) or []),
    )


def _run_corrector_v2(
    *,
    question: str,
    q_triples: List[Triple],
    pool: ExtractorPool,
    searchr1: SearchR1Inference,
    encoder,
    args: argparse.Namespace,
) -> CorrectorOutputs:
    """System-D: hint-style verigraph corrector.

    Same per-turn loop as System-B, but:
      - Uses ``_format_feedback_block_v2`` (short, no PASS/FAIL, value-only).
      - Threshold ``args.corrector_v2_threshold`` (default 0.50).
      - Block is wrapped in ``<vg_hint>``; we also sanitize the search query
        in case the model echoes the block (Mode A protection).
    """
    d_extractor = pool.get(args.document_model)
    t_extractor = pool.get(args.think_model)

    doc_cache: Dict[str, GraphGenResult] = {}
    accumulated_doc_triples: List[Triple] = []
    seen_doc_keys: set = set()
    turn_records: List[Dict[str, Any]] = []
    last_block_summary: Optional[str] = None

    expected_answer_type = _expected_answer_type(question)

    def _add_doc(doc: str) -> List[Triple]:
        if doc in doc_cache:
            return doc_cache[doc].triples
        truncated = _truncate_words(doc, int(args.doc_max_words))
        result = d_extractor.generate(
            "document",
            make_graph_input("document", document=truncated),
        )
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
        return result.triples

    def on_turn(event: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal last_block_summary
        turn = int(event.get("turn", len(turn_records)))
        raw_query = _normalize_text(str(event.get("query", "") or ""))
        # Mode A protection: strip any vg_hint content that leaked into <search>
        clean_query = _strip_vg_hint_from_query(raw_query)
        docs = list(event.get("search_results", []) or [])[: int(args.max_docs_per_sample)]
        think_text = _extract_latest_think(str(event.get("output_text", "") or ""))

        new_doc_triples_count_before = len(accumulated_doc_triples)
        for d in docs:
            _add_doc(d)
        n_new_doc_triples = len(accumulated_doc_triples) - new_doc_triples_count_before

        think_result = GraphGenResult(text="", triples=[])
        if think_text:
            think_result = t_extractor.generate(
                args.think_task,
                make_graph_input(args.think_task, think=think_text, search_query=clean_query),
            )

        alignment_rows = _per_q_alignment(
            q_triples,
            accumulated_doc_triples,
            think_result.triples,
            encoder,
            top_k=int(args.cosine_doc_top_k),
            threshold=float(args.corrector_v2_threshold),
        )

        feedback_block, block_summary = _format_feedback_block_v2(
            turn=turn,
            alignment_rows=alignment_rows,
            threshold=float(args.corrector_v2_threshold),
            expected_answer_type=expected_answer_type,
            q_triples_in_order=q_triples,
            last_block_summary=last_block_summary,
        )
        last_block_summary = block_summary

        turn_records.append({
            "turn": turn,
            "query": clean_query,
            "raw_query_pre_sanitize": raw_query,
            "query_was_sanitized": bool(clean_query != raw_query),
            "think_text": think_text,
            "n_docs_this_turn": len(docs),
            "n_new_doc_triples": int(n_new_doc_triples),
            "n_total_doc_triples": len(accumulated_doc_triples),
            "n_think_triples": len(think_result.triples),
            "think_raw_graph": think_result.text,
            "think_triples": triples_to_raw(think_result.triples),
            "alignment": alignment_rows,
            "expected_answer_type": expected_answer_type,
            "block_summary": block_summary,
            "injected_feedback": feedback_block,
        })

        return {"prompt_injection": feedback_block}

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


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------


def _process_sample(
    raw: Dict[str, Any],
    dataset: str,
    pool: ExtractorPool,
    searchr1: SearchR1Inference,
    encoder,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    question = _normalize_text(raw.get("question", ""))
    gold_answer = str(raw.get("answer", "") or "")
    gold_aliases = list(raw.get("answer_aliases", []) or [])

    # Phase 0: extract question graph once (used for all corrector turns)
    q_extractor = pool.get(args.question_model)
    q_result = q_extractor.generate(
        args.question_task,
        make_graph_input(args.question_task, question=question),
    )
    q_triples, _q_def_triples = _split_question_triples(q_result.triples)

    # Phase 1: vanilla SearchR1 (no verigraph)
    if args.searchr1_verbose:
        vanilla_raw = searchr1.infer(question, verbose=True)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            vanilla_raw = searchr1.infer(question, verbose=False)
    vanilla_answer = str(vanilla_raw.get("predicted_answer", "") or "")
    vanilla_turns = int(vanilla_raw.get("num_turns", 0) or 0)

    # Decision: trigger verigraph or trust vanilla
    is_long = vanilla_turns > int(args.vanilla_trigger_threshold)
    is_no_answer = not vanilla_answer
    trigger_verigraph = is_long or is_no_answer
    if is_long and is_no_answer:
        trigger_reason_label = "vanilla_long_and_no_answer"
    elif is_long:
        trigger_reason_label = "vanilla_long"
    elif is_no_answer:
        trigger_reason_label = "vanilla_no_answer"
    else:
        trigger_reason_label = ""

    corrector_record: Optional[Dict[str, Any]] = None
    final_answer = vanilla_answer
    final_mode = "vanilla"
    final_turns = vanilla_turns
    abstained = False

    if trigger_verigraph:
        cor = _run_corrector(
            question=question,
            q_triples=q_triples,
            pool=pool,
            searchr1=searchr1,
            encoder=encoder,
            args=args,
        )
        corrector_record = {
            "full_response": cor.full_response,
            "predicted_answer": cor.predicted_answer,
            "num_turns": cor.num_turns,
            "retrieval_turns": cor.retrieval_turns,
            "turn_records": cor.turn_records,
        }
        if cor.predicted_answer:
            final_answer = cor.predicted_answer
            final_mode = "verigraph"
            final_turns = cor.num_turns
        else:
            final_answer = ""
            final_mode = "abstain"
            final_turns = cor.num_turns
            abstained = True

    # ------------------------------------------------------------------
    # System-C ablation: vanilla SearchR1 with the EXTENDED budget,
    # no verigraph injection.  Runs AFTER the corrector so it does not
    # perturb the corrector's RNG state.  For non-triggered samples we
    # copy the vanilla 5-turn result because SearchR1 emits <answer> +
    # EOS before turn 5, so the 10-turn run would produce the same text.
    # ------------------------------------------------------------------
    control_record: Optional[Dict[str, Any]] = None
    if trigger_verigraph:
        if args.searchr1_verbose:
            control_raw = searchr1.infer_with_observer(
                question, on_turn=None, verbose=True,
                max_turns_override=int(args.control_max_turns),
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                control_raw = searchr1.infer_with_observer(
                    question, on_turn=None, verbose=False,
                    max_turns_override=int(args.control_max_turns),
                )
        control_answer = str(control_raw.get("predicted_answer", "") or "")
        control_turns = int(control_raw.get("num_turns", 0) or 0)
        control_em, control_f1 = score_answer(control_answer, gold_answer, gold_aliases)
        control_record = {
            "full_response": str(control_raw.get("full_response", "")),
            "predicted_answer": control_answer,
            "num_turns": control_turns,
            "retrieval_turns": list(control_raw.get("retrieval_turns", []) or []),
            "em": float(control_em),
            "f1": float(control_f1),
        }
    else:
        # Skip the extra inference; the result would be identical to vanilla.
        control_answer = vanilla_answer
        control_turns = vanilla_turns
        control_em, control_f1 = 0.0, 0.0  # filled below after vanilla EM is computed

    # ------------------------------------------------------------------
    # System-D: hint-style verigraph corrector (lessons from case study).
    # Runs after control (and after corrector V1) so it does not perturb
    # earlier phases' RNG states. Skip when not triggered (same logic as
    # control — vanilla short would have terminated identically).
    # ------------------------------------------------------------------
    corrector_v2_record: Optional[Dict[str, Any]] = None
    if trigger_verigraph:
        cor2 = _run_corrector_v2(
            question=question,
            q_triples=q_triples,
            pool=pool,
            searchr1=searchr1,
            encoder=encoder,
            args=args,
        )
        corrector_v2_record = {
            "full_response": cor2.full_response,
            "predicted_answer": cor2.predicted_answer,
            "num_turns": cor2.num_turns,
            "retrieval_turns": cor2.retrieval_turns,
            "turn_records": cor2.turn_records,
        }
        system_d_answer = cor2.predicted_answer if cor2.predicted_answer else ""
        system_d_turns = cor2.num_turns
    else:
        system_d_answer = vanilla_answer
        system_d_turns = vanilla_turns

    em_val, f1_val = score_answer(final_answer, gold_answer, gold_aliases)
    vanilla_em, vanilla_f1 = score_answer(vanilla_answer, gold_answer, gold_aliases)
    if not trigger_verigraph:
        # Non-triggered: control == vanilla (early EOS guarantees same answer).
        control_em, control_f1 = vanilla_em, vanilla_f1
    system_d_em, system_d_f1 = score_answer(system_d_answer, gold_answer, gold_aliases)

    csv_row: Dict[str, Any] = {
        "dataset": dataset,
        "uid": raw.get("uid") or raw.get("id"),
        "index": raw.get("index"),
        "question": question,
        "answer": gold_answer,
        "is_yesno": bool(_is_yesno_question(question)),
        "n_hops": int(raw.get("num_hops", 0) or 0),
        "vanilla_num_turns": vanilla_turns,
        "vanilla_predicted_answer": vanilla_answer,
        "vanilla_em": float(vanilla_em),
        "vanilla_f1": float(vanilla_f1),
        "triggered_verigraph": bool(trigger_verigraph),
        "trigger_reason": trigger_reason_label,
        "corrector_num_turns": (corrector_record["num_turns"] if corrector_record else 0),
        "corrector_predicted_answer": (
            corrector_record["predicted_answer"] if corrector_record else ""
        ),
        # System-C control: vanilla SearchR1 with extended budget, no verigraph
        "control_predicted_answer": control_answer,
        "control_num_turns": int(control_turns),
        "control_em": float(control_em),
        "control_f1": float(control_f1),
        "control_ran_inference": bool(trigger_verigraph),  # False = copied from vanilla
        # System-D: hint-style verigraph (post-case-study redesign)
        "system_d_predicted_answer": system_d_answer,
        "system_d_num_turns": int(system_d_turns),
        "system_d_em": float(system_d_em),
        "system_d_f1": float(system_d_f1),
        "system_d_ran_inference": bool(trigger_verigraph),
        "mode": final_mode,
        "abstained": bool(abstained),
        "final_answer": final_answer,
        "em": float(em_val),
        "f1": float(f1_val),
        "is_correct": bool(em_val >= 1.0),
        "n_q_triples": len(q_triples),
        "final_total_turns": int(final_turns),
    }

    case_record: Dict[str, Any] = {
        "dataset": dataset,
        "uid": raw.get("uid") or raw.get("id"),
        "index": raw.get("index"),
        "question": question,
        "answer": gold_answer,
        "answer_aliases": gold_aliases,
        "csv_row": csv_row,  # makes JSONL the single source of truth for resume
        "question_graph": {
            "model": args.question_model,
            "task": args.question_task,
            "raw_graph": q_result.text,
            "raw_triples": triples_to_raw(q_result.triples),
            "kept_triples": triples_to_raw(q_triples),
        },
        "vanilla": {
            "full_response": vanilla_raw.get("full_response", ""),
            "predicted_answer": vanilla_answer,
            "num_turns": vanilla_turns,
            "retrieval_turns": vanilla_raw.get("retrieval_turns", []),
            "em": float(vanilla_em),
            "f1": float(vanilla_f1),
        },
        "trigger": {
            "threshold": int(args.vanilla_trigger_threshold),
            "triggered": bool(trigger_verigraph),
            "reason": csv_row["trigger_reason"],
        },
        "corrector": corrector_record,
        "control": control_record,  # None if not triggered (control == vanilla)
        "corrector_v2": corrector_v2_record,  # None if not triggered
        "final": {
            "mode": final_mode,
            "answer": final_answer,
            "em": float(em_val),
            "f1": float(f1_val),
            "abstained": bool(abstained),
            "num_turns": int(final_turns),
        },
    }
    return csv_row, case_record


# ---------------------------------------------------------------------------
# Argparse + main loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, nargs="+", required=True)
    p.add_argument("--input-filename", type=str, default="train_sampled.json")
    p.add_argument("--limit", type=int, default=0,
                   help="Global limit per dataset (0 = no global limit, use --dataset-limits).")
    p.add_argument("--dataset-limits", type=int, nargs="*", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--retriever-url", type=str, default="http://127.0.0.1:8000/retrieve")
    # SearchR1
    p.add_argument("--searchr1-model", type=str,
                   default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    p.add_argument("--searchr1-top-k", type=int, default=3)
    p.add_argument("--searchr1-max-turns", type=int, default=4,
                   help="Vanilla SearchR1 max turns (phase 1).")
    p.add_argument("--searchr1-max-new-tokens", type=int, default=500)
    p.add_argument("--searchr1-temperature", type=float, default=1.0)
    p.add_argument("--searchr1-verbose", action="store_true")
    # Trigger / corrector
    p.add_argument("--vanilla-trigger-threshold", type=int, default=3,
                   help="If vanilla used STRICTLY more than this many search turns, run the corrector. Also fired when vanilla has no <answer>.")
    p.add_argument("--corrector-max-turns", type=int, default=8,
                   help="Extended thinking budget when the corrector is active.")
    p.add_argument("--control-max-turns", type=int, default=10,
                   help="System-C ablation: when the verigraph is triggered, also "
                        "run vanilla SearchR1 (no verigraph) at this extended budget. "
                        "This isolates the contribution of extra reasoning turns from "
                        "the contribution of verigraph feedback. Skipped for samples "
                        "where verigraph was not triggered (vanilla 5-turn already "
                        "answered → 10-turn would emit the same answer).")
    # System-D (hint-style verigraph corrector, post-case-study redesign)
    p.add_argument("--corrector-v2-max-turns", type=int, default=8,
                   help="System-D budget when the v2 (hint-style) corrector is active.")
    p.add_argument("--corrector-v2-threshold", type=float, default=0.50,
                   help="System-D cosine threshold (relaxed from 0.60 → 0.50 so "
                        "active/passive paraphrases like 'directed by' ≡ 'is the "
                        "director of' (cos≈0.59) are not falsely rejected).")
    p.add_argument("--max-docs-per-sample", type=int, default=10)
    p.add_argument("--doc-max-words", type=int, default=500)
    # Verigraph extractors
    p.add_argument("--question-model", type=str,
                   default="doupari/Llama-3.2-1B-Instruct-question-think-search")
    p.add_argument("--document-model", type=str,
                   default="doupari/Llama-3.2-1B-Instruct-document")
    p.add_argument("--think-model", type=str,
                   default="doupari/Llama-3.2-1B-Instruct-question-think-search")
    p.add_argument("--question-task", type=str, default="question")
    p.add_argument("--think-task", type=str, default="think+search")
    p.add_argument("--graph-dtype", type=str, default="bfloat16")
    p.add_argument("--graph-device-map", type=str, default="auto")
    p.add_argument("--graph-base-model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--graph-max-new-tokens", type=int, default=512)
    p.add_argument("--graph-temperature", type=float, default=0.0)
    # Cosine
    p.add_argument("--cosine-threshold", type=float, default=0.60)
    p.add_argument("--cosine-doc-top-k", type=int, default=3,
                   help="Top-K candidate triples (by whole-triple cosine) considered "
                        "for the per-field-max alignment. Matches the run_online_verigraph "
                        "top-K convention.")
    p.add_argument("--encoder", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_raw_dataset(
    dataset: str,
    input_filename: str,
    *,
    limit: Optional[int],
    start: int,
) -> List[Dict[str, Any]]:
    path = pathlib.Path("datasets") / dataset / "claims" / input_filename
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = rows[start:]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def _load_existing_rows_and_uids(cases_path: pathlib.Path) -> Tuple[List[Dict[str, Any]], set]:
    """Reconstruct previously-saved CSV rows + UID set from the cases JSONL.

    JSONL is the single source of truth for resume; CSV is rewritten at the
    end of every dataset.  This avoids losing rows when CSV writing was
    interrupted.
    """
    rows: List[Dict[str, Any]] = []
    done: set = set()
    if not cases_path.exists():
        return rows, done
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            uid = obj.get("uid") or obj.get("question") or ""
            if uid:
                done.add(uid)
            csv_row = obj.get("csv_row")
            if isinstance(csv_row, dict):
                rows.append(csv_row)
    return rows, done


def _summarize(df: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(df))
    if n == 0:
        return {"n_samples": 0}
    abstain_mask = df["abstained"].astype(bool)
    answered = df[~abstain_mask]
    summary: Dict[str, Any] = {
        "n_samples": n,
        "em_mean_all": float(df["em"].mean()) if n else 0.0,
        "f1_mean_all": float(df["f1"].mean()) if n else 0.0,
        "abstain_rate": float(abstain_mask.mean()),
        "n_abstain": int(abstain_mask.sum()),
        "n_answered": int(len(answered)),
        "em_mean_when_answered": float(answered["em"].mean()) if len(answered) else 0.0,
        "f1_mean_when_answered": float(answered["f1"].mean()) if len(answered) else 0.0,
        "mode_counts": {k: int(v) for k, v in df["mode"].value_counts().to_dict().items()},
        "vanilla_em_mean": float(df["vanilla_em"].mean()),
        "vanilla_f1_mean": float(df["vanilla_f1"].mean()),
        "trigger_rate": float(df["triggered_verigraph"].astype(bool).mean()),
        "n_triggered": int(df["triggered_verigraph"].astype(bool).sum()),
    }
    triggered = df[df["triggered_verigraph"].astype(bool)]
    summary["em_mean_in_corrector_subset"] = (
        float(triggered["em"].mean()) if len(triggered) else 0.0
    )
    summary["vanilla_em_mean_in_corrector_subset"] = (
        float(triggered["vanilla_em"].mean()) if len(triggered) else 0.0
    )

    # System A / B / C / D head-to-head.  All four measured on the SAME samples,
    # apples-to-apples with the same scoring function (score_answer).
    if "control_em" in df.columns:
        sc: Dict[str, Any] = {
            "A_vanilla_short": {
                "description": "vanilla SearchR1, max_turns matches vanilla phase, verigraph always OFF",
                "em_mean_all":  float(df["vanilla_em"].mean()),
                "f1_mean_all":  float(df["vanilla_f1"].mean()),
            },
            "B_selective_verigraph_v1": {
                "description": "selective verigraph V1 (verbose, PASS/FAIL, threshold 0.60): vanilla short first; on trigger, V1 corrector",
                "em_mean_all":  float(df["em"].mean()),
                "f1_mean_all":  float(df["f1"].mean()),
                "abstain_rate": float(abstain_mask.mean()),
            },
            "C_vanilla_long": {
                "description": "vanilla SearchR1 with the EXTENDED budget, verigraph always OFF (control for extra turns)",
                "em_mean_all":  float(df["control_em"].mean()),
                "f1_mean_all":  float(df["control_f1"].mean()),
            },
        }
        deltas: Dict[str, float] = {
            "B_minus_A": float(df["em"].mean() - df["vanilla_em"].mean()),
            "C_minus_A": float(df["control_em"].mean() - df["vanilla_em"].mean()),
            "B_minus_C": float(df["em"].mean() - df["control_em"].mean()),
        }
        if "system_d_em" in df.columns:
            sc["D_hint_verigraph_v2"] = {
                "description": "System-D: hint-style verigraph V2 (short block, value-only no PASS/FAIL, threshold 0.50, paraphrase-detect, entity-masking, expected-answer-type, query-sanitizer)",
                "em_mean_all":  float(df["system_d_em"].mean()),
                "f1_mean_all":  float(df["system_d_f1"].mean()),
            }
            deltas["D_minus_A"] = float(df["system_d_em"].mean() - df["vanilla_em"].mean())
            deltas["D_minus_B"] = float(df["system_d_em"].mean() - df["em"].mean())
            deltas["D_minus_C"] = float(df["system_d_em"].mean() - df["control_em"].mean())
        sc["deltas"] = deltas
        summary["system_comparison"] = sc

        if len(triggered):
            sub: Dict[str, Any] = {
                "n": int(len(triggered)),
                "A_vanilla_short_em": float(triggered["vanilla_em"].mean()),
                "B_selective_em":      float(triggered["em"].mean()),
                "C_vanilla_long_em":   float(triggered["control_em"].mean()),
                "B_minus_C":           float(triggered["em"].mean() - triggered["control_em"].mean()),
            }
            if "system_d_em" in df.columns:
                sub["D_hint_verigraph_em"] = float(triggered["system_d_em"].mean())
                sub["D_minus_B"] = float(triggered["system_d_em"].mean() - triggered["em"].mean())
                sub["D_minus_C"] = float(triggered["system_d_em"].mean() - triggered["control_em"].mean())
            summary["system_comparison"]["on_triggered_subset"] = sub

    # Trajectory length breakdown
    by_n = {}
    for ns, grp in df.groupby("vanilla_num_turns"):
        by_n[int(ns)] = {
            "n": int(len(grp)),
            "em_mean": float(grp["em"].mean()),
            "vanilla_em_mean": float(grp["vanilla_em"].mean()),
            "control_em_mean": (
                float(grp["control_em"].mean()) if "control_em" in df.columns else None
            ),
            "abstain_rate": float(grp["abstained"].astype(bool).mean()),
        }
    summary["by_vanilla_num_turns"] = by_n
    return summary


def main() -> int:
    args = parse_args()

    # Resolve dataset → per-dataset limit
    dataset_limits: Optional[List[int]] = args.dataset_limits if args.dataset_limits else None
    if dataset_limits and len(dataset_limits) != len(args.datasets):
        if len(dataset_limits) == 1:
            dataset_limits = dataset_limits * len(args.datasets)
        else:
            raise ValueError(
                f"--dataset-limits length ({len(dataset_limits)}) must match "
                f"--datasets length ({len(args.datasets)})"
            )

    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Init heavy resources once
    logger.info("[corrector] init SearchR1 (model=%s, vanilla max_turns=%d)",
                args.searchr1_model, args.searchr1_max_turns)
    searchr1 = SearchR1Inference(
        model_id=args.searchr1_model,
        retriever_url=args.retriever_url,
        max_turns=int(args.searchr1_max_turns),
        max_new_tokens=int(args.searchr1_max_new_tokens),
        temperature=float(args.searchr1_temperature),
        topk=int(args.searchr1_top_k),
    )
    encoder = get_default_encoder(model_name=args.encoder, device=args.device)
    pool = ExtractorPool(args)

    combined_rows: List[Dict[str, Any]] = []

    for d_idx, dataset in enumerate(args.datasets):
        per_limit: Optional[int] = (
            dataset_limits[d_idx] if dataset_limits else None
        )
        if args.limit and args.limit > 0:
            per_limit = int(args.limit)
        raw_rows = _load_raw_dataset(
            dataset,
            args.input_filename,
            limit=per_limit,
            start=int(args.start),
        )

        out_dir = output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        cases_path = out_dir / f"online_corrector_{dataset}_cases.jsonl"
        csv_path = out_dir / f"online_corrector_{dataset}.csv"
        summary_path = out_dir / f"online_corrector_{dataset}_summary.json"

        # Resume: cases JSONL is the source of truth
        rows, done_uids = _load_existing_rows_and_uids(cases_path)
        if done_uids:
            logger.info(
                "[corrector] %s: resuming with %d done (rows recovered=%d)",
                dataset, len(done_uids), len(rows),
            )

        cases_handle = open(cases_path, "a", encoding="utf-8")
        try:
            for raw in tqdm(raw_rows, desc=f"[corrector:{dataset}]"):
                uid = raw.get("uid") or raw.get("id") or raw.get("question", "")
                if uid in done_uids:
                    continue
                try:
                    csv_row, case_record = _process_sample(
                        raw, dataset, pool, searchr1, encoder, args,
                    )
                    rows.append(csv_row)
                    cases_handle.write(
                        json.dumps(case_record, ensure_ascii=False, default=float) + "\n"
                    )
                    cases_handle.flush()
                    done_uids.add(uid)
                except Exception as exc:
                    logger.exception("[corrector] sample failed: %s", exc)
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            cases_handle.close()

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        summary = _summarize(df)
        summary["dataset"] = dataset
        summary["config"] = {
            "vanilla_trigger_threshold": int(args.vanilla_trigger_threshold),
            "searchr1_max_turns": int(args.searchr1_max_turns),
            "corrector_max_turns": int(args.corrector_max_turns),
            "control_max_turns": int(args.control_max_turns),
            "corrector_v2_max_turns": int(args.corrector_v2_max_turns),
            "corrector_v2_threshold": float(args.corrector_v2_threshold),
            "cosine_threshold": float(args.cosine_threshold),
            "cosine_doc_top_k": int(args.cosine_doc_top_k),
            "question_model": args.question_model,
            "document_model": args.document_model,
            "think_model": args.think_model,
            "encoder": args.encoder,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

        logger.info(
            "[corrector:%s] n=%d em_all=%.3f em_answered=%.3f abstain=%.3f trigger=%.3f",
            dataset,
            summary.get("n_samples", 0),
            summary.get("em_mean_all", 0.0),
            summary.get("em_mean_when_answered", 0.0),
            summary.get("abstain_rate", 0.0),
            summary.get("trigger_rate", 0.0),
        )
        sysc = summary.get("system_comparison") or {}
        if sysc:
            a = sysc["A_vanilla_short"]["em_mean_all"]
            b = sysc["B_selective_verigraph_v1"]["em_mean_all"]
            c = sysc["C_vanilla_long"]["em_mean_all"]
            dline = f"A={a:.3f} | B(v1)={b:.3f} | C={c:.3f}"
            if "D_hint_verigraph_v2" in sysc:
                d = sysc["D_hint_verigraph_v2"]["em_mean_all"]
                dline += f" | D(v2-hint)={d:.3f}"
                dline += (
                    f"  ::  B-A={sysc['deltas']['B_minus_A']:+.3f} "
                    f"C-A={sysc['deltas']['C_minus_A']:+.3f} "
                    f"D-A={sysc['deltas']['D_minus_A']:+.3f} "
                    f"D-B={sysc['deltas']['D_minus_B']:+.3f} "
                    f"D-C={sysc['deltas']['D_minus_C']:+.3f}"
                )
            else:
                dline += (
                    f"  ::  B-A={sysc['deltas']['B_minus_A']:+.3f} "
                    f"B-C={sysc['deltas']['B_minus_C']:+.3f}"
                )
            logger.info("[corrector:%s] %s", dataset, dline)

        combined_rows.extend(rows)

    # Combined CSV / summary
    if len(args.datasets) > 1 and combined_rows:
        combined_df = pd.DataFrame(combined_rows)
        combined_df.to_csv(output_root / "online_corrector_all.csv", index=False)
        combined_summary = _summarize(combined_df)
        combined_summary["datasets"] = list(args.datasets)
        with open(output_root / "online_corrector_all_summary.json", "w", encoding="utf-8") as f:
            json.dump(combined_summary, f, indent=2, ensure_ascii=False, default=float)
        logger.info(
            "[corrector:ALL] n=%d em_all=%.3f em_answered=%.3f abstain=%.3f trigger=%.3f",
            combined_summary.get("n_samples", 0),
            combined_summary.get("em_mean_all", 0.0),
            combined_summary.get("em_mean_when_answered", 0.0),
            combined_summary.get("abstain_rate", 0.0),
            combined_summary.get("trigger_rate", 0.0),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
