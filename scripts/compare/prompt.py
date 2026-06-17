"""Prompts for SearchR1 evidence-grounded verification."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _truncate_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = "\n".join(str(x) for x in value)
    else:
        text = str(value)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def build_prompt(
    check_name: str,
    sample: Dict[str, Any],
    question_graph_text: str,
    context: Dict[str, str],
    output_schema: Dict[str, Any],
    searchr1_answer: str = "",
    gold_answers: Optional[List[str]] = None,
) -> str:
    question = _truncate_text(sample.get("question", ""), 2000)
    gold_answers = gold_answers or []

    shared = f"""
You are an evidence-grounded verifier for SearchR1 retrieval traces.

Rules:
- Judge only from the provided fields.
- Do not use outside knowledge.
- Return JSON only with the key: label.
- The JSON must satisfy this Pydantic schema:
{json.dumps(output_schema, ensure_ascii=False)}

Question:
{question}

Question graph:
{question_graph_text}
""".strip()

    if check_name == "question_graph_vs_document":
        task = f"""
Task: Decide whether this turn's document text provides evidence for the question graph.
Label true if the document contains the final answer or a necessary intermediate entity/relation/hop represented by the question graph.
Label false if the document is off-topic, only matches surface words, or does not provide evidence needed to resolve the question graph.
Allowed labels: true, false.

Turn subquery:
{context["subquery"]}

Turn document text:
{context["document_text"]}
""".strip()
    elif check_name == "question_graph_vs_think":
        task = f"""
Task: Decide whether this turn's SearchR1 think is aligned with the question graph.
Label true if the think identifies a useful subgoal, entity, relation, or hop needed to resolve the question graph.
Label false if the think is empty, irrelevant, contradicts the question graph, or moves to an unsupported path.
Allowed labels: true, false.

Turn think:
{context["think"]}
""".strip()
    elif check_name == "think_vs_query":
        task = f"""
Task: Decide whether the turn query is a natural search action from the turn think.
Label true if the query searches for the entity, relation, missing fact, or subgoal stated or clearly implied by the think.
Label false if the think is empty, or if the query is unrelated, contradictory, overly broad in a way that loses the stated subgoal, or drifts away.
Allowed labels: true, false.

Turn think:
{context["think"]}

Turn subquery:
{context["subquery"]}
""".strip()
    elif check_name == "document_vs_next_think":
        task = f"""
Task: Compare this turn's document text with the next turn's think.
Use the turn query and question graph to decide what information is relevant.

Decision rule:
1. First decide whether the document contains relevant evidence for the current subquery or question graph.
2. If the document contains relevant evidence:
   - Label true if the next think extracts the relevant evidence correctly.
   - Label fail if the next think misses the evidence, misreads it, extracts the wrong fact, or attributes an entity to the wrong relation.
   - In this case, prefer fail over hallu even if the wrong entity appears elsewhere in the document.
3. If the document does not contain relevant evidence:
   - Label hallu if the next think makes a relevant factual claim that is not supported by the document.
   - Label false if the next think also does not make a relevant factual claim from the document.

Allowed labels: true, false, hallu, fail.

Turn subquery:
{context["subquery"]}

This turn document text:
{context["document_text"]}

Next turn think:
{context["next_think"]}
""".strip()
    elif check_name == "searchr1_final_answer_correct":
        task = f"""
Task: Decide whether SearchR1's final answer is semantically equivalent to any gold answer or alias.
Label true if equivalent.
Label false if not equivalent, missing, too broad, too narrow, or contradictory.
Allowed labels: true, false.

SearchR1 final answer:
{searchr1_answer}

Gold answers and aliases:
{json.dumps(gold_answers, ensure_ascii=False)}
""".strip()
    else:
        raise ValueError(f"Unknown check: {check_name}")

    return shared + "\n\n" + task
