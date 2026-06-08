"""Online GraphQA evaluation with Veri-Graph extraction models.

This runner does not read precomputed triplet JSON files.  For each raw QA
sample it:

  1. retrieves documents online (BM25 server or SearchR1),
  2. extracts Q/D/T graphs with doupari Veri-Graph models,
  3. builds a GraphSample,
  4. runs stepwise cosine-only doc/think checking + slot filling.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from tqdm import tqdm

from direct import Direct
from graphqa.alignment import DEFAULT_ALIGN_THRESHOLDS, compute_sample_alignment
from graphqa.data import GraphSample, StepEvidence, Triple
from graphqa.data.schema import is_unknown
from graphqa.data.loader import DATASETS, _query_to_triples
from graphqa.data.schema import GraphStep
from graphqa.llm_qa import (
    _best_field_match_from_topk,
    _best_match_in_pool_for_kind,
    _fill_slots_from_aligned_triple,
    _format_triplet_fill_trace,
    _slot_id,
)
from graphqa.qa import _is_yesno_question, score_answer
from graphqa.tasi.embedding import get_default_encoder
from search_r1 import SearchR1Inference


logger = logging.getLogger(__name__)


SYSTEM_PROMPTS = {
    "document": (
        "You are a knowledge graph extraction expert. "
        "Extract all factual triples from the given document. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "question": (
        "You are a knowledge graph extraction expert. "
        "Given a multi-hop question, extract the reasoning triples that represent the relationships needed to answer it. "
        "Do not output generic entity definition/type triples such as (ENT1) [SEP] is [SEP] a person, location, country, or entity. "
        "Use placeholders only inside relation triples that are necessary for solving the question. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "think": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step from a chain-of-thought, extract the factual triples it asserts. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "think+search": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step and its search query, extract the factual triples it asserts. "
        "For facts already stated, use concrete values. For facts still being searched, use placeholders like (ENT1), (ENT2). "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "think+nosearch": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step from a chain-of-thought, extract the factual triples it asserts. "
        "For facts already stated, use concrete values. For facts still being searched, use placeholders like (ENT1), (ENT2). "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
}


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "")).strip()


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _clean_generated_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", s)
    s = s.strip().strip("`").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def parse_graph_output(text: str) -> List[Triple]:
    """Model text -> valid Triple list, preserving raw line text."""
    triples: List[Triple] = []
    seen = set()
    for raw_line in str(text or "").splitlines():
        line = _clean_generated_line(raw_line)
        if "[SEP]" not in line:
            continue
        triple = Triple.from_str(line)
        if triple is None:
            continue
        key = (
            " ".join(triple.head.split()),
            " ".join(triple.relation.split()),
            " ".join(triple.tail.split()),
            " ".join((triple.context or "").split()),
        )
        if key in seen:
            continue
        seen.add(key)
        triples.append(triple)
    return triples


def triples_to_raw(triples: Sequence[Triple]) -> List[str]:
    out: List[str] = []
    for t in triples:
        if t.raw:
            out.append(t.raw)
        elif t.context:
            out.append(f"{t.head} [SEP] {t.relation} [SEP] {t.tail} [PREP] {t.context}")
        else:
            out.append(f"{t.head} [SEP] {t.relation} [SEP] {t.tail}")
    return out


@dataclass
class GraphGenResult:
    text: str
    triples: List[Triple]


class VeriGraphExtractor:
    """Small HF causal-LM wrapper for the fine-tuned graph extractors."""

    def __init__(
        self,
        model_id: str,
        *,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        base_model_id: str = "",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        peft_config = None
        try:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(model_id)
        except Exception:
            peft_config = None

        if peft_config is not None:
            self.tokenizer, self.model = self._load_peft_adapter(
                model_id,
                dtype=dtype,
                device_map=device_map,
                base_model_id=base_model_id,
                original_error=None,
                peft_config=peft_config,
            )
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device_map,
                    attn_implementation="eager",
                )
            except Exception as exc:
                try:
                    self.tokenizer, self.model = self._load_peft_adapter(
                        model_id,
                        dtype=dtype,
                        device_map=device_map,
                        base_model_id=base_model_id,
                        original_error=exc,
                    )
                except Exception as peft_exc:
                    if "Unrecognized model" in str(exc) or "model_type" in str(exc):
                        raise peft_exc from exc
                    raise exc
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None
        self.model.eval()

    @staticmethod
    def _load_peft_adapter(
        model_id: str,
        *,
        dtype: torch.dtype,
        device_map: str,
        base_model_id: str,
        original_error: Optional[Exception],
        peft_config: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Load LoRA/PEFT adapter-only graph extractor repos.

        The doupari Veri-Graph repos are published as adapters: they have
        adapter_config.json and adapter_model.safetensors, but no config.json.
        AutoModelForCausalLM therefore cannot load them directly.
        """
        try:
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as peft_exc:
            raise RuntimeError(
                f"{model_id} looks like a PEFT adapter repo, but `peft` is not "
                "installed in this environment. Install it in the graphcheck "
                "conda env, e.g. `python -m pip install peft`."
            ) from peft_exc

        if peft_config is None:
            peft_config = PeftConfig.from_pretrained(model_id)
        resolved_base = (base_model_id or getattr(peft_config, "base_model_name_or_path", "") or "").strip()
        if not resolved_base:
            raise RuntimeError(
                f"{model_id} looks like a PEFT adapter repo, but no base model "
                "could be resolved from adapter_config.json. Pass "
                "`--graph-base-model` explicitly."
            ) from original_error

        logger.info("[graph-model] %s is a PEFT adapter; base=%s", model_id, resolved_base)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            logger.info("[graph-model] tokenizer fallback to base model %s", resolved_base)
            tokenizer = AutoTokenizer.from_pretrained(resolved_base)

        base = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, model_id)
        return tokenizer, model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _format_prompt(self, task: str, user_content: str) -> str:
        system = SYSTEM_PROMPTS[task]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        return f"{system}\n\n{user_content}\n"

    def generate(self, task: str, user_content: str) -> GraphGenResult:
        if task not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown graph task: {task}")
        prompt = self._format_prompt(task, user_content)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "max_length": None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = output[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return GraphGenResult(text=text, triples=parse_graph_output(text))


class ExtractorPool:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._pool: Dict[str, VeriGraphExtractor] = {}
        self._failed: Dict[str, str] = {}

    def get(self, model_id: str) -> VeriGraphExtractor:
        if model_id in self._failed:
            raise RuntimeError(f"graph model {model_id} failed to load earlier: {self._failed[model_id]}")
        if model_id not in self._pool:
            logger.info("[graph-model] loading %s", model_id)
            try:
                self._pool[model_id] = VeriGraphExtractor(
                    model_id,
                    torch_dtype=self.args.graph_dtype,
                    device_map=self.args.graph_device_map,
                    max_new_tokens=self.args.graph_max_new_tokens,
                    temperature=self.args.graph_temperature,
                    base_model_id=self.args.graph_base_model,
                )
            except Exception as exc:
                self._failed[model_id] = str(exc)
                raise
        return self._pool[model_id]


def make_graph_input(task: str, *, question: str = "", document: str = "",
                     think: str = "", search_query: str = "") -> str:
    if task == "document":
        return f"Document:\n{document}"
    if task == "question":
        return f"Question:\n{question}"
    if task in ("think", "think+nosearch"):
        return f"Reasoning step:\n{think}"
    if task == "think+search":
        return f"Reasoning step:\n{think}\n\nSearch query:\n{search_query}"
    raise ValueError(f"unknown graph task: {task}")


def load_raw_dataset(dataset: str, input_filename: str,
                     limit: Optional[int], start: int = 0) -> List[Dict[str, Any]]:
    path = pathlib.Path("datasets") / dataset / "claims" / input_filename
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _docs_from_information_block(block: str) -> List[str]:
    docs: List[str] = []
    for part in re.split(r"(?:^|\n)\s*Doc\s+\d+\s*", block or ""):
        doc = _normalize_text(part)
        if doc.startswith("(Title: "):
            docs.append(doc)
    return docs


def _parse_searchr1_steps(
    full_response: str,
    retrieval_turns: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return SearchR1 steps in reasoning order.

    Each item carries the think text, its search query, and the documents
    returned immediately after that search.  This is what lets the online
    checker force slot filling to follow the thinking-step order.
    """
    query_fallbacks = [str(t.get("query", "") or "") for t in retrieval_turns]
    steps: List[Dict[str, Any]] = []
    for idx, match in enumerate(re.finditer(r"<think>(.*?)</think>", full_response or "", re.DOTALL)):
        think = _normalize_text(match.group(1))
        tail = (full_response or "")[match.end():]
        next_think = tail.find("<think>")
        next_answer = tail.find("<answer>")
        stops = [x for x in (next_think, next_answer) if x >= 0]
        window = tail[: min(stops)] if stops else tail
        sm = re.search(r"<search>(.*?)</search>", window, flags=re.DOTALL)
        query = _normalize_text(sm.group(1)) if sm else ""
        if not query and idx < len(query_fallbacks):
            query = _normalize_text(query_fallbacks[idx])
        im = re.search(r"<information>(.*?)</information>", window, flags=re.DOTALL)
        docs = _docs_from_information_block(im.group(1) if im else "")
        if think:
            steps.append({
                "think": think,
                "search_query": query,
                "documents": docs,
            })
    return steps


def make_direct_retriever(dataset: str, args: argparse.Namespace) -> Direct:
    from argparse import Namespace

    direct_args = Namespace(
        dataset=dataset,
        input_filename=args.input_filename,
        direct_filename=None,
        base_model_name="google/flan-t5-xl",
        setting="open-book",
        bm25_top_k=args.bm25_top_k,
        use_searchr1=False,
        searchr1_top_k=args.searchr1_top_k,
        searchr1_max_turns=args.searchr1_max_turns,
        use_total_search_results=False,
        retriever_url=args.retriever_url,
    )
    return Direct(direct_args)


def retrieve_documents_bm25(raw: Dict[str, Any],
                            direct: Direct,
                            args: argparse.Namespace) -> Tuple[List[str], Dict[str, Any], List[str]]:
    docs, info = direct.retrieve_evidence(
        raw.get("question", ""),
        raw.get("gold_id_list", []) or [],
        raw.get("gold_evidence_list", []) or [],
        top_k=args.bm25_top_k,
    )
    queries = [raw.get("question", "")]
    return docs, info, queries


def retrieve_documents_searchr1(raw: Dict[str, Any],
                                searchr1: SearchR1Inference,
                                args: argparse.Namespace) -> Tuple[List[str], Dict[str, Any], List[str]]:
    if args.searchr1_verbose:
        info = searchr1.infer(raw.get("question", ""), verbose=True)
    else:
        # SearchR1Inference currently prints every full response unconditionally.
        # Keep online experiment logs readable unless verbose debugging is asked.
        with contextlib.redirect_stdout(io.StringIO()):
            info = searchr1.infer(raw.get("question", ""), verbose=False)
    docs = (
        info.get("total_search_results", [])
        if args.searchr1_use_total_results
        else info.get("last_search_results_list", [])
    )
    queries = [str(t.get("query", "") or "") for t in info.get("retrieval_turns", []) or []]
    return list(docs or []), dict(info or {}), queries


def answer_with_final_q_hint(
    question: str,
    final_q: Sequence[Triple],
    searchr1: SearchR1Inference,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    graph_hint = "\n".join(triples_to_raw(final_q))
    if args.searchr1_verbose:
        return searchr1.infer_with_graph_hint(question, graph_hint, verbose=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return searchr1.infer_with_graph_hint(question, graph_hint, verbose=False)


def answer_with_subgoal_hint(
    question: str,
    current_q: Sequence[Triple],
    target_triple: Triple,
    searchr1: SearchR1Inference,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    graph_hint = "\n".join(triples_to_raw(current_q))
    target = _triple_line(target_triple)
    if args.searchr1_verbose:
        return searchr1.infer_with_subgoal(
            question,
            graph_hint,
            target,
            verbose=True,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        return searchr1.infer_with_subgoal(
            question,
            graph_hint,
            target,
            verbose=False,
        )


def _triple_line(t: Triple) -> str:
    if t.raw:
        return t.raw
    if t.context:
        return f"{t.head} [SEP] {t.relation} [SEP] {t.tail} [PREP] {t.context}"
    return f"{t.head} [SEP] {t.relation} [SEP] {t.tail}"


_GENERIC_SLOT_TYPES = {
    "an individual", "a person", "person", "individual",
    "a location", "location", "a place", "place",
    "a country", "country", "a nation", "nation",
    "a city", "city", "a town", "town", "a village", "village",
    "a state", "state", "a province", "province", "a region", "region",
    "a date", "date", "a year", "year", "a number", "number",
    "a film", "film", "a movie", "movie", "a book", "book",
    "a novel", "novel", "a song", "song", "an album", "album",
    "a company", "company", "an organization", "organization",
    "an entity", "entity", "an organization", "organization",
    "an organisation", "organisation",
}


def _is_generic_slot_type(value: str) -> bool:
    return " ".join((value or "").lower().strip().split()) in _GENERIC_SLOT_TYPES


def _is_question_definition_triple(t: Triple) -> bool:
    """Generic slot-type hints should not become solver Q triples."""
    rel = " ".join((t.relation or "").lower().strip().split())
    if rel not in {"is", "are", "was", "were", "be"}:
        return False
    if t.head_unknown and _is_generic_slot_type(t.tail):
        return True
    if t.tail_unknown and _is_generic_slot_type(t.head):
        return True
    return False


def _split_question_triples(triples: Sequence[Triple]) -> Tuple[List[Triple], List[Triple]]:
    q: List[Triple] = []
    q_def: List[Triple] = []
    for t in triples:
        if _is_question_definition_triple(t):
            q_def.append(t)
        else:
            q.append(t)
    return q, q_def


def _doc_rescue_queries(
    final_q: Sequence[Triple],
    remaining_slots: Sequence[str],
    *,
    max_queries: int,
) -> List[str]:
    remaining = set(remaining_slots or [])
    queries: List[str] = []
    seen = set()
    for t in final_q:
        h_sid = _slot_id(t.head) if t.head_unknown else None
        tail_sid = _slot_id(t.tail) if t.tail_unknown else None
        if h_sid not in remaining and tail_sid not in remaining:
            continue
        if t.head_unknown and t.tail_unknown:
            continue
        if t.tail_unknown:
            if not t.head or _is_generic_slot_type(t.head):
                continue
            query = f"{t.head} {t.relation}".strip()
        else:
            if not t.tail or _is_generic_slot_type(t.tail):
                continue
            if t.relation.lower().strip() in {"is", "are", "was", "were"}:
                continue
            query = f"{t.tail} {t.relation}".strip()
        query = " ".join(query.split())
        key = query.lower()
        if len(query) < 3 or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def rescue_unfilled_slots_with_docs(
    sample: GraphSample,
    trace: Dict[str, Any],
    check: Dict[str, Any],
    *,
    args: argparse.Namespace,
    pool: ExtractorPool,
    searchr1: Optional[SearchR1Inference],
    round_number: int = 1,
) -> List[Dict[str, Any]]:
    if args.doc_rescue_rounds <= 0 or searchr1 is None:
        return []
    remaining = list(check.get("remaining_slots", []) or [])
    if not remaining:
        return []

    d_extractor = pool.get(args.document_model)
    per_doc = trace.get("document_graph", {}).setdefault("per_document", [])
    seen_docs = {str(d.get("document", "") or "") for d in per_doc}
    seen_queries = set(str(q).lower() for q in sample.search_queries)
    rescue_records: List[Dict[str, Any]] = []

    for round_idx in range(1):
        queries = _doc_rescue_queries(
            check["final_Q"],
            check.get("remaining_slots", []) or [],
            max_queries=args.doc_rescue_max_queries,
        )
        queries = [q for q in queries if q.lower() not in seen_queries]
        if not queries:
            break

        round_docs: List[str] = []
        query_records: List[Dict[str, Any]] = []
        for query in queries:
            seen_queries.add(query.lower())
            docs = searchr1._search(query)[: args.searchr1_top_k]
            docs = _dedupe_texts(docs, args.doc_max_words)
            kept_docs = []
            for doc in docs:
                if doc in seen_docs:
                    continue
                seen_docs.add(doc)
                kept_docs.append(doc)
                round_docs.append(doc)
            query_records.append({
                "query": query,
                "n_retrieved": len(docs),
                "n_new_docs": len(kept_docs),
            })

        new_doc_records: List[Dict[str, Any]] = []
        for doc in round_docs[: args.doc_rescue_max_docs]:
            d_result = d_extractor.generate(
                "document",
                make_graph_input("document", document=doc),
            )
            sample.D.extend(d_result.triples)
            doc_record = {
                "doc_index": len(per_doc),
                "document": doc,
                "raw_graph": d_result.text,
                "triples": triples_to_raw(d_result.triples),
                "rescue_round": int(round_number),
            }
            per_doc.append(doc_record)
            new_doc_records.append(doc_record)

        trace["document_graph"]["num_documents"] = len(per_doc)
        trace["document_graph"]["triples"] = triples_to_raw(sample.D)
        start_turn = len(sample.search_queries)
        for offset, query in enumerate(queries):
            sample.Sr.extend(_query_to_triples(query, start_turn + offset))
        sample.search_queries.extend(queries)
        trace["search_graph"]["queries"] = list(sample.search_queries)
        trace["search_graph"]["triples"] = triples_to_raw(sample.Sr)
        rescue_records.append({
            "round": int(round_number),
            "remaining_before": list(check.get("remaining_slots", []) or []),
            "queries": query_records,
            "n_new_docs": len(new_doc_records),
            "new_documents": new_doc_records,
        })
        if not new_doc_records:
            break
    return rescue_records


def _dedupe_texts(texts: Iterable[str], max_words: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for text in texts:
        t = _truncate_words(_normalize_text(text), max_words)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def build_online_sample(
    raw: Dict[str, Any],
    dataset: str,
    args: argparse.Namespace,
    pool: ExtractorPool,
    direct: Optional[Direct],
    searchr1: Optional[SearchR1Inference],
) -> Tuple[GraphSample, Dict[str, Any]]:
    question = _normalize_text(raw.get("question", ""))

    if args.retrieval_mode == "searchr1":
        if searchr1 is None:
            raise RuntimeError("SearchR1 retriever is not initialized")
        documents, retrieval_info, search_queries = retrieve_documents_searchr1(raw, searchr1, args)
    else:
        if direct is None:
            raise RuntimeError("BM25 retriever is not initialized")
        documents, retrieval_info, search_queries = retrieve_documents_bm25(raw, direct, args)

    search_steps: List[Dict[str, Any]] = []
    if args.retrieval_mode == "searchr1":
        search_steps = _parse_searchr1_steps(
            str(retrieval_info.get("full_response", "") or ""),
            retrieval_info.get("retrieval_turns", []) or [],
        )
        step_docs: List[str] = []
        for step in search_steps:
            docs = _dedupe_texts(step.get("documents", []) or [], args.doc_max_words)
            step["documents"] = docs
            step_docs.extend(docs)
        documents = list(documents or []) + step_docs

    documents = _dedupe_texts(documents, args.doc_max_words)

    q_extractor = pool.get(args.question_model)
    q_result = q_extractor.generate(
        args.question_task,
        make_graph_input(args.question_task, question=question),
    )
    q_triples, q_def_triples = _split_question_triples(q_result.triples)

    d_extractor = pool.get(args.document_model)
    D: List[Triple] = []
    per_doc: List[Dict[str, Any]] = []
    doc_cache: Dict[str, GraphGenResult] = {}

    def graph_document(doc: str) -> GraphGenResult:
        if doc not in doc_cache:
            doc_cache[doc] = d_extractor.generate(
                "document",
                make_graph_input("document", document=doc),
            )
        return doc_cache[doc]

    for doc_i, doc in enumerate(documents[: args.max_docs_per_sample]):
        d_result = d_extractor.generate(
            "document",
            make_graph_input("document", document=doc),
        )
        doc_cache[doc] = d_result
        D.extend(d_result.triples)
        per_doc.append({
            "doc_index": doc_i,
            "document": doc,
            "raw_graph": d_result.text,
            "triples": triples_to_raw(d_result.triples),
        })

    Sr: List[Triple] = []
    clean_queries = [_normalize_text(q) for q in search_queries if _normalize_text(q)]
    if not clean_queries:
        clean_queries = [question]
    for turn, query in enumerate(clean_queries):
        Sr.extend(_query_to_triples(query, turn))

    T: List[Triple] = []
    T_steps: List[GraphStep] = []
    step_evidence: List[StepEvidence] = []
    think_records: List[Dict[str, Any]] = []
    if args.retrieval_mode == "searchr1" and search_steps:
        t_extractor = pool.get(args.think_model)
        for step_i, step in enumerate(search_steps[: args.max_think_steps]):
            think = str(step.get("think", "") or "")
            query = str(step.get("search_query", "") or "")
            docs_for_step = list(step.get("documents", []) or [])[: args.max_docs_per_sample]
            t_result = t_extractor.generate(
                args.think_task,
                make_graph_input(args.think_task, think=think, search_query=query),
            )
            doc_triples_for_step: List[Triple] = []
            for doc in docs_for_step:
                doc_triples_for_step.extend(graph_document(doc).triples)
            T.extend(t_result.triples)
            T_steps.append(GraphStep(
                step_index=step_i,
                step_text=think,
                triples=t_result.triples,
            ))
            step_evidence.append(StepEvidence(
                step_index=step_i,
                query=query,
                think_text=think,
                think_triples=t_result.triples,
                doc_triples=doc_triples_for_step,
                doc_texts=docs_for_step,
            ))
            think_records.append({
                "step_index": step_i,
                "think": think,
                "search_query": query,
                "n_step_docs": len(docs_for_step),
                "n_step_doc_triples": len(doc_triples_for_step),
                "raw_graph": t_result.text,
                "triples": triples_to_raw(t_result.triples),
                "doc_triples": triples_to_raw(doc_triples_for_step),
            })

    sample = GraphSample(
        uid=str(raw.get("uid") or raw.get("index") or ""),
        question=question,
        answer=str(raw.get("answer", "")),
        answer_aliases=list(raw.get("answer_aliases", []) or []),
        num_hops=int(raw.get("num_hops", 0) or 0),
        dataset=dataset,
        Q_def=q_def_triples,
        Q=q_triples,
        T=T,
        T_steps=T_steps,
        step_evidence=step_evidence,
        Sr=Sr,
        D=D,
        search_queries=clean_queries,
        predicted_answer=str(retrieval_info.get("predicted_answer", "") or ""),
        gold_id_list=list(raw.get("gold_id_list", []) or []),
    )

    trace = {
        "dataset": dataset,
        "index": raw.get("index"),
        "uid": sample.uid,
        "question": question,
        "answer": sample.answer,
        "answer_aliases": sample.answer_aliases,
        "retrieval_mode": args.retrieval_mode,
        "retrieval_info": retrieval_info,
        "question_graph": {
            "model": args.question_model,
            "task": args.question_task,
            "raw_graph": q_result.text,
            "raw_triples": triples_to_raw(q_result.triples),
            "triples": triples_to_raw(q_triples),
            "definition_triples": triples_to_raw(q_def_triples),
        },
        "document_graph": {
            "model": args.document_model,
            "num_documents": len(per_doc),
            "triples": triples_to_raw(D),
            "per_document": per_doc,
        },
        "think_graph": {
            "model": args.think_model,
            "task": args.think_task,
            "triples": triples_to_raw(T),
            "per_step": think_records,
        },
        "search_graph": {
            "queries": clean_queries,
            "triples": triples_to_raw(Sr),
        },
    }
    return sample, trace


def _select_next_unresolved_query_triple(
    q_cur: Sequence[Triple],
    exhausted: set[int],
) -> Tuple[int, Optional[Triple]]:
    candidates: List[Tuple[int, int, Triple]] = []
    for i, t in enumerate(q_cur):
        if i in exhausted:
            continue
        n_unknown = int(t.head_unknown) + int(t.tail_unknown)
        if n_unknown <= 0:
            continue
        candidates.append((n_unknown, i, t))
    if not candidates:
        return -1, None
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, idx, triple = candidates[0]
    return idx, triple


def build_online_sample_q_guided(
    raw: Dict[str, Any],
    dataset: str,
    args: argparse.Namespace,
    pool: ExtractorPool,
    searchr1: SearchR1Inference,
    encoder,
) -> Tuple[GraphSample, Dict[str, Any], Dict[str, Any]]:
    """Build graphs by letting unresolved Q triples drive SearchR1 subgoals.

    Unlike the default SearchR1-first path, this extracts Q first, then each
    selected unresolved Q triple becomes the next online reasoning/search
    target. Retrieved documents and generated think steps are immediately
    graphed and cosine-verified for that target.
    """
    question = _normalize_text(raw.get("question", ""))
    q_extractor = pool.get(args.question_model)
    q_result = q_extractor.generate(
        args.question_task,
        make_graph_input(args.question_task, question=question),
    )
    q_triples, q_def_triples = _split_question_triples(q_result.triples)

    d_extractor = pool.get(args.document_model)
    t_extractor = pool.get(args.think_model)
    doc_cache: Dict[str, GraphGenResult] = {}
    seen_docs: set[str] = set()
    seen_queries: set[str] = set()

    D: List[Triple] = []
    T: List[Triple] = []
    Sr: List[Triple] = []
    T_steps: List[GraphStep] = []
    step_evidence: List[StepEvidence] = []
    search_queries: List[str] = []
    per_doc: List[Dict[str, Any]] = []
    think_records: List[Dict[str, Any]] = []
    subgoal_runs: List[Dict[str, Any]] = []

    def graph_document(doc: str, *, subgoal_step: int) -> GraphGenResult:
        if doc not in doc_cache:
            d_result = d_extractor.generate(
                "document",
                make_graph_input("document", document=doc),
            )
            doc_cache[doc] = d_result
            seen_docs.add(doc)
            D.extend(d_result.triples)
            per_doc.append({
                "doc_index": len(per_doc),
                "document": doc,
                "raw_graph": d_result.text,
                "triples": triples_to_raw(d_result.triples),
                "q_guided_subgoal_step": int(subgoal_step),
            })
        return doc_cache[doc]

    filled: Dict[str, str] = {}
    exhausted: set[int] = set()
    step_records: List[Dict[str, Any]] = []
    abstained = False
    abstain_reason = ""
    last_subgoal_answer = ""

    for step_pos in range(1, max(1, int(args.cosine_max_steps)) + 1):
        q_cur = _apply_slot_values(q_triples, filled)
        if not _remaining_slot_ids(q_cur):
            break
        q_idx, q = _select_next_unresolved_query_triple(q_cur, exhausted)
        if q is None or q_idx < 0:
            break

        retrieval_info = answer_with_subgoal_hint(
            question,
            q_cur,
            q,
            searchr1,
            args,
        )
        last_subgoal_answer = str(retrieval_info.get("predicted_answer", "") or "")
        search_steps = _parse_searchr1_steps(
            str(retrieval_info.get("full_response", "") or ""),
            retrieval_info.get("retrieval_turns", []) or [],
        )

        raw_docs = (
            retrieval_info.get("total_search_results", [])
            if args.searchr1_use_total_results
            else retrieval_info.get("last_search_results_list", [])
        )
        step_docs: List[str] = []
        for st in search_steps:
            docs = _dedupe_texts(st.get("documents", []) or [], args.doc_max_words)
            st["documents"] = docs
            step_docs.extend(docs)
        documents = _dedupe_texts(list(raw_docs or []) + step_docs, args.doc_max_words)

        subgoal_doc_triples: List[Triple] = []
        for doc in documents[: args.max_docs_per_sample]:
            d_result = graph_document(doc, subgoal_step=step_pos)
            subgoal_doc_triples.extend(d_result.triples)

        clean_queries = [
            _normalize_text(str(t.get("query", "") or ""))
            for t in retrieval_info.get("retrieval_turns", []) or []
            if _normalize_text(str(t.get("query", "") or ""))
        ]
        if not clean_queries:
            clean_queries = [
                " ".join(x for x in (q.head, q.relation, q.tail) if x and not is_unknown(x))
            ]
        turn_start = len(search_queries)
        for offset, query in enumerate(clean_queries):
            key = query.lower()
            if not query or key in seen_queries:
                continue
            seen_queries.add(key)
            search_queries.append(query)
            Sr.extend(_query_to_triples(query, turn_start + offset))

        subgoal_think_triples: List[Triple] = []
        step_records_for_trace: List[Dict[str, Any]] = []
        if not search_steps and retrieval_info.get("reasoning_path"):
            search_steps = [{
                "think": str(retrieval_info.get("reasoning_path", "") or ""),
                "search_query": clean_queries[0] if clean_queries else "",
                "documents": documents[: args.max_docs_per_sample],
            }]

        for local_i, st in enumerate(search_steps[: args.max_think_steps]):
            think = str(st.get("think", "") or "")
            query = str(st.get("search_query", "") or "")
            docs_for_step = list(st.get("documents", []) or [])[: args.max_docs_per_sample]
            t_result = t_extractor.generate(
                args.think_task,
                make_graph_input(args.think_task, think=think, search_query=query),
            )
            doc_triples_for_step: List[Triple] = []
            for doc in docs_for_step:
                d_result = graph_document(doc, subgoal_step=step_pos)
                doc_triples_for_step.extend(d_result.triples)
            step_index = len(T_steps)
            T.extend(t_result.triples)
            subgoal_think_triples.extend(t_result.triples)
            T_steps.append(GraphStep(
                step_index=step_index,
                step_text=think,
                triples=t_result.triples,
            ))
            step_evidence.append(StepEvidence(
                step_index=step_index,
                query=query,
                think_text=think,
                think_triples=t_result.triples,
                doc_triples=doc_triples_for_step,
                doc_texts=docs_for_step,
            ))
            think_rec = {
                "step_index": step_index,
                "q_guided_subgoal_step": int(step_pos),
                "local_step_index": int(local_i),
                "think": think,
                "search_query": query,
                "n_step_docs": len(docs_for_step),
                "n_step_doc_triples": len(doc_triples_for_step),
                "raw_graph": t_result.text,
                "triples": triples_to_raw(t_result.triples),
                "doc_triples": triples_to_raw(doc_triples_for_step),
            }
            think_records.append(think_rec)
            step_records_for_trace.append(think_rec)

        doc_pool = subgoal_doc_triples or list(D)
        think_pool = subgoal_think_triples or list(T)
        d_idx, d_whole, d_score, d_inv, _d_detail, d_top = _best_field_match_from_topk(
            q,
            doc_pool,
            encoder,
            top_k=args.cosine_doc_top_k,
            field_threshold=args.cosine_threshold,
        )
        t_idx, t_whole, t_score, t_inv, _t_detail, t_top = _best_match_in_pool_for_kind(
            q,
            think_pool,
            encoder,
            kind="think",
            top_k=args.cosine_doc_top_k,
            field_threshold=args.cosine_threshold,
        )
        doc_ok = bool(d_score == d_score and d_score >= args.cosine_threshold)
        think_ok = bool(t_score == t_score and t_score >= args.cosine_threshold)
        gate_ok = _gate_passes(doc_ok, think_ok, args.cosine_gate_on)

        fills: Dict[str, str] = {}
        fill_src = ""
        if args.cosine_fill_source in ("doc", "doc_then_think") and doc_ok and 0 <= d_idx < len(doc_pool):
            fills = _fill_slots_from_aligned_triple(q, doc_pool[d_idx], used_inverse=d_inv)
            fill_src = "doc" if fills else ""
        if (
            not fills
            and args.cosine_fill_source in ("think", "doc_then_think")
            and think_ok
            and 0 <= t_idx < len(think_pool)
        ):
            fills = _fill_slots_from_aligned_triple(q, think_pool[t_idx], used_inverse=t_inv)
            fill_src = "think" if fills else ""

        new_fills = {k: v for k, v in fills.items() if k not in filled and v}
        if new_fills:
            filled.update(new_fills)
            exhausted.clear()
        else:
            exhausted.add(q_idx)

        rec = {
            "step": int(step_pos),
            "step_order": "q_guided",
            "q_idx": int(q_idx),
            "query_triple": _triple_line(q),
            "doc_ok": bool(doc_ok),
            "think_ok": bool(think_ok),
            "ok_pair": f"{int(doc_ok)}/{int(think_ok)}",
            "gate_ok": bool(gate_ok),
            "doc_score": float(d_score) if d_score == d_score else float("nan"),
            "think_score": float(t_score) if t_score == t_score else float("nan"),
            "doc_whole_cosine": float(d_whole) if d_whole == d_whole else float("nan"),
            "think_whole_cosine": float(t_whole) if t_whole == t_whole else float("nan"),
            "doc_index": int(d_idx),
            "think_index": int(t_idx),
            "fills": "|".join(f"{k}={v}" for k, v in new_fills.items()),
            "fill_source": fill_src,
            "doc_top_candidates": d_top,
            "think_top_candidates": t_top,
        }
        step_records.append(rec)
        subgoal_runs.append({
            "subgoal_step": int(step_pos),
            "q_idx": int(q_idx),
            "target_triple": _triple_line(q),
            "current_graph_hint": triples_to_raw(q_cur),
            "retrieval_info": retrieval_info,
            "parsed_search_steps": search_steps,
            "documents": documents[: args.max_docs_per_sample],
            "think_graph_steps": step_records_for_trace,
            "cosine_record": rec,
        })

        if not gate_ok and args.cosine_on_fail == "abstain":
            abstained = True
            abstain_reason = f"q_guided_cosine_gate_failed_step_{step_pos}_{rec['ok_pair']}"
            break

    final_q = _apply_slot_values(q_triples, filled)
    remaining = _remaining_slot_ids(final_q)
    doc_vals = [bool(r.get("doc_ok", False)) for r in step_records]
    think_vals = [bool(r.get("think_ok", False)) for r in step_records]
    doc_ok_all = bool(all(doc_vals)) if doc_vals else False
    think_ok_all = bool(all(think_vals)) if think_vals else False
    ok_pair = f"{int(doc_ok_all)}/{int(think_ok_all)}"
    step_ok_pairs = "|".join(str(r.get("ok_pair", "0/0")) for r in step_records)
    min_doc = min(
        [float(r["doc_score"]) for r in step_records if r.get("doc_score") == r.get("doc_score")],
        default=float("nan"),
    )
    min_think = min(
        [float(r["think_score"]) for r in step_records if r.get("think_score") == r.get("think_score")],
        default=float("nan"),
    )

    sample = GraphSample(
        uid=str(raw.get("uid") or raw.get("index") or ""),
        question=question,
        answer=str(raw.get("answer", "")),
        answer_aliases=list(raw.get("answer_aliases", []) or []),
        num_hops=int(raw.get("num_hops", 0) or 0),
        dataset=dataset,
        Q_def=q_def_triples,
        Q=q_triples,
        T=T,
        T_steps=T_steps,
        step_evidence=step_evidence,
        Sr=Sr,
        D=D,
        search_queries=search_queries,
        predicted_answer=last_subgoal_answer,
        gold_id_list=list(raw.get("gold_id_list", []) or []),
    )
    trace = {
        "dataset": dataset,
        "index": raw.get("index"),
        "uid": sample.uid,
        "question": question,
        "answer": sample.answer,
        "answer_aliases": sample.answer_aliases,
        "retrieval_mode": args.retrieval_mode,
        "trajectory_mode": "q_guided",
        "retrieval_info": {
            "mode": "q_guided_subgoals",
            "subgoal_runs": subgoal_runs,
            "predicted_answer": last_subgoal_answer,
        },
        "question_graph": {
            "model": args.question_model,
            "task": args.question_task,
            "raw_graph": q_result.text,
            "raw_triples": triples_to_raw(q_result.triples),
            "triples": triples_to_raw(q_triples),
            "definition_triples": triples_to_raw(q_def_triples),
        },
        "document_graph": {
            "model": args.document_model,
            "num_documents": len(per_doc),
            "triples": triples_to_raw(D),
            "per_document": per_doc,
        },
        "think_graph": {
            "model": args.think_model,
            "task": args.think_task,
            "triples": triples_to_raw(T),
            "per_step": think_records,
        },
        "search_graph": {
            "queries": search_queries,
            "triples": triples_to_raw(Sr),
        },
    }
    check = {
        "abstained": bool(abstained),
        "abstain_reason": abstain_reason,
        "filled_slots": dict(filled),
        "final_Q": final_q,
        "remaining_slots": remaining,
        "n_slot_steps": int(len(step_records)),
        "triplet_doc_ok": float(int(doc_ok_all)),
        "triplet_think_ok": float(int(think_ok_all)),
        "triplet_ok_pair": ok_pair,
        "triplet_step_ok_pairs": step_ok_pairs,
        "triplet_min_doc_score": float(min_doc),
        "triplet_min_think_score": float(min_think),
        "triplet_step_trace": _format_triplet_fill_trace(step_records),
        "triplet_gate_failed": bool(any(not _gate_passes(bool(r["doc_ok"]), bool(r["think_ok"]), args.cosine_gate_on) for r in step_records)),
        "triplet_fail_reason": abstain_reason,
        "triplet_threshold": float(args.cosine_threshold),
        "triplet_gate_on": args.cosine_gate_on,
        "triplet_on_fail": args.cosine_on_fail,
        "triplet_fill_source": args.cosine_fill_source,
        "triplet_step_order": "q_guided",
        "step_records": step_records,
    }
    return sample, trace, check


def _extract_observer_think(output_text: str) -> str:
    matches = re.findall(r"<think>(.*?)</think>", output_text or "", flags=re.DOTALL)
    if matches:
        return _normalize_text(matches[-1])
    text = re.sub(r"<search>.*?</search>", "", output_text or "", flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    return _normalize_text(text)


def _fill_conflicts(fills: Dict[str, str], filled: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    conflicts: Dict[str, Dict[str, str]] = {}
    for sid, val in fills.items():
        old = filled.get(sid)
        if not old:
            continue
        if " ".join(old.lower().split()) != " ".join(str(val or "").lower().split()):
            conflicts[sid] = {"old": old, "new": str(val or "")}
    return conflicts


def build_online_sample_observer(
    raw: Dict[str, Any],
    dataset: str,
    args: argparse.Namespace,
    pool: ExtractorPool,
    searchr1: SearchR1Inference,
    encoder,
) -> Tuple[GraphSample, Dict[str, Any], Dict[str, Any]]:
    """Build graphs with Q as a hidden observer over a free SearchR1 run.

    SearchR1 receives only the original question.  After each generated
    reasoning/search turn, the online controller extracts think triples and
    document triples, maps them to unresolved Q graph slots, and verifies a
    fill only when evidence actually aligns to a Q triple.
    """
    question = _normalize_text(raw.get("question", ""))
    q_extractor = pool.get(args.question_model)
    q_result = q_extractor.generate(
        args.question_task,
        make_graph_input(args.question_task, question=question),
    )
    q_triples, q_def_triples = _split_question_triples(q_result.triples)

    d_extractor = pool.get(args.document_model)
    t_extractor = pool.get(args.think_model)
    doc_cache: Dict[str, GraphGenResult] = {}
    seen_docs: set[str] = set()
    seen_queries: set[str] = set()

    D: List[Triple] = []
    T: List[Triple] = []
    Sr: List[Triple] = []
    T_steps: List[GraphStep] = []
    step_evidence: List[StepEvidence] = []
    search_queries: List[str] = []
    per_doc: List[Dict[str, Any]] = []
    think_records: List[Dict[str, Any]] = []
    observer_events: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []
    filled: Dict[str, str] = {}
    observer_state: Dict[str, Any] = {
        "abstained": False,
        "abstain_reason": "",
        "stop_reason": "",
    }

    def graph_document(doc: str, *, observer_turn: int) -> GraphGenResult:
        if doc not in doc_cache:
            d_result = d_extractor.generate(
                "document",
                make_graph_input("document", document=doc),
            )
            doc_cache[doc] = d_result
            seen_docs.add(doc)
            D.extend(d_result.triples)
            per_doc.append({
                "doc_index": len(per_doc),
                "document": doc,
                "raw_graph": d_result.text,
                "triples": triples_to_raw(d_result.triples),
                "observer_turn": int(observer_turn),
            })
        return doc_cache[doc]

    def observe_turn(event: Dict[str, Any]) -> Dict[str, Any]:
        if observer_state.get("abstained"):
            return {
                "stop": True,
                "abstain": True,
                "reason": observer_state.get("abstain_reason", "observer_abstained"),
            }

        observer_turn = int(event.get("turn", len(step_evidence)) or 0)
        query = _normalize_text(str(event.get("query", "") or ""))
        if query and query.lower() not in seen_queries:
            seen_queries.add(query.lower())
            search_queries.append(query)
            Sr.extend(_query_to_triples(query, len(search_queries) - 1))

        docs_for_step = _dedupe_texts(
            event.get("search_results", []) or [],
            args.doc_max_words,
        )[: args.max_docs_per_sample]
        doc_triples_for_step: List[Triple] = []
        for doc in docs_for_step:
            doc_triples_for_step.extend(graph_document(doc, observer_turn=observer_turn).triples)

        think = _extract_observer_think(str(event.get("output_text", "") or ""))
        t_result = GraphGenResult(text="", triples=[])
        if think and len(T_steps) < int(args.max_think_steps):
            t_result = t_extractor.generate(
                args.think_task,
                make_graph_input(args.think_task, think=think, search_query=query),
            )
            step_index = len(T_steps)
            T.extend(t_result.triples)
            T_steps.append(GraphStep(
                step_index=step_index,
                step_text=think,
                triples=t_result.triples,
            ))
        else:
            step_index = len(step_evidence)

        step_evidence.append(StepEvidence(
            step_index=step_index,
            query=query,
            think_text=think,
            think_triples=t_result.triples,
            doc_triples=doc_triples_for_step,
            doc_texts=docs_for_step,
        ))
        think_records.append({
            "step_index": int(step_index),
            "observer_turn": int(observer_turn),
            "think": think,
            "search_query": query,
            "n_step_docs": len(docs_for_step),
            "n_step_doc_triples": len(doc_triples_for_step),
            "raw_graph": t_result.text,
            "triples": triples_to_raw(t_result.triples),
            "doc_triples": triples_to_raw(doc_triples_for_step),
        })

        event_records: List[Dict[str, Any]] = []
        turn_exhausted: set[int] = set()
        while len(step_records) < int(args.cosine_max_steps):
            q_cur = _apply_slot_values(q_triples, filled)
            if not _remaining_slot_ids(q_cur):
                observer_state["stop_reason"] = "all_slots_filled"
                break

            attempts: List[Tuple[Tuple[float, float, float, float, float], Dict[str, Any], Dict[str, str], Dict[str, Dict[str, str]]]] = []
            for q_idx, q in enumerate(q_cur):
                if q_idx in turn_exhausted:
                    continue
                n_unknown = int(q.head_unknown) + int(q.tail_unknown)
                if n_unknown <= 0:
                    continue

                d_idx, d_whole, d_score, d_inv, _d_detail, d_top = _best_field_match_from_topk(
                    q,
                    doc_triples_for_step,
                    encoder,
                    top_k=args.cosine_doc_top_k,
                    field_threshold=args.cosine_threshold,
                )
                t_idx, t_whole, t_score, t_inv, _t_detail, t_top = _best_match_in_pool_for_kind(
                    q,
                    t_result.triples,
                    encoder,
                    kind="think",
                    top_k=args.cosine_doc_top_k,
                    field_threshold=args.cosine_threshold,
                )
                doc_ok = bool(d_score == d_score and d_score >= args.cosine_threshold)
                think_ok = bool(t_score == t_score and t_score >= args.cosine_threshold)
                if not (doc_ok or think_ok):
                    continue

                gate_ok = _gate_passes(doc_ok, think_ok, args.cosine_gate_on)
                fills: Dict[str, str] = {}
                fill_src = ""
                if args.cosine_fill_source in ("doc", "doc_then_think") and doc_ok and 0 <= d_idx < len(doc_triples_for_step):
                    fills = _fill_slots_from_aligned_triple(q, doc_triples_for_step[d_idx], used_inverse=d_inv)
                    fill_src = "doc" if fills else ""
                if (
                    not fills
                    and args.cosine_fill_source in ("think", "doc_then_think")
                    and think_ok
                    and 0 <= t_idx < len(t_result.triples)
                ):
                    fills = _fill_slots_from_aligned_triple(q, t_result.triples[t_idx], used_inverse=t_inv)
                    fill_src = "think" if fills else ""

                conflicts = _fill_conflicts(fills, filled)
                new_fills = {
                    k: v for k, v in fills.items()
                    if k not in filled and v and not is_unknown(v)
                }
                problem_codes: List[str] = []
                if not gate_ok:
                    if think_ok and not doc_ok:
                        problem_codes.append("P1_THINK_UNSUPPORTED_BY_DOC")
                    else:
                        problem_codes.append("P1_COSINE_GATE_FAILED")
                if gate_ok and not fills:
                    problem_codes.append("P2_NO_SLOT_FILL")
                if conflicts:
                    problem_codes.append("P3_SLOT_CONFLICT")

                rec = {
                    "step": int(len(step_records) + len(attempts) + 1),
                    "observer_turn": int(observer_turn),
                    "think_step_index": int(step_index),
                    "think_step_query": query,
                    "step_order": "observer",
                    "q_idx": int(q_idx),
                    "query_triple": _triple_line(q),
                    "doc_ok": bool(doc_ok),
                    "think_ok": bool(think_ok),
                    "ok_pair": f"{int(doc_ok)}/{int(think_ok)}",
                    "gate_ok": bool(gate_ok),
                    "problem_codes": "|".join(problem_codes),
                    "doc_score": float(d_score) if d_score == d_score else float("nan"),
                    "think_score": float(t_score) if t_score == t_score else float("nan"),
                    "doc_whole_cosine": float(d_whole) if d_whole == d_whole else float("nan"),
                    "think_whole_cosine": float(t_whole) if t_whole == t_whole else float("nan"),
                    "doc_index": int(d_idx),
                    "think_index": int(t_idx),
                    "fills": "|".join(f"{k}={v}" for k, v in new_fills.items()),
                    "attempted_fills": "|".join(f"{k}={v}" for k, v in fills.items()),
                    "fill_source": fill_src,
                    "conflicts": json.dumps(conflicts, ensure_ascii=False) if conflicts else "",
                    "doc_top_candidates": d_top,
                    "think_top_candidates": t_top,
                }
                best_score = max(_score_or_neg_inf(d_score), _score_or_neg_inf(t_score))
                key = (
                    float(bool(gate_ok and new_fills)),
                    float(bool(gate_ok)),
                    float(bool(new_fills)),
                    float(best_score),
                    -float(n_unknown),
                )
                attempts.append((key, rec, new_fills, conflicts))

            if not attempts:
                break

            attempts.sort(key=lambda x: x[0], reverse=True)
            _key, rec, new_fills, _conflicts = attempts[0]
            rec["step"] = int(len(step_records) + 1)
            step_records.append(rec)
            event_records.append(rec)

            problem_codes = [
                code for code in str(rec.get("problem_codes", "") or "").split("|")
                if code
            ]
            if problem_codes and args.cosine_on_fail == "abstain":
                observer_state["abstained"] = True
                observer_state["abstain_reason"] = (
                    f"observer_{problem_codes[0]}_turn_{observer_turn}_step_{rec['step']}"
                )
                observer_state["stop_reason"] = observer_state["abstain_reason"]
                break

            if new_fills and rec.get("gate_ok"):
                filled.update(new_fills)
                turn_exhausted.clear()
            else:
                turn_exhausted.add(int(rec["q_idx"]))

            if not _remaining_slot_ids(_apply_slot_values(q_triples, filled)):
                observer_state["stop_reason"] = "all_slots_filled"
                break

        observer_events.append({
            "observer_turn": int(observer_turn),
            "query": query,
            "think": think,
            "documents": docs_for_step,
            "n_doc_triples": len(doc_triples_for_step),
            "n_think_triples": len(t_result.triples),
            "step_records": event_records,
            "filled_after_turn": dict(filled),
            "remaining_after_turn": _remaining_slot_ids(_apply_slot_values(q_triples, filled)),
        })

        if observer_state.get("abstained"):
            return {
                "stop": True,
                "abstain": True,
                "reason": observer_state.get("abstain_reason", "observer_abstained"),
            }
        if observer_state.get("stop_reason") == "all_slots_filled":
            return {"stop": True, "reason": "all_slots_filled"}
        return {}

    if args.searchr1_verbose:
        retrieval_info = searchr1.infer_with_observer(
            question,
            on_turn=observe_turn,
            verbose=True,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            retrieval_info = searchr1.infer_with_observer(
                question,
                on_turn=observe_turn,
                verbose=False,
            )

    final_q = _apply_slot_values(q_triples, filled)
    remaining = _remaining_slot_ids(final_q)
    if (
        remaining
        and args.cosine_on_fail == "abstain"
        and not observer_state.get("abstained")
    ):
        observer_state["abstained"] = True
        observer_state["abstain_reason"] = "observer_unfilled_slots_" + ",".join(remaining)
        observer_state["stop_reason"] = observer_state["abstain_reason"]

    doc_vals = [bool(r.get("doc_ok", False)) for r in step_records]
    think_vals = [bool(r.get("think_ok", False)) for r in step_records]
    doc_ok_all = bool(all(doc_vals)) if doc_vals else False
    think_ok_all = bool(all(think_vals)) if think_vals else False
    ok_pair = f"{int(doc_ok_all)}/{int(think_ok_all)}"
    step_ok_pairs = "|".join(str(r.get("ok_pair", "0/0")) for r in step_records)
    min_doc = min(
        [float(r["doc_score"]) for r in step_records if r.get("doc_score") == r.get("doc_score")],
        default=float("nan"),
    )
    min_think = min(
        [float(r["think_score"]) for r in step_records if r.get("think_score") == r.get("think_score")],
        default=float("nan"),
    )

    sample = GraphSample(
        uid=str(raw.get("uid") or raw.get("index") or ""),
        question=question,
        answer=str(raw.get("answer", "")),
        answer_aliases=list(raw.get("answer_aliases", []) or []),
        num_hops=int(raw.get("num_hops", 0) or 0),
        dataset=dataset,
        Q_def=q_def_triples,
        Q=q_triples,
        T=T,
        T_steps=T_steps,
        step_evidence=step_evidence,
        Sr=Sr,
        D=D,
        search_queries=search_queries,
        predicted_answer=str(retrieval_info.get("predicted_answer", "") or ""),
        gold_id_list=list(raw.get("gold_id_list", []) or []),
    )
    trace = {
        "dataset": dataset,
        "index": raw.get("index"),
        "uid": sample.uid,
        "question": question,
        "answer": sample.answer,
        "answer_aliases": sample.answer_aliases,
        "retrieval_mode": args.retrieval_mode,
        "trajectory_mode": "observer",
        "retrieval_info": {
            "mode": "observer_searchr1",
            "searchr1": retrieval_info,
            "observer_events": observer_events,
            "observer_stop_reason": observer_state.get("stop_reason", ""),
            "predicted_answer": sample.predicted_answer,
        },
        "question_graph": {
            "model": args.question_model,
            "task": args.question_task,
            "raw_graph": q_result.text,
            "raw_triples": triples_to_raw(q_result.triples),
            "triples": triples_to_raw(q_triples),
            "definition_triples": triples_to_raw(q_def_triples),
        },
        "document_graph": {
            "model": args.document_model,
            "num_documents": len(per_doc),
            "triples": triples_to_raw(D),
            "per_document": per_doc,
        },
        "think_graph": {
            "model": args.think_model,
            "task": args.think_task,
            "triples": triples_to_raw(T),
            "per_step": think_records,
        },
        "search_graph": {
            "queries": search_queries,
            "triples": triples_to_raw(Sr),
        },
    }
    check = {
        "abstained": bool(observer_state.get("abstained", False)),
        "abstain_reason": str(observer_state.get("abstain_reason", "")),
        "filled_slots": dict(filled),
        "final_Q": final_q,
        "remaining_slots": remaining,
        "n_slot_steps": int(len(step_records)),
        "triplet_doc_ok": float(int(doc_ok_all)),
        "triplet_think_ok": float(int(think_ok_all)),
        "triplet_ok_pair": ok_pair,
        "triplet_step_ok_pairs": step_ok_pairs,
        "triplet_min_doc_score": float(min_doc),
        "triplet_min_think_score": float(min_think),
        "triplet_step_trace": _format_triplet_fill_trace(step_records),
        "triplet_gate_failed": bool(
            any(not _gate_passes(bool(r["doc_ok"]), bool(r["think_ok"]), args.cosine_gate_on) for r in step_records)
        ),
        "triplet_fail_reason": str(observer_state.get("abstain_reason", "")),
        "triplet_threshold": float(args.cosine_threshold),
        "triplet_gate_on": args.cosine_gate_on,
        "triplet_on_fail": args.cosine_on_fail,
        "triplet_fill_source": args.cosine_fill_source,
        "triplet_step_order": "observer",
        "observer_stop_reason": str(observer_state.get("stop_reason", "")),
        "step_records": step_records,
    }
    return sample, trace, check


def selected_accuracy_report(
    df: pd.DataFrame,
    *,
    score_cols: Sequence[str],
    thresholds: Sequence[float],
    correct_col: str = "is_correct",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df.empty or correct_col not in df.columns:
        return out
    for col in score_cols:
        if col not in df.columns:
            continue
        rows = []
        scores = pd.to_numeric(df[col], errors="coerce")
        for tau in thresholds:
            selected = scores >= tau
            n_selected = int(selected.sum())
            if n_selected:
                acc = float(df.loc[selected, correct_col].astype(float).mean())
            else:
                acc = float("nan")
            rows.append({
                "threshold": float(tau),
                "coverage": float(n_selected / max(1, len(df))),
                "n_selected": n_selected,
                "selected_accuracy": acc,
            })
        out[col] = rows
    return out


def align_by_answer_report(
    df: pd.DataFrame,
    *,
    correct_cols: Sequence[str],
    score_cols: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df.empty:
        return out
    for correct_col in correct_cols:
        if correct_col not in df.columns:
            continue
        version = correct_col.removesuffix("_is_correct")
        version_rows: List[Dict[str, Any]] = []
        correct = df[correct_col].astype(bool)
        for label, mask in (("wrong", ~correct), ("correct", correct)):
            sub = df.loc[mask]
            row: Dict[str, Any] = {"answer_group": label, "n": int(len(sub))}
            for score_col in score_cols:
                if score_col in sub.columns and len(sub):
                    row[f"{score_col}_mean"] = float(pd.to_numeric(sub[score_col], errors="coerce").mean())
                elif score_col in df.columns:
                    row[f"{score_col}_mean"] = float("nan")
            version_rows.append(row)
        out[version] = version_rows
    if {"searchr1_is_correct", "final_q_hint_is_correct"}.issubset(df.columns):
        pair_rows: List[Dict[str, Any]] = []
        base = df["searchr1_is_correct"].astype(bool)
        hint = df["final_q_hint_is_correct"].astype(bool)
        for pair in PAIR_ORDER:
            b, h = pair.split("/")
            mask = (base == bool(int(b))) & (hint == bool(int(h)))
            sub = df.loc[mask]
            row = {"pair": pair, "n": int(len(sub))}
            for score_col in score_cols:
                if score_col in sub.columns and len(sub):
                    row[f"{score_col}_mean"] = float(pd.to_numeric(sub[score_col], errors="coerce").mean())
                elif score_col in df.columns:
                    row[f"{score_col}_mean"] = float("nan")
            pair_rows.append(row)
        out["searchr1/final_q_hint_pairs"] = pair_rows
    return out


def _apply_slot_values(Q: Sequence[Triple], filled: Dict[str, str]) -> List[Triple]:
    out: List[Triple] = []
    for t in Q:
        h = t.head
        r = t.relation
        tail = t.tail
        h_sid = _slot_id(h) if is_unknown(h) else None
        t_sid = _slot_id(tail) if is_unknown(tail) else None
        if h_sid and h_sid in filled:
            h = filled[h_sid]
        if t_sid and t_sid in filled:
            tail = filled[t_sid]
        raw = f"{h} [SEP] {r} [SEP] {tail}"
        if t.context:
            raw = f"{raw} [PREP] {t.context}"
        out.append(Triple(head=h, relation=r, tail=tail, context=t.context, raw=raw))
    return out


def _remaining_slot_ids(Q: Sequence[Triple]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in Q:
        for v in (t.head, t.tail):
            sid = _slot_id(v) if is_unknown(v) else None
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def _gate_passes(doc_ok: bool, think_ok: bool, gate_on: str) -> bool:
    gate_on = (gate_on or "doc").lower()
    if gate_on == "doc":
        return bool(doc_ok)
    if gate_on == "think":
        return bool(think_ok)
    if gate_on == "both":
        return bool(doc_ok and think_ok)
    if gate_on == "either":
        return bool(doc_ok or think_ok)
    return bool(doc_ok)


def _select_next_unchecked_query_triple(
    q_cur: Sequence[Triple],
    exhausted: set[int],
) -> Tuple[int, Optional[Triple]]:
    candidates: List[Tuple[int, int, int, Triple]] = []
    for i, t in enumerate(q_cur):
        if i in exhausted:
            continue
        n_unknown = int(t.head_unknown) + int(t.tail_unknown)
        # Prefer triples that still have slots; among those, use the one with
        # the fewest unknown fields. If Q is fully concrete, still check it.
        concrete_penalty = 0 if n_unknown > 0 else 1
        candidates.append((concrete_penalty, n_unknown, i, t))
    if not candidates:
        return -1, None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, idx, triple = candidates[0]
    return idx, triple


def _score_or_neg_inf(v: float) -> float:
    return float(v) if v == v else float("-inf")


def _dedupe_triples(triples: Iterable[Triple]) -> List[Triple]:
    out: List[Triple] = []
    seen = set()
    for t in triples:
        key = (
            " ".join((t.head or "").split()),
            " ".join((t.relation or "").split()),
            " ".join((t.tail or "").split()),
            " ".join((t.context or "").split()),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _doc_pool_for_check(
    sample: GraphSample,
    step_ev: Optional[StepEvidence],
    *,
    ordered_by_think: bool,
) -> List[Triple]:
    if step_ev is not None and step_ev.doc_triples:
        if ordered_by_think:
            return list(step_ev.doc_triples)
        return _dedupe_triples(list(step_ev.doc_triples) + list(sample.D))
    return list(sample.D)


def _select_query_triple_for_thinking_step(
    q_cur: Sequence[Triple],
    exhausted: set[int],
    doc_pool: Sequence[Triple],
    think_pool: Sequence[Triple],
    encoder,
    *,
    threshold: float,
    doc_top_k: int,
) -> Tuple[int, Optional[Triple]]:
    """Pick the question triple for the current thinking step.

    This makes the filling order follow the thinking-step order: step k chooses
    whichever current question triple best aligns with step k's think/doc graph.
    """
    best: Optional[Tuple[float, float, float, int, Triple]] = None
    for i, q in enumerate(q_cur):
        if i in exhausted:
            continue
        n_unknown = int(q.head_unknown) + int(q.tail_unknown)
        unknown_priority = 1.0 if n_unknown > 0 else 0.0
        _d_idx, _d_whole, d_score, _d_inv, _d_detail, _d_top = _best_field_match_from_topk(
            q,
            doc_pool,
            encoder,
            top_k=doc_top_k,
            field_threshold=threshold,
        )
        _t_idx, _t_whole, t_score, _t_inv, _t_detail, _t_top = _best_match_in_pool_for_kind(
            q,
            think_pool,
            encoder,
            kind="think",
            top_k=doc_top_k,
            field_threshold=threshold,
        )
        align_score = max(_score_or_neg_inf(d_score), _score_or_neg_inf(t_score))
        key = (unknown_priority, align_score, -float(n_unknown), -float(i))
        if best is None or key > best[:4]:
            best = (*key, q)
    if best is None:
        return -1, None
    idx = int(-best[3])
    return idx, best[4]


def _select_evidence_step_for_question_triple(
    q: Triple,
    step_evidence: Sequence[StepEvidence],
    used_steps: set[int],
    encoder,
    *,
    threshold: float,
    doc_top_k: int,
) -> Optional[StepEvidence]:
    """Choose the SearchR1 step evidence that best supports the chosen Q triple.

    In the question-driven mode, question graph order is the source of truth:
    we first select the Q triple with the fewest unknowns, then attach the most
    compatible think/search/doc step to that fill attempt.
    """
    best: Optional[Tuple[float, float, int, StepEvidence]] = None
    for fallback_pos, st in enumerate(step_evidence):
        step_idx = int(st.step_index if st.step_index is not None else fallback_pos)
        if step_idx in used_steps:
            continue
        _d_idx, d_whole, d_score, _d_inv, _d_detail, _d_top = _best_field_match_from_topk(
            q,
            st.doc_triples,
            encoder,
            top_k=doc_top_k,
            field_threshold=threshold,
        )
        _t_idx, t_whole, t_score, _t_inv, _t_detail, _t_top = _best_match_in_pool_for_kind(
            q,
            st.think_triples,
            encoder,
            kind="think",
            top_k=doc_top_k,
            field_threshold=threshold,
        )
        field_score = max(_score_or_neg_inf(d_score), _score_or_neg_inf(t_score))
        whole_score = max(_score_or_neg_inf(d_whole), _score_or_neg_inf(t_whole))
        key = (field_score, whole_score, -float(step_idx))
        if best is None or key > best[:3]:
            best = (*key, st)
    return best[3] if best is not None else None


def run_cosine_step_check(
    sample: GraphSample,
    encoder,
    *,
    threshold: float,
    max_steps: int,
    doc_top_k: int,
    on_fail: str,
    gate_on: str,
    fill_source: str,
    step_order: str,
) -> Dict[str, Any]:
    """Stepwise cosine-only doc/think check and slot filling.

    No TASI score is computed here.  Each selected question triple is matched
    against document triples and think triples by whole-triple cosine + known
    field-level min cosine.  The four 0/0, 1/0, 0/1, 1/1 states are recorded at
    both step level and sample level (all-step AND).
    """
    q_cur = list(sample.Q)
    filled: Dict[str, str] = {}
    exhausted: set[int] = set()
    used_evidence_steps: set[int] = set()
    step_records: List[Dict[str, Any]] = []
    abstained = False
    abstain_reason = ""

    max_steps = max(1, int(max_steps))
    ordered_by_think = (step_order == "think") and bool(sample.step_evidence)
    if ordered_by_think:
        loop_steps: List[Optional[StepEvidence]] = list(sample.step_evidence[:max_steps])
    else:
        loop_steps = [None] * max_steps

    for step_pos, step_ev in enumerate(loop_steps, start=1):
        if len(exhausted) >= len(q_cur):
            break
        if ordered_by_think:
            doc_pool = _doc_pool_for_check(sample, step_ev, ordered_by_think=ordered_by_think)
            think_pool = (
                list(step_ev.think_triples)
                if step_ev is not None and step_ev.think_triples
                else list(sample.T)
            )
            q_idx, q = _select_query_triple_for_thinking_step(
                q_cur,
                exhausted,
                doc_pool,
                think_pool,
                encoder,
                threshold=threshold,
                doc_top_k=doc_top_k,
            )
        else:
            q_idx, q = _select_next_unchecked_query_triple(q_cur, exhausted)
        if q is None or q_idx < 0:
            break
        if not ordered_by_think and sample.step_evidence:
            step_ev = _select_evidence_step_for_question_triple(
                q,
                sample.step_evidence,
                used_evidence_steps,
                encoder,
                threshold=threshold,
                doc_top_k=doc_top_k,
            )
            if step_ev is not None:
                used_evidence_steps.add(int(step_ev.step_index))
        doc_pool = _doc_pool_for_check(sample, step_ev, ordered_by_think=ordered_by_think)
        think_pool = (
            list(step_ev.think_triples)
            if step_ev is not None and step_ev.think_triples
            else list(sample.T)
        )

        d_idx, d_whole, d_score, d_inv, _d_detail, d_top = _best_field_match_from_topk(
            q,
            doc_pool,
            encoder,
            top_k=doc_top_k,
            field_threshold=threshold,
        )
        t_idx, t_whole, t_score, t_inv, _t_detail, t_top = _best_match_in_pool_for_kind(
            q,
            think_pool,
            encoder,
            kind="think",
            top_k=doc_top_k,
            field_threshold=threshold,
        )
        doc_ok = bool(d_score == d_score and d_score >= threshold)
        think_ok = bool(t_score == t_score and t_score >= threshold)
        gate_ok = _gate_passes(doc_ok, think_ok, gate_on)

        fills: Dict[str, str] = {}
        fill_src = ""
        if fill_source in ("doc", "doc_then_think") and doc_ok and 0 <= d_idx < len(doc_pool):
            fills = _fill_slots_from_aligned_triple(q, doc_pool[d_idx], used_inverse=d_inv)
            fill_src = "doc" if fills else ""
        if (
            not fills
            and fill_source in ("think", "doc_then_think")
            and think_ok
            and 0 <= t_idx < len(think_pool)
        ):
            fills = _fill_slots_from_aligned_triple(q, think_pool[t_idx], used_inverse=t_inv)
            fill_src = "think" if fills else ""

        new_fills = {k: v for k, v in fills.items() if k not in filled and v}
        if new_fills:
            filled.update(new_fills)
            q_cur = _apply_slot_values(sample.Q, filled)
            if not ordered_by_think:
                exhausted.clear()
        elif not ordered_by_think:
            exhausted.add(q_idx)

        rec = {
            "step": int(step_pos),
            "think_step_index": int(step_ev.step_index) if step_ev is not None else None,
            "think_step_query": str(step_ev.query) if step_ev is not None else "",
            "step_order": "think" if ordered_by_think else "question",
            "q_idx": int(q_idx),
            "query_triple": q.raw or f"{q.head} [SEP] {q.relation} [SEP] {q.tail}",
            "doc_ok": bool(doc_ok),
            "think_ok": bool(think_ok),
            "ok_pair": f"{int(doc_ok)}/{int(think_ok)}",
            "gate_ok": bool(gate_ok),
            "doc_score": float(d_score) if d_score == d_score else float("nan"),
            "think_score": float(t_score) if t_score == t_score else float("nan"),
            "doc_whole_cosine": float(d_whole) if d_whole == d_whole else float("nan"),
            "think_whole_cosine": float(t_whole) if t_whole == t_whole else float("nan"),
            "doc_index": int(d_idx),
            "think_index": int(t_idx),
            "fills": "|".join(f"{k}={v}" for k, v in new_fills.items()),
            "fill_source": fill_src,
            "doc_top_candidates": d_top,
            "think_top_candidates": t_top,
        }
        step_records.append(rec)

        if not gate_ok and on_fail == "abstain":
            abstained = True
            abstain_reason = f"cosine_gate_failed_step_{step_pos}_{rec['ok_pair']}"
            break

        if filled and not _remaining_slot_ids(q_cur):
            break

    doc_vals = [bool(r.get("doc_ok", False)) for r in step_records]
    think_vals = [bool(r.get("think_ok", False)) for r in step_records]
    doc_ok_all = bool(all(doc_vals)) if doc_vals else False
    think_ok_all = bool(all(think_vals)) if think_vals else False
    ok_pair = f"{int(doc_ok_all)}/{int(think_ok_all)}"
    step_ok_pairs = "|".join(str(r.get("ok_pair", "0/0")) for r in step_records)
    min_doc = min(
        [float(r["doc_score"]) for r in step_records if r.get("doc_score") == r.get("doc_score")],
        default=float("nan"),
    )
    min_think = min(
        [float(r["think_score"]) for r in step_records if r.get("think_score") == r.get("think_score")],
        default=float("nan"),
    )
    return {
        "abstained": bool(abstained),
        "abstain_reason": abstain_reason,
        "filled_slots": dict(filled),
        "final_Q": _apply_slot_values(sample.Q, filled),
        "remaining_slots": _remaining_slot_ids(q_cur),
        "n_slot_steps": int(len(step_records)),
        "triplet_doc_ok": float(int(doc_ok_all)),
        "triplet_think_ok": float(int(think_ok_all)),
        "triplet_ok_pair": ok_pair,
        "triplet_step_ok_pairs": step_ok_pairs,
        "triplet_min_doc_score": float(min_doc),
        "triplet_min_think_score": float(min_think),
        "triplet_step_trace": _format_triplet_fill_trace(step_records),
        "triplet_gate_failed": bool(any(not _gate_passes(bool(r["doc_ok"]), bool(r["think_ok"]), gate_on) for r in step_records)),
        "triplet_fail_reason": abstain_reason,
        "triplet_threshold": float(threshold),
        "triplet_gate_on": gate_on,
        "triplet_on_fail": on_fail,
        "triplet_fill_source": fill_source,
        "triplet_step_order": "think" if ordered_by_think else "question",
        "step_records": step_records,
    }


PAIR_ORDER = ["0/0", "1/0", "0/1", "1/1"]


def ok_pair_report(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "triplet_ok_pair" not in df.columns:
        return {}
    out: Dict[str, Any] = {
        "overall": {
            "n": int(len(df)),
            "qa_accuracy": float(df["is_correct"].mean()) if "is_correct" in df else float("nan"),
            "abstain_rate": float(df["abstained"].mean()) if "abstained" in df else float("nan"),
        },
        "sample_groups": [],
        "step_groups": [],
    }
    for pair in PAIR_ORDER:
        sub = df.loc[df["triplet_ok_pair"].fillna("").astype(str) == pair]
        n = int(len(sub))
        out["sample_groups"].append({
            "pair": pair,
            "n": n,
            "qa_accuracy": float(sub["is_correct"].mean()) if n and "is_correct" in sub else float("nan"),
            "f1_mean": float(sub["f1"].mean()) if n and "f1" in sub else float("nan"),
            "abstain_rate": float(sub["abstained"].mean()) if n and "abstained" in sub else float("nan"),
        })

    step_rows: List[Dict[str, Any]] = []
    if "triplet_step_ok_pairs" in df.columns:
        for _, row in df.iterrows():
            for pair in str(row.get("triplet_step_ok_pairs", "") or "").split("|"):
                if pair in PAIR_ORDER:
                    step_rows.append({
                        "pair": pair,
                        "is_correct": bool(row.get("is_correct", False)),
                        "f1": float(row.get("f1", 0.0) or 0.0),
                        "abstained": bool(row.get("abstained", False)),
                    })
    if step_rows:
        sdf = pd.DataFrame(step_rows)
        for pair in PAIR_ORDER:
            sub = sdf.loc[sdf["pair"] == pair]
            n = int(len(sub))
            out["step_groups"].append({
                "pair": pair,
                "n": n,
                "qa_accuracy": float(sub["is_correct"].mean()) if n else float("nan"),
                "f1_mean": float(sub["f1"].mean()) if n else float("nan"),
                "abstain_rate": float(sub["abstained"].mean()) if n else float("nan"),
            })
    return out


def summarize_online_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"n_samples": 0}
    answered = ~df.get("abstained", pd.Series([False] * len(df))).astype(bool)
    align_score_cols = ("align_QD", "align_QD_mean", "triplet_min_doc_score", "triplet_min_think_score")
    out = {
        "n_samples": int(len(df)),
        "qa_accuracy": float(df["is_correct"].mean()) if "is_correct" in df else float("nan"),
        "em_mean": float(df["em"].mean()) if "em" in df else float("nan"),
        "f1_mean": float(df["f1"].mean()) if "f1" in df else float("nan"),
        "abstain_rate": float(df["abstained"].mean()) if "abstained" in df else 0.0,
        "accuracy_when_answered": (
            float(df.loc[answered, "is_correct"].mean())
            if "is_correct" in df and answered.any()
            else float("nan")
        ),
        "ok_pair_report": ok_pair_report(df),
        "align_by_answer": align_by_answer_report(
            df,
            correct_cols=("searchr1_is_correct", "final_q_hint_is_correct"),
            score_cols=align_score_cols,
        ),
    }
    if "searchr1_is_correct" in df:
        out["searchr1_qa_accuracy"] = float(df["searchr1_is_correct"].mean())
        out["searchr1_em_mean"] = float(df["searchr1_em"].mean()) if "searchr1_em" in df else float("nan")
        out["searchr1_f1_mean"] = float(df["searchr1_f1"].mean()) if "searchr1_f1" in df else float("nan")
    if "final_q_hint_is_correct" in df:
        hint_mask = df.get("final_q_hint_answer", pd.Series([""] * len(df))).fillna("").astype(str).str.len() > 0
        if hint_mask.any():
            out["final_q_hint_qa_accuracy"] = float(df.loc[hint_mask, "final_q_hint_is_correct"].mean())
            out["final_q_hint_em_mean"] = float(df.loc[hint_mask, "final_q_hint_em"].mean()) if "final_q_hint_em" in df else float("nan")
            out["final_q_hint_f1_mean"] = float(df.loc[hint_mask, "final_q_hint_f1"].mean()) if "final_q_hint_f1" in df else float("nan")
            out["final_q_hint_n"] = int(hint_mask.sum())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["2wikimultihopqa"],
                   choices=list(DATASETS.keys()))
    p.add_argument("--input-filename", default="train_sampled.json")
    p.add_argument("--limit", type=int, default=20,
                   help="per-dataset limit. Use 0 for no limit.")
    p.add_argument("--dataset-limits", type=int, nargs="*", default=None,
                   help="Optional per-dataset limits aligned with --datasets. "
                        "Use 0 for no limit for that dataset.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output-dir", default="graphqa/outputs/online_verigraph")
    p.add_argument("--save-graphs", action="store_true", default=True)
    p.add_argument("--no-save-graphs", dest="save_graphs", action="store_false")

    p.add_argument("--retrieval-mode", choices=["searchr1", "bm25"], default="searchr1")
    p.add_argument("--trajectory-mode", choices=["searchr1_first", "q_guided", "observer"],
                   default="searchr1_first",
                   help="searchr1_first: run full SearchR1 trajectory first, then "
                        "align Q triples post hoc. q_guided: extract Q first and "
                        "condition each SearchR1 search on the selected unresolved "
                        "Q triple subgoal. observer: extract Q first, but give "
                        "SearchR1 only the original question and fill/verify Q "
                        "slots from observed think/doc triples after each search.")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--bm25-top-k", type=int, default=10)
    p.add_argument("--searchr1-model", default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    p.add_argument("--searchr1-top-k", type=int, default=3)
    p.add_argument("--searchr1-max-turns", type=int, default=4)
    p.add_argument("--searchr1-max-new-tokens", type=int, default=500)
    p.add_argument("--searchr1-temperature", type=float, default=1.0)
    p.add_argument("--searchr1-use-total-results", action="store_true", default=True)
    p.add_argument("--searchr1-last-results-only", dest="searchr1_use_total_results",
                   action="store_false")
    p.add_argument("--searchr1-verbose", action="store_true")
    p.add_argument("--answer-mode", choices=["searchr1", "final_q_hint", "both"],
                   default="both",
                   help="searchr1: score the first SearchR1 answer. "
                        "final_q_hint: after cosine filling, answer again with final_Q "
                        "as a SearchR1 hint and score that answer. "
                        "both: keep SearchR1 as the scored answer, but also save/score "
                        "the final_Q-hint answer in separate columns.")

    p.add_argument("--document-model", default="doupari/Llama-3.2-1B-Instruct-document")
    p.add_argument("--question-model", default="doupari/Llama-3.2-1B-Instruct-question-think-search")
    p.add_argument("--think-model", default="doupari/Llama-3.2-1B-Instruct-question-think-search")
    p.add_argument("--question-task", choices=list(SYSTEM_PROMPTS.keys()), default="question")
    p.add_argument("--think-task", choices=list(SYSTEM_PROMPTS.keys()), default="think+search")
    p.add_argument("--graph-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--graph-device-map", default="auto")
    p.add_argument("--graph-base-model", default="",
                   help="optional base model/path for PEFT adapter-only graph models. "
                        "If empty, adapter_config.json base_model_name_or_path is used.")
    p.add_argument("--graph-max-new-tokens", type=int, default=512)
    p.add_argument("--graph-temperature", type=float, default=0.0)
    p.add_argument("--doc-max-words", type=int, default=500)
    p.add_argument("--max-docs-per-sample", type=int, default=10)
    p.add_argument("--max-think-steps", type=int, default=8)
    p.add_argument("--doc-rescue-rounds", type=int, default=1,
                   help="extra doc-only retrieval rounds for unresolved final_Q slots. "
                        "The retrieved documents are graphed and slot filling is rerun; "
                        "think triples are still not used as fill values.")
    p.add_argument("--doc-rescue-max-queries", type=int, default=4)
    p.add_argument("--doc-rescue-max-docs", type=int, default=8)

    p.add_argument("--cosine-threshold", type=float, default=0.60,
                   help="field-level min cosine threshold for doc/think OK.")
    p.add_argument("--cosine-max-steps", type=int, default=16,
                   help="max query-triple checks/fill attempts per sample.")
    p.add_argument("--cosine-doc-top-k", type=int, default=5,
                   help="doc/think candidates narrowed by whole-triple cosine top-K "
                        "before field-level scoring.")
    p.add_argument("--cosine-on-fail", choices=["continue", "abstain"],
                   default="continue",
                   help="when the configured cosine gate fails.")
    p.add_argument("--cosine-gate-on", choices=["doc", "think", "both", "either"],
                   default="doc",
                   help="which OK bit controls abstain/continue gating.")
    p.add_argument("--cosine-fill-source", choices=["doc", "think", "doc_then_think"],
                   default="doc",
                   help="source used to fill UNKNOWN slots after cosine OK.")
    p.add_argument("--cosine-step-order", choices=["think", "question"],
                   default="question",
                   help="question: choose the least-empty question triple first, "
                        "then attach the best matching SearchR1 step evidence. "
                        "think: force slot filling/check order to follow SearchR1 "
                        "thinking-step order.")

    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default=None)
    p.add_argument("--selected-thresholds", type=float, nargs="+",
                   default=list(DEFAULT_ALIGN_THRESHOLDS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        args.limit = None
    if args.dataset_limits:
        args.dataset_limits = [
            None if int(v) <= 0 else int(v)
            for v in args.dataset_limits
        ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

    output_root = pathlib.Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pool = ExtractorPool(args)
    graph_model_ids = list(dict.fromkeys([
        args.question_model,
        args.document_model,
        args.think_model,
    ]))
    logger.info("[graph-model] preflight loading %d graph model(s)", len(graph_model_ids))
    for model_id in graph_model_ids:
        pool.get(model_id)

    encoder = get_default_encoder(model_name=args.encoder, device=args.device)
    shared_searchr1 = None
    if args.trajectory_mode in ("q_guided", "observer") and args.retrieval_mode != "searchr1":
        raise ValueError(f"--trajectory-mode {args.trajectory_mode} requires --retrieval-mode searchr1")
    if args.retrieval_mode == "searchr1":
        logger.info("[searchr1] loading %s", args.searchr1_model)
        shared_searchr1 = SearchR1Inference(
            model_id=args.searchr1_model,
            retriever_url=args.retriever_url,
            max_turns=args.searchr1_max_turns,
            max_new_tokens=args.searchr1_max_new_tokens,
            temperature=args.searchr1_temperature,
            topk=args.searchr1_top_k,
        )
    elif args.answer_mode in ("final_q_hint", "both"):
        raise ValueError("--answer-mode final_q_hint/both requires --retrieval-mode searchr1")

    combined_dfs: List[pd.DataFrame] = []
    for dataset_idx, dataset in enumerate(args.datasets):
        print("\n" + "#" * 90)
        print(f"# online dataset = {dataset}")
        print("#" * 90)
        dataset_limit = args.limit
        if args.dataset_limits and dataset_idx < len(args.dataset_limits):
            dataset_limit = args.dataset_limits[dataset_idx]
        raw_rows = load_raw_dataset(dataset, args.input_filename, dataset_limit, args.start)
        out_dir = output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        direct = make_direct_retriever(dataset, args) if args.retrieval_mode == "bm25" else None
        searchr1 = shared_searchr1

        rows: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []
        for raw in tqdm(raw_rows, desc=f"online-{dataset}", dynamic_ncols=True):
            try:
                doc_rescue_records: List[Dict[str, Any]] = []
                if args.trajectory_mode == "q_guided":
                    if searchr1 is None:
                        raise RuntimeError("Q-guided trajectory mode requires SearchR1")
                    sample, trace, check = build_online_sample_q_guided(
                        raw,
                        dataset,
                        args,
                        pool,
                        searchr1,
                        encoder,
                    )
                elif args.trajectory_mode == "observer":
                    if searchr1 is None:
                        raise RuntimeError("Observer trajectory mode requires SearchR1")
                    sample, trace, check = build_online_sample_observer(
                        raw,
                        dataset,
                        args,
                        pool,
                        searchr1,
                        encoder,
                    )
                else:
                    sample, trace = build_online_sample(raw, dataset, args, pool, direct, searchr1)
                    check = run_cosine_step_check(
                        sample,
                        encoder,
                        threshold=args.cosine_threshold,
                        max_steps=args.cosine_max_steps,
                        doc_top_k=args.cosine_doc_top_k,
                        on_fail=args.cosine_on_fail,
                        gate_on=args.cosine_gate_on,
                        fill_source=args.cosine_fill_source,
                        step_order=args.cosine_step_order,
                    )
                    for rescue_round in range(1, int(args.doc_rescue_rounds) + 1):
                        if not check.get("remaining_slots"):
                            break
                        new_rescue_records = rescue_unfilled_slots_with_docs(
                            sample,
                            trace,
                            check,
                            args=args,
                            pool=pool,
                            searchr1=searchr1,
                            round_number=rescue_round,
                        )
                        doc_rescue_records.extend(new_rescue_records)
                        if not new_rescue_records or not any(
                            int(r.get("n_new_docs", 0) or 0) > 0 for r in new_rescue_records
                        ):
                            break
                        check = run_cosine_step_check(
                            sample,
                            encoder,
                            threshold=args.cosine_threshold,
                            max_steps=args.cosine_max_steps,
                            doc_top_k=args.cosine_doc_top_k,
                            on_fail=args.cosine_on_fail,
                            gate_on=args.cosine_gate_on,
                            fill_source=args.cosine_fill_source,
                            step_order=args.cosine_step_order,
                        )
                searchr1_answer = str(sample.predicted_answer or "")
                searchr1_em, searchr1_f1 = score_answer(
                    searchr1_answer,
                    sample.answer,
                    sample.answer_aliases,
                )
                final_q_hint_info: Dict[str, Any] = {}
                final_q_hint_answer = ""
                final_q_hint_em = float("nan")
                final_q_hint_f1 = float("nan")
                if bool(check["abstained"]):
                    final_q_hint_info = {
                        "skipped": True,
                        "reason": str(check["abstain_reason"]),
                    }
                elif args.answer_mode in ("final_q_hint", "both"):
                    if searchr1 is None:
                        raise RuntimeError("SearchR1 is required for final_Q hint answering")
                    final_q_hint_info = answer_with_final_q_hint(
                        sample.question,
                        check["final_Q"],
                        searchr1,
                        args,
                    )
                    final_q_hint_answer = str(final_q_hint_info.get("predicted_answer", "") or "")
                    final_q_hint_em, final_q_hint_f1 = score_answer(
                        final_q_hint_answer,
                        sample.answer,
                        sample.answer_aliases,
                    )

                if bool(check["abstained"]):
                    qa_answer = ""
                    em, f1 = score_answer("", sample.answer, sample.answer_aliases)
                elif args.answer_mode == "final_q_hint" or (
                    args.trajectory_mode in ("q_guided", "observer") and args.answer_mode == "both"
                ):
                    qa_answer = final_q_hint_answer
                    em, f1 = final_q_hint_em, final_q_hint_f1
                else:
                    qa_answer = searchr1_answer
                    em, f1 = searchr1_em, searchr1_f1

                align = compute_sample_alignment(sample, encoder)
                row = {
                    "dataset": dataset,
                    "uid": sample.uid,
                    "question": sample.question,
                    "answer": sample.answer,
                    "answer_mode": args.answer_mode,
                    "predicted_answer": qa_answer,
                    "em": float(em),
                    "f1": float(f1),
                    "is_correct": bool(em >= 1.0),
                    "searchr1_answer": searchr1_answer,
                    "searchr1_em": float(searchr1_em),
                    "searchr1_f1": float(searchr1_f1),
                    "searchr1_is_correct": bool(searchr1_em >= 1.0),
                    "final_q_hint_answer": final_q_hint_answer,
                    "final_q_hint_em": float(final_q_hint_em),
                    "final_q_hint_f1": float(final_q_hint_f1),
                    "final_q_hint_is_correct": (
                        bool(final_q_hint_em >= 1.0)
                        if final_q_hint_answer
                        else False
                    ),
                    "final_q_hint_num_turns": int(final_q_hint_info.get("num_turns", 0) or 0),
                    "doc_rescue_rounds_used": len(doc_rescue_records),
                    "doc_rescue_new_docs": sum(
                        int(r.get("n_new_docs", 0) or 0) for r in doc_rescue_records
                    ),
                    "is_yesno": bool(_is_yesno_question(sample.question)),
                    "abstained": bool(check["abstained"]),
                    "abstain_reason": str(check["abstain_reason"]),
                    "n_hops": int(sample.num_hops),
                    "n_Q": len(sample.Q),
                    "n_T": len(sample.T),
                    "n_Sr": len(sample.Sr),
                    "n_D": len(sample.D),
                    "n_steps": len(sample.T_steps),
                    "n_slot_steps": int(check["n_slot_steps"]),
                    "filled_slots": "|".join(
                        f"{k}={v}" for k, v in dict(check["filled_slots"]).items()
                    ),
                    "remaining_slots": "|".join(check["remaining_slots"]),
                    "triplet_doc_ok": check["triplet_doc_ok"],
                    "triplet_think_ok": check["triplet_think_ok"],
                    "triplet_ok_pair": check["triplet_ok_pair"],
                    "triplet_step_ok_pairs": check["triplet_step_ok_pairs"],
                    "triplet_min_doc_score": check["triplet_min_doc_score"],
                    "triplet_min_think_score": check["triplet_min_think_score"],
                    "triplet_step_trace": check["triplet_step_trace"],
                    "triplet_gate_failed": check["triplet_gate_failed"],
                    "triplet_fail_reason": check["triplet_fail_reason"],
                    "triplet_problem_codes": "|".join(
                        str(r.get("problem_codes", "") or "")
                        for r in check.get("step_records", [])
                        if str(r.get("problem_codes", "") or "")
                    ),
                    "triplet_threshold": check["triplet_threshold"],
                    "triplet_gate_on": check["triplet_gate_on"],
                    "triplet_on_fail": check["triplet_on_fail"],
                    "triplet_fill_source": check["triplet_fill_source"],
                    "triplet_step_order": check["triplet_step_order"],
                    "observer_stop_reason": str(check.get("observer_stop_reason", "")),
                }
                row.update(align.to_row())
                row.update({
                    "online_trajectory_mode": args.trajectory_mode,
                    "online_retrieval_mode": args.retrieval_mode,
                    "online_question_model": args.question_model,
                    "online_think_model": args.think_model,
                    "online_document_model": args.document_model,
                    "online_n_docs": len(trace["document_graph"]["per_document"]),
                    "online_n_q_triples": len(sample.Q),
                    "online_n_d_triples": len(sample.D),
                    "online_n_t_triples": len(sample.T),
                    "online_n_sr_triples": len(sample.Sr),
                })
                rows.append(row)
                trace["cosine_check"] = {
                    k: v for k, v in check.items()
                    if k not in {"final_Q"}
                }
                trace["cosine_check"]["final_Q"] = triples_to_raw(check["final_Q"])
                trace["doc_rescue"] = doc_rescue_records
                trace["answering"] = {
                    "answer_mode": args.answer_mode,
                    "selected_answer": qa_answer,
                    "selected_em": float(em),
                    "selected_f1": float(f1),
                    "searchr1_answer": searchr1_answer,
                    "searchr1_em": float(searchr1_em),
                    "searchr1_f1": float(searchr1_f1),
                    "final_q_hint": {
                        "requested": bool(args.answer_mode in ("final_q_hint", "both")),
                        "enabled": bool(args.answer_mode in ("final_q_hint", "both") and not check["abstained"]),
                        "graph_hint": triples_to_raw(check["final_Q"]),
                        "answer": final_q_hint_answer,
                        "em": float(final_q_hint_em),
                        "f1": float(final_q_hint_f1),
                        "info": final_q_hint_info,
                    },
                }
                traces.append(trace)
            except Exception as exc:
                logger.exception("[online] sample failed: %s", exc)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        df = pd.DataFrame(rows)
        csv_path = out_dir / f"online_eval_{dataset}.csv"
        df.to_csv(csv_path, index=False)
        summary_dict = summarize_online_df(df)
        summary_dict["online"] = {
            "retrieval_mode": args.retrieval_mode,
            "trajectory_mode": args.trajectory_mode,
            "document_model": args.document_model,
            "question_model": args.question_model,
            "think_model": args.think_model,
            "question_task": args.question_task,
            "think_task": args.think_task,
            "answer_mode": args.answer_mode,
            "cosine_threshold": args.cosine_threshold,
            "cosine_on_fail": args.cosine_on_fail,
            "cosine_gate_on": args.cosine_gate_on,
            "cosine_fill_source": args.cosine_fill_source,
            "cosine_step_order": args.cosine_step_order,
            "doc_rescue_rounds": args.doc_rescue_rounds,
            "doc_rescue_max_queries": args.doc_rescue_max_queries,
            "doc_rescue_max_docs": args.doc_rescue_max_docs,
            "selected_accuracy": selected_accuracy_report(
                df,
                score_cols=("align_QD", "align_QD_mean", "triplet_min_doc_score", "triplet_min_think_score"),
                thresholds=args.selected_thresholds,
            ),
            "selected_accuracy_by_answer": {
                "searchr1": selected_accuracy_report(
                    df,
                    score_cols=("align_QD", "align_QD_mean", "triplet_min_doc_score", "triplet_min_think_score"),
                    thresholds=args.selected_thresholds,
                    correct_col="searchr1_is_correct",
                ),
                "final_q_hint": selected_accuracy_report(
                    df,
                    score_cols=("align_QD", "align_QD_mean", "triplet_min_doc_score", "triplet_min_think_score"),
                    thresholds=args.selected_thresholds,
                    correct_col="final_q_hint_is_correct",
                ),
            },
        }
        json_path = out_dir / f"online_eval_{dataset}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False, default=float)
        if args.save_graphs:
            graph_path = out_dir / f"online_eval_{dataset}_graphs.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=2, ensure_ascii=False)
            logger.info("[online] saved graphs: %s", graph_path)
        logger.info("[online] saved %s and %s", csv_path, json_path)
        print(
            "[online-summary] n={n} qa_acc={acc:.3f} em={emv:.3f} "
            "f1={f1v:.3f} abstain={abst:.3f}".format(
                n=summary_dict.get("n_samples", 0),
                acc=summary_dict.get("qa_accuracy", 0.0),
                emv=summary_dict.get("em_mean", 0.0),
                f1v=summary_dict.get("f1_mean", 0.0),
                abst=summary_dict.get("abstain_rate", 0.0),
            )
        )
        combined_dfs.append(df.assign(dataset=dataset))

    if len(combined_dfs) > 1:
        combined = pd.concat(combined_dfs, ignore_index=True)
        combined_path = output_root / "online_eval_all.csv"
        combined.to_csv(combined_path, index=False)
        combined_summary = summarize_online_df(combined)
        combined_summary["online"] = {
            "trajectory_mode": args.trajectory_mode,
            "answer_mode": args.answer_mode,
            "cosine_fill_source": args.cosine_fill_source,
            "doc_rescue_rounds": args.doc_rescue_rounds,
            "doc_rescue_max_queries": args.doc_rescue_max_queries,
            "doc_rescue_max_docs": args.doc_rescue_max_docs,
            "selected_accuracy": selected_accuracy_report(
                combined,
                score_cols=("align_QD", "align_QD_mean", "triplet_min_doc_score", "triplet_min_think_score"),
                thresholds=args.selected_thresholds,
            ),
            "selected_accuracy_by_answer": {
                "searchr1": selected_accuracy_report(
                    combined,
                    score_cols=("align_QD", "align_QD_mean", "triplet_min_doc_score", "triplet_min_think_score"),
                    thresholds=args.selected_thresholds,
                    correct_col="searchr1_is_correct",
                ),
                "final_q_hint": selected_accuracy_report(
                    combined,
                    score_cols=("align_QD", "align_QD_mean", "triplet_min_doc_score", "triplet_min_think_score"),
                    thresholds=args.selected_thresholds,
                    correct_col="final_q_hint_is_correct",
                ),
            },
        }
        with open(output_root / "online_eval_all_summary.json", "w", encoding="utf-8") as f:
            json.dump(combined_summary, f, indent=2, ensure_ascii=False, default=float)
        logger.info("[online] saved combined CSV: %s", combined_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
