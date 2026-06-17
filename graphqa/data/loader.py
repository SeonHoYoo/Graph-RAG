"""4가지 그래프(Q, T, Sr, D)를 uid로 매칭해 GraphSample 리스트로 로딩."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from graphqa.data.schema import GraphSample, GraphStep, StepEvidence, Triple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 데이터셋별 파일 경로 매핑
# ---------------------------------------------------------------------------
GRAPH_ROOT = Path("/data3/seonhoyoo/graphcheck-qa")


@dataclass(frozen=True)
class DatasetPaths:
    name: str                # logical 이름 (e.g. '2wikimultihopqa')
    short: str               # searchr1 파일에서 쓰는 짧은 이름 (e.g. '2wiki')
    triplets_file: Path      # Q + D 가 들어있는 triplets_train_sampled.json
    reasoning_file: Path     # T (reasoning_graph_*.json)
    searchr1_file: Path      # Sr 추출용 원본 searchr1 출력


DATASETS: Dict[str, DatasetPaths] = {
    "2wikimultihopqa": DatasetPaths(
        name="2wikimultihopqa",
        short="2wiki",
        triplets_file=GRAPH_ROOT / "results/2wikimultihopqa/triplets/Qwen2.5-7B-Instruct/triplets_train_sampled.json",
        reasoning_file=GRAPH_ROOT / "graph_data/searchr1/0407(open-book)/reasoning_graph/reasoning_graph_2wiki_vanilla_searchr1_128615_500.json",
        searchr1_file=GRAPH_ROOT / "graph_data/searchr1/0407(open-book)/2wiki_vanilla_searchr1_128615_500.json",
    ),
    "hotpotqa": DatasetPaths(
        name="hotpotqa",
        short="hotpotqa",
        triplets_file=GRAPH_ROOT / "results/hotpotqa/triplets/Qwen2.5-7B-Instruct/triplets_train_sampled.json",
        reasoning_file=GRAPH_ROOT / "graph_data/searchr1/0407(open-book)/reasoning_graph/reasoning_graph_hotpotqa_vanilla_searchr1_128616_500.json",
        searchr1_file=GRAPH_ROOT / "graph_data/searchr1/0407(open-book)/hotpotqa_vanilla_searchr1_128616_500.json",
    ),
    "musique": DatasetPaths(
        name="musique",
        short="musique",
        triplets_file=GRAPH_ROOT / "results/musique/triplets/Qwen2.5-7B-Instruct/triplets_train_sampled.json",
        reasoning_file=GRAPH_ROOT / "graph_data/searchr1/0407(open-book)/reasoning_graph/reasoning_graph_musique_vanilla_searchr1_128617_1000.json",
        searchr1_file=GRAPH_ROOT / "graph_data/searchr1/0407(open-book)/musique_vanilla_searchr1_128617_1000.json",
    ),
}


# ---------------------------------------------------------------------------
# Search query → triples 변환 (옵션 A: LLM 호출 없이)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "with", "to", "from",
    "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "who", "whom", "whose", "which", "where",
    "when", "why", "how", "that", "this", "these", "those",
}


def _clean_query(q: str) -> str:
    q = q.strip().strip("?.").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _query_to_triples(query: str, turn_idx: int) -> List[Triple]:
    """검색 쿼리 텍스트 → triple-like 표현 (LLM 없이).

    전략:
    1) "A of B" / "A's B" / "B of A" 패턴이 있으면 (B, of, A) triple 1개 + 슬롯 triple
    2) "X nationality" "X birthplace" 같은 속성 질문이면 (X, has_attribute, ?Y)
    3) 기본: (?XK, related_to, <키워드 모음>) — placeholder를 답 슬롯으로
    """
    q = _clean_query(query)
    if not q:
        return []

    triples: List[Triple] = []
    slot = f"?S{turn_idx}"

    # "A of B" 패턴: "directors of Slums of Berlin"
    m = re.match(r"^([\w\s\-\.\,'\"]+?)\s+of\s+(.+)$", q, flags=re.IGNORECASE)
    if m:
        attr, target = m.group(1).strip(), m.group(2).strip()
        # (target, has_attribute, ?S)  +  (?S, is_a, attr)
        triples.append(Triple(head=target, relation="has_attribute", tail=slot,
                              raw=f"{target} [SEP] has_attribute [SEP] {slot}"))
        triples.append(Triple(head=slot, relation="is", tail=attr,
                              raw=f"{slot} [SEP] is [SEP] {attr}"))
        return triples

    # "X 's Y" / "X's Y": "Vikas Bahl's nationality"
    m = re.match(r"^(.+?)\s*['\u2019]s\s+(.+)$", q)
    if m:
        owner, attr = m.group(1).strip(), m.group(2).strip()
        triples.append(Triple(head=owner, relation="has_attribute", tail=slot,
                              raw=f"{owner} [SEP] has_attribute [SEP] {slot}"))
        triples.append(Triple(head=slot, relation="is", tail=attr,
                              raw=f"{slot} [SEP] is [SEP] {attr}"))
        return triples

    # "X nationality" / "X birthplace" / "X director" 형 (마지막 단어가 속성)
    tokens = q.split()
    if len(tokens) >= 2:
        head = " ".join(tokens[:-1]).strip()
        attr = tokens[-1].strip()
        if attr.lower() not in _STOPWORDS and head:
            triples.append(Triple(head=head, relation="has_attribute", tail=slot,
                                  raw=f"{head} [SEP] has_attribute [SEP] {slot}"))
            triples.append(Triple(head=slot, relation="is", tail=attr,
                                  raw=f"{slot} [SEP] is [SEP] {attr}"))
            return triples

    # fallback
    triples.append(Triple(head=slot, relation="related_to", tail=q,
                          raw=f"{slot} [SEP] related_to [SEP] {q}"))
    return triples


# ---------------------------------------------------------------------------
# 파일별 파서
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strs_to_triples(triple_strs: List[str]) -> List[Triple]:
    out: List[Triple] = []
    for s in triple_strs or []:
        t = Triple.from_str(s)
        if t is not None:
            out.append(t)
    return out


def _triple_key(t: Triple) -> Tuple[str, str, str, str]:
    return (
        " ".join((t.head or "").split()),
        " ".join((t.relation or "").split()),
        " ".join((t.tail or "").split()),
        " ".join((t.context or "").split()),
    )


def _dedupe_triples(triples: Iterable[Triple]) -> List[Triple]:
    out: List[Triple] = []
    seen = set()
    for t in triples:
        key = _triple_key(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _load_triplets_file(path: Path) -> Dict[str, dict]:
    """triplets_train_sampled.json → uid 인덱스. (Q + D)"""
    by_uid: Dict[str, dict] = {}
    for entry in _load_json(path):
        uid = entry.get("uid")
        if not uid:
            continue
        by_uid[uid] = entry
    return by_uid


def _load_reasoning_file(path: Path) -> Dict[str, dict]:
    by_uid: Dict[str, dict] = {}
    for entry in _load_json(path):
        uid = entry.get("uid")
        if not uid:
            continue
        by_uid[uid] = entry
    return by_uid


def _load_searchr1_file(path: Path) -> Dict[str, dict]:
    by_uid: Dict[str, dict] = {}
    for entry in _load_json(path):
        uid = entry.get("uid")
        if not uid:
            continue
        by_uid[uid] = entry
    return by_uid


def _resolve_triplets_file(
    dataset_name: str,
    default_file: Path,
    *,
    triplets_file: Optional[str | Path] = None,
    triplets_filename: Optional[str] = None,
    triplets_model_dir: str = "Qwen2.5-7B-Instruct",
) -> Path:
    """Resolve Q/D triplets path.

    기본값은 기존 gold+open 파일을 그대로 쓰되, online/open-book-only 실험처럼
    같은 dataset 아래 다른 triplets 파일을 읽고 싶을 때 CLI에서 override한다.
    """
    if triplets_file is not None:
        return Path(triplets_file)
    if triplets_filename:
        return (
            GRAPH_ROOT
            / "results"
            / dataset_name
            / "triplets"
            / triplets_model_dir
            / triplets_filename
        )
    return default_file


def _combined_file_for(dataset_name: str, combined_dir: Path) -> Path:
    stem = DATASETS[dataset_name].short
    return combined_dir / f"{stem}_combined_0514.json"


def _load_combined_file(path: Path) -> Dict[str, dict]:
    by_uid: Dict[str, dict] = {}
    if not path.exists():
        logger.warning("[loader] combined file not found: %s", path)
        return by_uid
    for entry in _load_json(path):
        uid = entry.get("uid")
        if not uid:
            continue
        by_uid[uid] = entry
    return by_uid


def _doc_graph_to_triples(graph_obj: object) -> List[Triple]:
    if isinstance(graph_obj, list):
        return _strs_to_triples([str(x) for x in graph_obj])
    if isinstance(graph_obj, dict):
        vals = graph_obj.get("triples", [])
        if isinstance(vals, list):
            return _strs_to_triples([str(x) for x in vals])
    return []


def _combined_step_evidence(entry: Optional[dict]) -> List[StepEvidence]:
    if not entry:
        return []
    retrieval_info = entry.get("retrieval_info", {}) or {}
    per_step = retrieval_info.get("per_step", []) or []
    steps: List[StepEvidence] = []
    for pos, step in enumerate(per_step):
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("step", pos) or pos)
        think = step.get("think", {}) or {}
        think_text = think.get("text", "") if isinstance(think, dict) else ""
        think_triples = (
            _strs_to_triples(think.get("triples", []) or [])
            if isinstance(think, dict) else []
        )
        doc_triples: List[Triple] = []
        doc_texts: List[str] = []
        for doc in step.get("documents", []) or []:
            if not isinstance(doc, dict):
                continue
            txt = str(doc.get("text", "") or "")
            if txt:
                doc_texts.append(txt)
            doc_triples.extend(_doc_graph_to_triples(doc.get("graph")))
        steps.append(StepEvidence(
            step_index=step_index,
            query=str(step.get("query", "") or ""),
            think_text=str(think_text or ""),
            think_triples=_dedupe_triples(think_triples),
            doc_triples=_dedupe_triples(doc_triples),
            doc_texts=doc_texts,
        ))
    return steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset(
    dataset_name: str,
    limit: Optional[int] = None,
    require_all_graphs: bool = True,
    combined_dir: Optional[str | Path] = None,
    triplets_file: Optional[str | Path] = None,
    triplets_filename: Optional[str] = None,
    triplets_model_dir: str = "Qwen2.5-7B-Instruct",
) -> List[GraphSample]:
    """uid로 4-그래프를 매칭해 GraphSample 리스트 반환."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS)}")
    paths = DATASETS[dataset_name]
    resolved_triplets_file = _resolve_triplets_file(
        dataset_name,
        paths.triplets_file,
        triplets_file=triplets_file,
        triplets_filename=triplets_filename,
        triplets_model_dir=triplets_model_dir,
    )
    logger.info(f"[loader] dataset={dataset_name}")
    logger.info(f"[loader] triplets={resolved_triplets_file}")

    triplets_idx = _load_triplets_file(resolved_triplets_file)
    reasoning_idx = _load_reasoning_file(paths.reasoning_file)
    searchr1_idx = _load_searchr1_file(paths.searchr1_file)
    combined_idx: Dict[str, dict] = {}
    if combined_dir is not None:
        combined_path = _combined_file_for(dataset_name, Path(combined_dir))
        combined_idx = _load_combined_file(combined_path)
    logger.info(
        f"[loader]  |triplets|={len(triplets_idx)}, |reasoning|={len(reasoning_idx)}, "
        f"|searchr1|={len(searchr1_idx)}, |combined|={len(combined_idx)}"
    )

    samples: List[GraphSample] = []
    skipped = 0

    # reasoning 파일 기준으로 순회 (T 기반)
    for uid, r_entry in reasoning_idx.items():
        triplets_entry = triplets_idx.get(uid)
        searchr1_entry = searchr1_idx.get(uid)
        if not triplets_entry or not searchr1_entry:
            skipped += 1
            continue

        # ---- Q ----
        q_graph = triplets_entry.get("question_graph", {}) or {}
        Q = _strs_to_triples(q_graph.get("triples", []))
        Q_def = _strs_to_triples(q_graph.get("definition_triples", []))

        # ---- D ----
        d_graph = triplets_entry.get("doc_graph", {}) or {}
        D = _strs_to_triples(d_graph.get("triples", []))

        # ---- T (reasoning_graph) ----
        rg = r_entry.get("reasoning_graph", {}) or {}
        T = _strs_to_triples(rg.get("triples", []))
        T_steps: List[GraphStep] = []
        for step in rg.get("per_step", []) or []:
            T_steps.append(GraphStep(
                step_index=step.get("step_index", len(T_steps)),
                step_text=step.get("step_text", ""),
                triples=_strs_to_triples(step.get("triples", [])),
            ))

        # ---- combined/0514 step evidence (optional) ----
        step_evidence = _combined_step_evidence(combined_idx.get(uid))
        if step_evidence:
            # For combined experiments, keep all scoring/evidence views on the
            # same source: union of observed SearchR1 step-level evidence.
            D = _dedupe_triples(t for st in step_evidence for t in st.doc_triples)
            T = _dedupe_triples(t for st in step_evidence for t in st.think_triples)
            T_steps = [
                GraphStep(
                    step_index=st.step_index,
                    step_text=st.think_text,
                    triples=list(st.think_triples),
                )
                for st in step_evidence
            ]

        # ---- Sr (search query → triple) ----
        retrieval_info = searchr1_entry.get("retrieval_info", {}) or {}
        retrieval_turns = retrieval_info.get("retrieval_turns", []) or []
        search_queries: List[str] = []
        Sr: List[Triple] = []
        for turn in retrieval_turns:
            q = turn.get("query", "")
            if not q:
                continue
            search_queries.append(q)
            Sr.extend(_query_to_triples(q, turn.get("turn", len(search_queries))))

        sample = GraphSample(
            uid=uid,
            question=triplets_entry.get("question", r_entry.get("question", "")),
            answer=str(triplets_entry.get("answer", r_entry.get("answer", ""))),
            answer_aliases=list(triplets_entry.get("answer_aliases", []) or []),
            num_hops=int(triplets_entry.get("num_hops", r_entry.get("num_hops", 0)) or 0),
            dataset=dataset_name,
            Q_def=Q_def,
            Q=Q,
            T=T,
            T_steps=T_steps,
            step_evidence=step_evidence,
            Sr=Sr,
            D=D,
            search_queries=search_queries,
            predicted_answer=r_entry.get("predicted_answer"),
            gold_id_list=list(triplets_entry.get("gold_id_list", []) or []),
        )

        if require_all_graphs and not sample.has_all_graphs:
            skipped += 1
            continue

        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break

    logger.info(f"[loader] -> samples={len(samples)} (skipped={skipped})")
    return samples


def iter_dataset(dataset_name: str, **kwargs) -> Iterator[GraphSample]:
    yield from load_dataset(dataset_name, **kwargs)
