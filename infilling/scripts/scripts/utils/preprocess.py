from typing import Any


def select_graph(sample: dict[str, Any], use_gold: int) -> tuple[str, dict[str, Any]]:
    graph_key = "gold_graph" if use_gold == 1 else "doc_graph"
    graph = sample.get(graph_key, {})
    if not isinstance(graph, dict):
        graph = {}
    return graph_key, graph


def extract_documents(graph: dict[str, Any]) -> list[str]:
    per_document = graph.get("per_document", [])
    if not isinstance(per_document, list):
        return []

    documents: list[str] = []
    for doc in per_document:
        if not isinstance(doc, dict):
            continue
        text = doc.get("document", "")
        if isinstance(text, str) and text.strip():
            documents.append(text)
    return documents


def extract_triples(graph: dict[str, Any]) -> list[str]:
    triples = graph.get("triples", [])
    if not isinstance(triples, list):
        return []
    return [str(t) for t in triples if str(t).strip()]
