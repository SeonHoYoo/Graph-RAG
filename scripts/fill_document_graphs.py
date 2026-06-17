import argparse
import json
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


LOGGER = logging.getLogger(__name__)


DATASET_NAME_BY_PREFIX = {
    "2wiki": "2wikimultihopqa",
    "hotpotqa": "hotpotqa",
    "musique": "musique",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill retrieval_info.per_step[*].documents[*].graph using the same "
            "document triplet extractor used for base doc_graph.per_document."
        )
    )
    parser.add_argument(
        "--input-file",
        nargs="+",
        required=True,
        help="Combined JSON file(s) to update.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, files are updated in place.",
    )
    parser.add_argument(
        "--graphcheck-root",
        default="/home/hyeseojeon/data/graph/__graphcheck-qa-2",
        help="Repo containing model_library.construct_model.",
    )
    parser.add_argument(
        "--construct-model-name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name passed to ConstructModel.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Debug limit for number of unique missing documents to extract per dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report coverage without writing files or loading the extraction model.",
    )
    return parser.parse_args()


def normalize_document(document: str) -> str:
    return " ".join((document or "").split())


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return data


def infer_dataset_name(path: Path) -> str:
    name = path.name
    for prefix, dataset_name in DATASET_NAME_BY_PREFIX.items():
        if name.startswith(prefix):
            return dataset_name
    raise ValueError(
        f"Cannot infer dataset from file name: {path}. "
        f"Expected prefix one of: {', '.join(DATASET_NAME_BY_PREFIX)}"
    )


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def iter_document_nodes(data: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for sample in data:
        retrieval_info = sample.get("retrieval_info") or {}
        for step in retrieval_info.get("per_step") or []:
            for document in step.get("documents") or []:
                if isinstance(document, dict):
                    yield document


def collect_missing_documents(
    combined_data: List[Dict[str, Any]],
    graph_by_document: Dict[str, List[str]],
) -> List[str]:
    missing: Dict[str, str] = {}
    for document_node in iter_document_nodes(combined_data):
        text = document_node.get("text") or ""
        key = normalize_document(text)
        if key and key not in graph_by_document:
            missing.setdefault(key, text)
    return list(missing.values())


def load_construct_model(
    graphcheck_root: Path,
    construct_model_name: str,
    dataset_name: str,
    api_key: Optional[str],
):
    sys.path.insert(0, str(graphcheck_root))
    if not construct_model_name.lower().startswith("claude"):
        try:
            __import__("anthropic")
        except ImportError:
            sys.modules["anthropic"] = types.SimpleNamespace(
                Anthropic=lambda *args, **kwargs: None
            )
    from model_library.construct_model import ConstructModel

    return ConstructModel(
        construct_model_name=construct_model_name,
        dataset_name=dataset_name,
        api_key=api_key,
    )


def extract_missing_graphs(
    documents: List[str],
    graph_by_document: Dict[str, List[str]],
    construct_model: Any,
    max_docs: Optional[int],
) -> Tuple[int, int]:
    extracted = 0
    failed = 0
    limited_documents = documents[:max_docs] if max_docs is not None else documents
    iterator = limited_documents
    if tqdm is not None:
        iterator = tqdm(limited_documents, desc="extract document graphs")
    for document in iterator:
        key = normalize_document(document)
        if key in graph_by_document:
            continue
        try:
            _, triples = construct_model.extract_triplets_from_document(document)
            graph_by_document[key] = list(triples)
            extracted += 1
        except Exception as exc:
            LOGGER.warning("Document graph extraction failed: %s", exc)
            graph_by_document[key] = []
            failed += 1
    return extracted, failed


def fill_graphs(
    combined_data: List[Dict[str, Any]],
    graph_by_document: Dict[str, List[str]],
) -> Tuple[int, int]:
    filled = 0
    missing = 0
    for document_node in iter_document_nodes(combined_data):
        text = document_node.get("text") or ""
        key = normalize_document(text)
        if key in graph_by_document:
            document_node["graph"] = list(graph_by_document[key])
            filled += 1
        else:
            document_node["graph"] = []
            missing += 1
    return filled, missing


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    for input_file in args.input_file:
        input_path = Path(input_file)
        dataset_name = infer_dataset_name(input_path)
        output_path = (output_dir / input_path.name) if output_dir else input_path

        combined_data = load_json(input_path)
        graph_by_document: Dict[str, List[str]] = {}
        missing_documents = collect_missing_documents(combined_data, graph_by_document)

        LOGGER.info(
            "%s: samples=%d unique_docs_to_extract=%d",
            input_path.name,
            len(combined_data),
            len(missing_documents),
        )

        if args.dry_run:
            filled, missing = fill_graphs(combined_data, graph_by_document)
            LOGGER.info("%s dry-run: fillable=%d missing=%d", input_path.name, filled, missing)
            continue

        if args.max_docs == 0:
            LOGGER.info("%s: max-docs=0, skipping model load and extraction", input_path.name)
            continue
        elif missing_documents:
            construct_model = load_construct_model(
                Path(args.graphcheck_root),
                args.construct_model_name,
                dataset_name,
                args.api_key,
            )
            extracted, failed = extract_missing_graphs(
                missing_documents,
                graph_by_document,
                construct_model,
                args.max_docs,
            )
            LOGGER.info("%s: extracted=%d failed=%d", input_path.name, extracted, failed)

        filled, missing = fill_graphs(combined_data, graph_by_document)
        atomic_write_json(output_path, combined_data)
        LOGGER.info("%s: wrote=%s filled=%d missing=%d", input_path.name, output_path, filled, missing)


if __name__ == "__main__":
    main()
