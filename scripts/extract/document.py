#!/usr/bin/env python
"""Extract document-level knowledge graphs from filtered dataset splits.

For each record in filtered/train.json, formats each document as:
  (Title: {title}) {text}
then calls ConstructModel.extract_triplets_from_document() and saves results
to a JSONL file in real-time (one line per document, flushed immediately).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import types
try:
    import anthropic  # noqa: F401
except ImportError:
    sys.modules["anthropic"] = types.SimpleNamespace(
        Anthropic=lambda *args, **kwargs: None
    )

from model_library.construct_model import ConstructModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--input_file", default=None, help="Override default filtered/train.json path.")
    parser.add_argument("--output_dir", default="results/graph/0515")
    parser.add_argument("--construct_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_docs_per_record", type=int, default=None)
    return parser.parse_args()


# ── document iterators ────────────────────────────────────────────────────────

def iter_docs_hotpotqa(record: Dict[str, Any]) -> Generator[Tuple[str, str], None, None]:
    """Yields (title, text) from context.title + context.sentences."""
    titles    = record["context"]["title"]
    sentences = record["context"]["sentences"]
    for title, text in zip(titles, sentences):
        yield title, text


def iter_docs_2wiki(record: Dict[str, Any]) -> Generator[Tuple[str, str], None, None]:
    """Same structure as hotpotqa."""
    yield from iter_docs_hotpotqa(record)


def iter_docs_musique(record: Dict[str, Any]) -> Generator[Tuple[str, str], None, None]:
    """Yields (title, paragraph_text) from paragraphs list."""
    for para in record.get("paragraphs", []):
        yield para["title"], para["paragraph_text"]


ITER_FN = {
    "hotpotqa":       iter_docs_hotpotqa,
    "2wikimultihopqa": iter_docs_2wiki,
    "musique":        iter_docs_musique,
}


def format_document(title: str, text: str) -> str:
    return f"(Title: {title}) {text}"


# ── already-done set ──────────────────────────────────────────────────────────

def load_done_keys(output_path: Path) -> set:
    """Read existing JSONL output and return set of (record_id, title) already extracted."""
    done = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj["id"], obj["title"]))
            except Exception:
                pass
    return done


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file) if args.input_file else (
        REPO_ROOT / "datasets" / args.dataset / "filtered" / "train.json"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_documents.jsonl"

    logger.info("Dataset   : %s", args.dataset)
    logger.info("Input     : %s", input_path)
    logger.info("Output    : %s", output_path)

    with input_path.open() as f:
        records = json.load(f)
    if args.max_records:
        records = records[:args.max_records]
    logger.info("Records   : %d", len(records))

    done_keys = load_done_keys(output_path)
    logger.info("Already done: %d document(s)", len(done_keys))

    construct_model = ConstructModel(
        construct_model_name=args.construct_model_name,
        dataset_name=args.dataset,
    )

    iter_docs = ITER_FN[args.dataset]

    with output_path.open("a") as out_f:
        for record in tqdm(records, desc=args.dataset):
            record_id = record.get("id") or record.get("_id", "")
            question  = record.get("question", "")

            docs = list(iter_docs(record))
            if args.max_docs_per_record:
                docs = docs[:args.max_docs_per_record]

            for title, text in docs:
                if (record_id, title) in done_keys:
                    continue

                document = format_document(title, text)
                try:
                    _, triples = construct_model.extract_triplets_from_document(document)
                    triples = list(triples)
                except Exception as exc:
                    logger.warning("Extraction failed [%s | %s]: %s", record_id, title, exc)
                    triples = []

                entry = {
                    "id":       record_id,
                    "question": question,
                    "title":    title,
                    "document": document,
                    "graph":    triples,
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                done_keys.add((record_id, title))

    logger.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
