#!/usr/bin/env python
"""Extract question-level knowledge graphs from filtered dataset splits.

For each record in the input file, calls ConstructModel.process_sample()
(2-step: latent entity detection → triplet extraction) and saves results
to a JSONL file in real-time (one line per question, flushed immediately).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

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
    parser.add_argument("--output_dir", default="results/graph/0516")
    parser.add_argument("--construct_model_name", default="gpt-4o-mini")
    parser.add_argument("--api_key", default=None, help="SKIML API key (overrides SKIML_API_KEY env var).")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def load_done_uids(output_path: Path) -> Set[str]:
    """Read existing JSONL and return set of already-processed UIDs."""
    done: Set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("uid") or obj.get("id") or obj.get("question", "")
                if uid:
                    done.add(uid)
            except Exception:
                pass
    return done


def get_uid(record: Dict[str, Any]) -> str:
    return record.get("uid") or record.get("id") or record.get("_id") or record.get("question", "")


def main() -> None:
    args = parse_args()

    if args.api_key:
        os.environ["SKIML_API_KEY"] = args.api_key

    input_path = Path(args.input_file) if args.input_file else (
        REPO_ROOT / "datasets" / args.dataset / "filtered" / "train.json"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_questions.jsonl"

    logger.info("Dataset  : %s", args.dataset)
    logger.info("Input    : %s", input_path)
    logger.info("Output   : %s", output_path)

    suffix = input_path.suffix
    with input_path.open() as f:
        if suffix == ".jsonl":
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    if args.max_samples is not None:
        records = records[:args.max_samples]
    logger.info("Records  : %d", len(records))

    done_uids = load_done_uids(output_path)
    logger.info("Already done: %d question(s)", len(done_uids))

    construct_model = ConstructModel(
        construct_model_name=args.construct_model_name,
        dataset_name=args.dataset,
        api_key=args.api_key,
    )

    with output_path.open("a") as out_f:
        for record in tqdm(records, desc=args.dataset):
            uid = get_uid(record)
            if uid in done_uids:
                continue

            question = record.get("question", "")
            try:
                result = construct_model.process_sample({"question": question})
                definition_triples = result.get("definition_triples", [])
                triples = result.get("triples", [])
            except Exception as exc:
                logger.warning("Extraction failed [%s]: %s", uid, exc)
                definition_triples = []
                triples = []

            entry = {
                "uid": uid,
                "question": question,
                "question_graph": {
                    "definition_triples": definition_triples,
                    "triples": triples,
                },
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            done_uids.add(uid)

    logger.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
