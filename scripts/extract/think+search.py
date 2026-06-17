#!/usr/bin/env python
"""Extract triplets from <think>/<search> turns in vanilla 0516 JSONL results.

For each sample, parses the full_response in retrieval_info to extract
(<think>, <search>) pairs per turn, calls
ConstructModel.extract_triplets_from_think_search(), and saves one line
per turn to a JSONL file in real-time (resume-safe).

Output schema per line:
  {uid, question, turn_index, think_text, search_query, triples}
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
    parser.add_argument("--input_file", default=None, help="Override default vanilla JSONL path.")
    parser.add_argument("--output_dir", default="results/graph/0516")
    parser.add_argument("--construct_model_name", default="openai/gpt-4.1-mini-2025-04-14")
    parser.add_argument("--api_key", default=None, help="SKIML API key (overrides SKIML_API_KEY env var).")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def find_default_input(dataset: str) -> Path:
    vanilla_dir = REPO_ROOT / "results" / "vanilla" / "0516"
    matches = list(vanilla_dir.glob(f"{dataset}_*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No vanilla JSONL found for {dataset} in {vanilla_dir}")
    return sorted(matches)[-1]


def parse_turns(full_response: str) -> List[Dict[str, str]]:
    """Parse full_response into a list of {think_text, search_query} dicts.

    Each turn is a <think>...</think> block optionally followed by a
    <search>...</search> block. The final <think> may have no search query.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    search_pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)

    think_spans = [(m.start(), m.end(), m.group(1).strip()) for m in think_pattern.finditer(full_response)]
    search_spans = [(m.start(), m.end(), m.group(1).strip()) for m in search_pattern.finditer(full_response)]

    turns = []
    for i, (t_start, t_end, think_text) in enumerate(think_spans):
        # Find the first <search> that comes after this <think> and before the next <think>
        next_think_start = think_spans[i + 1][0] if i + 1 < len(think_spans) else len(full_response)
        search_query = ""
        for s_start, s_end, s_text in search_spans:
            if t_end <= s_start < next_think_start:
                search_query = s_text
                break
        if think_text:
            turns.append({"think_text": think_text, "search_query": search_query})
    return turns


def get_full_response(sample: Dict[str, Any]) -> str:
    retrieval_info = sample.get("retrieval_info") or {}
    if isinstance(retrieval_info, str):
        try:
            retrieval_info = json.loads(retrieval_info)
        except Exception:
            return ""
    return retrieval_info.get("full_response", "")


def load_done_keys(output_path: Path) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("uid") or obj.get("question", "")
                turn_index = obj.get("turn_index")
                if uid and turn_index is not None:
                    done.add((uid, turn_index))
            except Exception:
                pass
    return done


def main() -> None:
    args = parse_args()

    if args.api_key:
        os.environ["SKIML_API_KEY"] = args.api_key

    input_path = Path(args.input_file) if args.input_file else find_default_input(args.dataset)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_think+search.jsonl"

    logger.info("Dataset  : %s", args.dataset)
    logger.info("Input    : %s", input_path)
    logger.info("Output   : %s", output_path)

    with input_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.max_samples is not None:
        records = records[:args.max_samples]
    logger.info("Records  : %d", len(records))

    done_keys = load_done_keys(output_path)
    logger.info("Already done: %d turn(s)", len(done_keys))

    construct_model = ConstructModel(
        construct_model_name=args.construct_model_name,
        dataset_name=args.dataset,
        api_key=args.api_key,
    )

    total_turns = sum(len(parse_turns(get_full_response(r))) for r in records)
    logger.info("Total turns to process: %d", total_turns)

    with output_path.open("a") as out_f:
        for record in tqdm(records, desc=args.dataset):
            uid = record.get("uid") or record.get("question", "")
            question = record.get("question", "")
            full_response = get_full_response(record)
            turns = parse_turns(full_response)

            for turn_index, turn in enumerate(turns):
                if (uid, turn_index) in done_keys:
                    continue

                think_text = turn["think_text"]
                search_query = turn["search_query"]

                if not think_text:
                    triples = []
                else:
                    try:
                        _, triples = construct_model.extract_triplets_from_think_search(
                            think_text=think_text,
                            search_query=search_query,
                        )
                    except Exception as exc:
                        logger.warning("Failed [%s] turn %d: %s", uid, turn_index, exc)
                        triples = []

                entry = {
                    "uid": uid,
                    "question": question,
                    "turn_index": turn_index,
                    "think_text": think_text,
                    "search_query": search_query,
                    "triples": list(triples),
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                done_keys.add((uid, turn_index))

    logger.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
