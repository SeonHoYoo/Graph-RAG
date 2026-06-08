import argparse
import json
import logging
import os
import sys
from typing import Any, Tuple

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_library.construct_model import ConstructModel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default=None)
    return parser.parse_args()


def infer_dataset(input_path: str) -> str:
    parts = os.path.normpath(input_path).split(os.sep)
    if "datasets" in parts:
        dataset_idx = parts.index("datasets") + 1
        if dataset_idx < len(parts):
            return parts[dataset_idx]
    path_lower = input_path.lower()
    if "2wiki" in path_lower or "2wikimultihopqa" in path_lower:
        return "2wikimultihopqa"
    if "hotpotqa" in path_lower:
        return "hotpotqa"
    if "musique" in path_lower:
        return "musique"
    raise ValueError("Unable to infer dataset name from input_file_path. Pass a path under datasets/<dataset>/claims.")


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input_file_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataset = infer_dataset(input_path)
    construct_model = ConstructModel(
        construct_model_name=args.model_name,
        dataset_name=dataset,
        api_key=args.api_key,
        batch_size=args.batch_size,
    )

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    results: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Extracting question graphs"):
        question = sample.get("question", "")
        question_sample = {"question": question}
        try:
            question_sample = construct_model.process_sample(question_sample)
            question_graph = {
                "definition_triples": question_sample.get("definition_triples", []),
                "triples": question_sample.get("triples", []),
            }
        except Exception as exc:
            logger.error("Failed to extract question_graph for index=%s: %s", sample.get("index"), exc)
            question_graph = {"definition_triples": [], "triples": []}

        results.append({
            "index": sample.get("index"),
            "uid": sample.get("uid"),
            "question": question,
            "question_graph": question_graph,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_filename or f"question_graph_{os.path.splitext(os.path.basename(input_path))[0]}.json"
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
