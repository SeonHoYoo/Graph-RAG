import argparse
import json
import os
import sys
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.latent.search_r1_generate_latent import SearchR1GenerateLatentInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Search-R1 vanilla through model.generate() and save hidden-state snapshots immediately before boundary markers."
    )
    parser.add_argument("--model_name", type=str, default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--question_file", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--input_filename", type=str, default=None)
    parser.add_argument("--input_file_path", type=str, default=None)
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--uid_key", type=str, default="uid")
    parser.add_argument("--index_key", type=str, default="index")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--boundary", action="append", dest="boundaries", default=None)
    parser.add_argument("--layer", action="append", dest="layers", type=int, default=None)
    parser.add_argument(
        "--think_token_offset",
        action="append",
        dest="think_token_offsets",
        type=int,
        default=None,
        help="Save fixed early-token anchors inside each <think> block. Repeatable. Defaults to 1, 5, 10, 20.",
    )
    parser.add_argument(
        "--dense_think_stride",
        type=int,
        default=None,
        help="Optionally save a dense <think> trajectory every N tokens, e.g. 5 for token 5, 10, 15, ...",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
    )
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.question is not None:
        return [{args.question_key: args.question, args.index_key: 0, args.uid_key: "single"}]

    if args.question_file is not None:
        with open(args.question_file, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        return [{args.question_key: line, args.index_key: idx, args.uid_key: f"line_{idx}"} for idx, line in enumerate(lines)]

    input_path = None
    if args.input_file_path is not None:
        input_path = os.path.abspath(args.input_file_path)
    elif args.dataset and args.input_filename:
        input_path = os.path.join(REPO_ROOT, "datasets", args.dataset, "claims", args.input_filename)

    if input_path is None:
        raise ValueError("Provide one of --question, --question_file, or (--dataset and --input_filename) / --input_file_path.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")
    if args.max_samples is not None:
        data = data[: args.max_samples]
    return data


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inferencer = SearchR1GenerateLatentInference(
        model_id=args.model_name,
        retriever_url=args.retriever_url,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        topk=args.topk,
        seed=args.seed,
        latent_output_dir=args.output_dir,
        latent_boundaries=args.boundaries,
        latent_layers=args.layers,
        latent_dtype=args.dtype,
        latent_think_fixed_token_offsets=args.think_token_offsets,
        latent_dense_think_stride=args.dense_think_stride,
    )

    samples = load_questions(args)
    results = []
    for sample in samples:
        question = str(sample.get(args.question_key, "")).strip()
        if not question:
            results.append(
                {
                    "index": sample.get(args.index_key),
                    "uid": sample.get(args.uid_key),
                    "error": f"Missing question key '{args.question_key}'",
                }
            )
            continue

        try:
            result = inferencer.infer(
                question,
                verbose=args.verbose,
                trace_context={
                    "sample_index": sample.get(args.index_key),
                    "sample_uid": sample.get(args.uid_key),
                },
            )
            result["index"] = sample.get(args.index_key)
            result["uid"] = sample.get(args.uid_key)
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "index": sample.get(args.index_key),
                    "uid": sample.get(args.uid_key),
                    "question": question,
                    "error": str(exc),
                }
            )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"summary_path": summary_path, "num_samples": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
