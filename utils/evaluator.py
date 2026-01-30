import argparse
import json
import os
from typing import List, Dict

from metrics.answer import AnswerMetric


def read_json(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        return json.load(file)  # JSON 배열 전체 로드


def evaluate(file_path: str, num_hops=None) -> Dict:
    instances = read_json(file_path)
    answer_metric = AnswerMetric()

    if num_hops is not None:
        instances = [item for item in instances if item.get("num_hops") == num_hops]

    for instance in instances:
        predicted_answer = instance["predicted_answer"]
        ground_truth_answers = [instance["answer"]] + instance.get("answer_aliases", [])

        answer_metric(predicted_answer, ground_truth_answers)

    metrics = {}
    metrics["answer_em"] = round(answer_metric.get_metric()[0], 3)
    metrics["answer_f1"] = round(answer_metric.get_metric()[1], 3)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions (Answer only).")
    parser.add_argument(
        "--input_path",
        type=str,
        help="json file path containing predicted and ground truth answers together.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="(optional) file path to save output metrics."
    )
    args = parser.parse_args()

    metrics_total = evaluate(args.input_path)
    metrics_2 = evaluate(args.input_path, num_hops=2)
    metrics_3 = evaluate(args.input_path, num_hops=3)
    metrics_4 = evaluate(args.input_path, num_hops=4)

    overall = {
        "total": metrics_total,
        "2_hops": metrics_2,
        "3_hops": metrics_3,
        "4_hops": metrics_4
    }

    print(json.dumps(overall, indent=4))

    if not args.output_path:
        args.output_path = args.input_path.replace(".json", ".out")
    print(f"Writing metrics output in: {args.output_path}")
    with open(args.output_path, "w") as file:
        json.dump(overall, file, indent=4)


if __name__ == "__main__":
    main()
