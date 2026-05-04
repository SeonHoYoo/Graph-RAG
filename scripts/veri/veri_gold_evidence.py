import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Literal, Optional, Tuple

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.model import load_model


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


class GoldEvidenceAnswerability(BaseModel):
    answerable: Literal["yes", "no"]
    predicted_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_trials", type=int, default=3)
    return parser.parse_args()


def build_gold_context(sample: dict[str, Any]) -> str:
    gold_ids = sample.get("gold_id_list", [])
    gold_evidence_list = sample.get("gold_evidence_list", [])
    context_parts = []
    for doc_id, text in zip(gold_ids, gold_evidence_list):
        if isinstance(text, str) and text.strip():
            context_parts.append(f"(Title: {doc_id}) {text.strip()}")
    return "\n".join(context_parts).strip()


def build_prompt(question: str, gold_context: str) -> str:
    return (
        "You are given a question and gold evidence.\n"
        "Decide whether the question is answerable from the evidence.\n"
        "If answerable, provide the short answer.\n"
        "Return valid JSON only in this schema:\n"
        "{\"answerable\": \"yes\" or \"no\", \"predicted_answer\": \"...\"}\n"
        "Rules:\n"
        "- Use answerable='yes' only if the evidence is sufficient to answer the question.\n"
        "- If answerable='no', predicted_answer must be an empty string.\n"
        "- predicted_answer must be a short answer only.\n\n"
        f"Question: {question}\n"
        f"Gold evidence:\n{gold_context}\n"
    )


def extract_json_block(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_answerable(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("y"):
        return "yes"
    if text.startswith("n"):
        return "no"
    return "no"


def normalize_predicted_answer(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    first_line = value.splitlines()[0].strip() if value.splitlines() else ""
    return first_line.strip().strip('"').strip("'")


def parse_answerability_output(raw_output: str) -> Optional[GoldEvidenceAnswerability]:
    parsed = extract_json_block(raw_output)
    if parsed is None:
        return None
    try:
        if hasattr(GoldEvidenceAnswerability, "model_validate"):
            validated = GoldEvidenceAnswerability.model_validate(parsed)
        else:
            validated = GoldEvidenceAnswerability.parse_obj(parsed)
    except ValidationError:
        return None

    validated.answerable = normalize_answerable(validated.answerable)  # type: ignore[assignment]
    validated.predicted_answer = normalize_predicted_answer(validated.predicted_answer)
    if validated.answerable == "no":
        validated.predicted_answer = ""
    return validated


def predict_answerability(sample: dict[str, Any], model_client: Any, max_trials: int) -> Tuple[str, str, str]:
    gold_context = build_gold_context(sample)
    prompt = build_prompt(sample.get("question", ""), gold_context)

    raw_output = ""
    for _ in range(max(1, max_trials)):
        raw_output = model_client.generate(
            user_message=prompt,
            max_tokens=128,
            temperature=0.0,
        )
        validated = parse_answerability_output(raw_output)
        if validated is not None:
            return validated.answerable, validated.predicted_answer, raw_output

    return "no", "", raw_output


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_file_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    model_client = load_model(args.model_name)
    logger.info("Loaded model=%s, samples=%d", args.model_name, len(samples))

    for sample in tqdm(samples, desc="Gold evidence verification"):
        gold_context = build_gold_context(sample)
        answerable, predicted_answer, raw_output = predict_answerability(sample, model_client, args.max_trials)

        aliases = sample.get("answer_aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        if predicted_answer and predicted_answer not in aliases:
            aliases.append(predicted_answer)

        sample["gold_evidence_context"] = gold_context
        sample["answerable"] = answerable
        sample["predicted_answer"] = predicted_answer
        sample["answer_aliases"] = aliases
        sample["gold_evidence_prompt_output"] = raw_output

    input_stem = os.path.splitext(os.path.basename(args.input_file_path))[0]
    output_name = args.output_filename or f"veri_gold_evidence_{input_stem}.json"
    output_path = os.path.join(args.output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
