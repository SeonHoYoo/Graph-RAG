#!/usr/bin/env python
"""Extract triplets from SearchR1 reasoning steps using a fine-tuned local model."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a knowledge graph extraction expert. "
    "Given a reasoning step from a chain-of-thought, extract the factual triples it asserts. "
    "Output one triple per line using the format: subject [SEP] relation [SEP] object"
)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_dir: Path, hf_home: str = None):
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")

    adapter_cfg = model_dir / "adapter_config.json"
    is_lora = adapter_cfg.exists()

    if is_lora:
        with adapter_cfg.open() as f:
            base_name = json.load(f)["base_model_name_or_path"]
        logger.info("Loading base model: %s", base_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s", model_dir)
        model = PeftModel.from_pretrained(model, str(model_dir))
        tok_path = str(model_dir) if (model_dir / "tokenizer_config.json").exists() else base_name
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, local_files_only=(tok_path != base_name))
    else:
        logger.info("Loading full model from %s", model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


# ── generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, text: str, max_new_tokens: int = 300) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_triples(text: str) -> List[str]:
    triples = []
    for line in text.split("\n"):
        line = line.strip()
        if line and "[SEP]" in line and line not in triples:
            triples.append(line)
    return triples


# ── data helpers ──────────────────────────────────────────────────────────────

def get_reasoning_steps(sample: Dict[str, Any]) -> List[str]:
    steps = sample.get("reasoning_steps")
    if steps:
        return steps
    retrieval_info = sample.get("retrieval_info", {}) or {}
    return retrieval_info.get("searchr1_reasoning_steps") or []


def find_default_input(dataset: str) -> Path:
    vanilla_dir = REPO_ROOT / "results" / "vanilla" / "0516"
    matches = list(vanilla_dir.glob(f"{dataset}_*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No vanilla JSONL found for {dataset} in {vanilla_dir}")
    return sorted(matches)[-1]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--hf_home", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file) if args.input_file else find_default_input(args.dataset)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_think.json"

    logger.info("Dataset  : %s", args.dataset)
    logger.info("Input    : %s", input_path)
    logger.info("Output   : %s", output_path)

    with input_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.max_samples:
        records = records[:args.max_samples]
    logger.info("Records  : %d", len(records))

    # resume
    results: List[Dict] = []
    done_uids: Set[str] = set()
    if output_path.exists():
        with output_path.open() as f:
            results = json.load(f)
        done_uids = {r["uid"] for r in results}
        logger.info("Resuming: %d uid(s) already done", len(done_uids))

    model, tokenizer = load_model(Path(args.model_dir), args.hf_home)

    total_steps = sum(len(get_reasoning_steps(r)) for r in records)
    logger.info("Total steps: %d", total_steps)

    new_count = 0
    for record in tqdm(records, desc=args.dataset):
        uid = record.get("uid") or record.get("question", "")
        if uid in done_uids:
            continue

        question = record.get("question", "")
        steps = []
        for step_text in get_reasoning_steps(record):
            step_text = (step_text or "").strip()
            if not step_text:
                triples = []
            else:
                try:
                    output = generate(model, tokenizer, step_text)
                    triples = parse_triples(output)
                except Exception as exc:
                    logger.warning("Failed [%s]: %s", uid, exc)
                    triples = []
            steps.append({"text": step_text, "triples": triples})

        results.append({"uid": uid, "question": question, "steps": steps})
        done_uids.add(uid)
        new_count += 1

        if new_count == 5:
            logger.info("=== Sample outputs (first 5) ===")
            for r in results[-5:]:
                logger.info("uid: %s", r["uid"])
                logger.info("question: %s", r["question"])
                for i, s in enumerate(r["steps"]):
                    logger.info("  step %d: %s", i, s["text"][:100])
                    logger.info("  triples: %s", s["triples"])
                logger.info("---")

    with output_path.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Done. %d uid(s) → %s", len(results), output_path)


if __name__ == "__main__":
    main()
