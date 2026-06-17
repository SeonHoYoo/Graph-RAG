"""
Question graph extraction for train split raw files.

Uses the same system prompt the model was fine-tuned with ("question" type):
  system: "You are a knowledge graph extraction expert. Given a multi-hop question,
           extract the reasoning triples ..."
  user:   <question text>

Output per sample: uid, index, question, question_graph
  question_graph: { "definition_triples": [...], "triples": [...] }
  - lines starting with "(ENTk) [SEP] is [SEP] ..." → definition_triples
  - all other [SEP] lines → triples
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a knowledge graph extraction expert. "
    "Given a multi-hop question, extract the reasoning triples that represent "
    "the relationships needed to answer it. "
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
        # load tokenizer from adapter dir (has chat_template) falling back to base model
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

def generate(model, tokenizer, question: str, max_new_tokens: int = 300) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
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


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_output(text: str) -> list:
    triples = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or "[SEP]" not in line:
            continue
        if line not in triples:
            triples.append(line)
    return triples


def process_sample(question: str, model, tokenizer) -> dict:
    output = generate(model, tokenizer, question)
    triples = parse_output(output)
    return {"definition_triples": [], "triples": triples}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_filename", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--hf_home", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input_file) as f:
        samples = json.load(f)
    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info("Loaded %d samples from %s", len(samples), args.input_file)

    model_dir = Path(args.model_dir)
    model, tokenizer = load_model(model_dir, args.hf_home)

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_filename or f"{args.dataset}_question_graph.json"
    output_path = os.path.join(args.output_dir, out_name)

    # resume if partial output exists
    results = []
    done_uids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_uids = {r["uid"] for r in results}
        logger.info("Resuming: %d already done", len(done_uids))

    for idx, sample in enumerate(tqdm(samples, desc=f"Extracting [{args.dataset}]")):
        uid = sample.get("uid") or sample.get("id", f"idx_{idx}")
        if uid in done_uids:
            continue

        question = sample.get("question", "")
        try:
            question_graph = process_sample(question, model, tokenizer)
        except Exception as e:
            logger.error("Failed for uid=%s: %s", uid, e)
            question_graph = {"definition_triples": [], "triples": []}

        results.append({
            "uid": uid,
            "index": idx,
            "question": question,
            "question_graph": question_graph,
        })

        if len(results) == 5:
            logger.info("=== Sample outputs (first 5) ===")
            for r in results:
                logger.info("uid: %s", r["uid"])
                logger.info("question: %s", r["question"])
                logger.info("definition_triples: %s", r["question_graph"]["definition_triples"])
                logger.info("triples: %s", r["question_graph"]["triples"])
                logger.info("---")

        if len(results) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info("Checkpoint saved (%d done)", len(results))

    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d results → %s", len(results), output_path)


if __name__ == "__main__":
    main()
