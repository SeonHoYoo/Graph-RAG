#!/usr/bin/env python
"""Extract triples from retrieved_documents using the fine-tuned document model."""
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
    "Extract all factual triples from the given document. "
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
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True,
            local_files_only=(tok_path != base_name)
        )
    else:
        logger.info("Loading full model from %s", model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


# ── generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, text: str, max_new_tokens: int = 512) -> str:
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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
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


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--hf_home", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_document.json"

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

    new_count = 0
    for record in tqdm(records, desc=args.dataset):
        uid = record.get("uid") or record.get("id", "")
        if uid in done_uids:
            continue

        question = record.get("question", "")
        retrieved_docs = record.get("retrieved_documents", [])

        documents = []
        for doc_text in retrieved_docs:
            doc_text = (doc_text or "").strip()
            if not doc_text:
                triples = []
            else:
                try:
                    output = generate(model, tokenizer, doc_text)
                    triples = parse_triples(output)
                except Exception as exc:
                    logger.warning("Failed [%s]: %s", uid, exc)
                    triples = []
            documents.append({"text": doc_text, "triples": triples})

        results.append({"uid": uid, "question": question, "documents": documents})
        done_uids.add(uid)
        new_count += 1

        if new_count == 5:
            logger.info("=== Sample outputs (first 5) ===")
            for r in results[-5:]:
                logger.info("uid: %s", r["uid"])
                logger.info("question: %s", r["question"])
                for i, d in enumerate(r["documents"][:2]):
                    logger.info("  doc %d: %s", i, d["text"][:80])
                    logger.info("  triples: %s", d["triples"])
                logger.info("---")

    with output_path.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Done. %d uid(s) → %s", len(results), output_path)


if __name__ == "__main__":
    main()
