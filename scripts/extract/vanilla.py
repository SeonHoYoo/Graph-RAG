#!/usr/bin/env python
"""Extract knowledge graphs from vanilla 0407 open-book results using finetuned models.

Two models are loaded at startup:
  - document model      : SYSTEM_PROMPTS["document"]       per search_result (batched per turn)
  - think+search model  : SYSTEM_PROMPTS["question"]       for the question
                          SYSTEM_PROMPTS["think+search"]   for each reasoning step

Output: one JSONL line per sample (resume-safe by uid).

Output schema per line:
  {
    uid, question: {text, triples},
    retrieval_turns: [
      {
        turn, query,
        reasoning_step: {text, triples},
        documents: [{text, triples}, ...]
      }, ...
    ],
    answer_steps: [{text, triples}, ...]
  }
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

SYSTEM_PROMPTS = {
    "document": (
        "You are a knowledge graph extraction expert. "
        "Extract all factual triples from the given document. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "question": (
        "You are a knowledge graph extraction expert. "
        "Given a multi-hop question, extract the reasoning triples that represent the relationships needed to answer it. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "think+search": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step and its search query, extract the factual triples it asserts. "
        "For facts already stated, use concrete values. For facts still being searched, use placeholders like (ENT1), (ENT2). "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_filename", default=None)
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--document_model_path", required=True,
                        help="Path to finetuned document model (LoRA checkpoint or merged).")
    parser.add_argument("--think_search_model_path", required=True,
                        help="Path to finetuned question+think+search model.")
    parser.add_argument("--document_max_new_tokens", type=int, default=384)
    parser.add_argument("--think_search_max_new_tokens", type=int, default=128)
    parser.add_argument("--merge_lora", action="store_true",
                        help="Merge LoRA weights into base model before inference.")
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def is_lora_checkpoint(model_dir: Path) -> bool:
    return (model_dir / "adapter_config.json").exists()


def get_base_model_name(model_dir: Path) -> str:
    adapter_cfg = model_dir / "adapter_config.json"
    if adapter_cfg.exists():
        with adapter_cfg.open() as f:
            return json.load(f)["base_model_name_or_path"]
    return str(model_dir)


def load_model(model_path: str, args: argparse.Namespace) -> Tuple[Any, Any]:
    from peft import PeftModel

    model_dir = Path(model_path)
    base_name = get_base_model_name(model_dir)
    logger.info("Loading base model: %s", base_name)

    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)

    if is_lora_checkpoint(model_dir):
        logger.info("Loading LoRA adapter: %s", model_dir)
        model = PeftModel.from_pretrained(model, str(model_dir))
        if args.merge_lora:
            logger.info("Merging LoRA weights")
            model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # warmup
    dummy = tokenizer(["warmup"], return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        model.generate(**dummy, max_new_tokens=4, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)

    return model, tokenizer


# ── generation helpers ────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, messages: List[Dict], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def generate_single(model, tokenizer, user_content: str, system_prompt: str,
                    max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_batch(model, tokenizer, user_contents: List[str], system_prompt: str,
                   max_new_tokens: int) -> List[str]:
    texts = [
        apply_chat_template(
            tokenizer,
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": u}],
            add_generation_prompt=True,
        )
        for u in user_contents
    ]
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=1536).to(next(model.parameters()).device)
    tokenizer.padding_side = "right"

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in outputs
    ]


def parse_triples(response: str) -> List[str]:
    return [line.strip() for line in response.split("\n") if line.strip()]


# ── extraction ────────────────────────────────────────────────────────────────

def extract_question(ts_model, ts_tok, question: str, max_new_tokens: int) -> Dict[str, Any]:
    response = generate_single(ts_model, ts_tok, question, SYSTEM_PROMPTS["question"], max_new_tokens)
    return {"text": question, "triples": parse_triples(response)}


def extract_reasoning_step(ts_model, ts_tok, step_text: str, query: str,
                            max_new_tokens: int) -> Dict[str, Any]:
    step_text = (step_text or "").strip()
    user_content = f"{step_text}\n[Search] {query}" if query else step_text
    response = generate_single(ts_model, ts_tok, user_content, SYSTEM_PROMPTS["think+search"], max_new_tokens)
    return {"text": step_text, "triples": parse_triples(response)}


def extract_documents(doc_model, doc_tok, docs: List[str], max_new_tokens: int) -> List[Dict[str, Any]]:
    if not docs:
        return []
    responses = generate_batch(doc_model, doc_tok, docs, SYSTEM_PROMPTS["document"], max_new_tokens)
    return [{"text": doc, "triples": parse_triples(resp)} for doc, resp in zip(docs, responses)]


# ── data helpers ──────────────────────────────────────────────────────────────

def get_uid(record: Dict[str, Any]) -> str:
    return record.get("uid") or record.get("id") or record.get("_id") or record.get("question", "")


def get_retrieval_turns(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (record.get("retrieval_info") or {}).get("retrieval_turns") or []


def get_reasoning_steps(record: Dict[str, Any]) -> List[str]:
    steps = record.get("reasoning_steps")
    if steps:
        return steps
    return (record.get("retrieval_info") or {}).get("searchr1_reasoning_steps") or []




# ── main ──────────────────────────────────────────────────────────────────────

def process_record(
    record: Dict[str, Any],
    doc_model, doc_tok,
    ts_model, ts_tok,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    uid = get_uid(record)
    question = record.get("question", "")
    raw_turns = get_retrieval_turns(record)
    reasoning_steps = get_reasoning_steps(record)

    question_graph = extract_question(ts_model, ts_tok, question, args.think_search_max_new_tokens)

    processed_turns = []
    for i, raw_turn in enumerate(raw_turns):
        query = (raw_turn.get("query") or "").strip()
        search_results = raw_turn.get("search_results") or []
        step_text = reasoning_steps[i] if i < len(reasoning_steps) else ""

        reasoning_graph = extract_reasoning_step(
            ts_model, ts_tok, step_text, query, args.think_search_max_new_tokens
        )
        documents = extract_documents(
            doc_model, doc_tok, search_results, args.document_max_new_tokens
        )

        processed_turns.append({
            "turn": raw_turn.get("turn", i),
            "query": query,
            "reasoning_step": reasoning_graph,
            "documents": documents,
        })

    answer_steps = [
        extract_reasoning_step(ts_model, ts_tok, step, "", args.think_search_max_new_tokens)
        for step in reasoning_steps[len(raw_turns):]
    ]

    return {
        "uid": uid,
        "question": question_graph,
        "retrieval_turns": processed_turns,
        "answer_steps": answer_steps,
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_filename:
        output_filename = args.output_filename
    elif args.model_tag:
        output_filename = f"{args.dataset}_{args.model_tag}_vanilla_graphs.json"
    else:
        output_filename = f"{args.dataset}_vanilla_graphs.json"
    output_path = output_dir / output_filename

    logger.info("Dataset  : %s", args.dataset)
    logger.info("Input    : %s", input_path)
    logger.info("Output   : %s", output_path)

    with input_path.open() as f:
        records = json.load(f)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    logger.info("Records  : %d", len(records))

    logger.info("Loading document model: %s", args.document_model_path)
    doc_model, doc_tok = load_model(args.document_model_path, args)

    logger.info("Loading think+search model: %s", args.think_search_model_path)
    ts_model, ts_tok = load_model(args.think_search_model_path, args)

    results: List[Dict[str, Any]] = []
    preview_count = 0
    for record in tqdm(records, desc=args.dataset):
        entry = process_record(record, doc_model, doc_tok, ts_model, ts_tok, args)
        results.append(entry)

        if preview_count < 5:
            logger.info("=== Sample %d ===\n%s", preview_count + 1,
                        json.dumps(entry, ensure_ascii=False, indent=2))
            preview_count += 1

    with output_path.open("w") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    logger.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
