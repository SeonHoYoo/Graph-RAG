import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a knowledge graph extraction expert. "
    "Extract all factual triples from the given document. "
    "Output one triple per line using the format: subject [SEP] relation [SEP] object"
)


# ── model type detection ──────────────────────────────────────────────────────

def is_lora_checkpoint(model_dir: Path) -> bool:
    return (model_dir / "adapter_config.json").exists()


def is_t5_model(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        # LoRA: check base model name
        adapter_cfg = model_dir / "adapter_config.json"
        if adapter_cfg.exists():
            with adapter_cfg.open() as f:
                base = json.load(f).get("base_model_name_or_path", "")
            return "t5" in base.lower()
        return False
    with config_path.open() as f:
        arch = json.load(f).get("architectures", [""])
    return any("t5" in a.lower() or "seq2seq" in a.lower() for a in arch)


def get_base_model_name(model_dir: Path) -> str:
    adapter_cfg = model_dir / "adapter_config.json"
    if adapter_cfg.exists():
        with adapter_cfg.open() as f:
            return json.load(f)["base_model_name_or_path"]
    return str(model_dir)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_causal_model(model_dir: Path, args):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base_name = get_base_model_name(model_dir)
    LOGGER.info("Loading base model: %s", base_name)

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)

    if is_lora_checkpoint(model_dir):
        LOGGER.info("Loading LoRA adapter from %s", model_dir)
        model = PeftModel.from_pretrained(model, str(model_dir))
        if args.merge_lora:
            LOGGER.info("Merging LoRA weights into base model")
            model = model.merge_and_unload()

    model.eval()
    return model


def load_t5_model(model_dir: Path, args):
    from transformers import AutoModelForSeq2SeqLM

    LOGGER.info("Loading T5 model from %s", model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model


# ── generation ────────────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
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


def generate_causal(model, tokenizer, source: str, args) -> tuple[str, float, int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
    ]
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(
        prompt, return_tensors="pt",
        max_length=args.max_length, truncation=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prediction, latency, len(generated_ids)


def generate_t5(model, tokenizer, source: str, args) -> tuple[str, float, int]:
    inputs = tokenizer(
        source, return_tensors="pt",
        max_length=args.max_length, truncation=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    latency = time.perf_counter() - t0

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return prediction, latency, len(output_ids[0])


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_triple_f1(pred: str, gold: str, delimiter: str) -> tuple:
    pred_set = {t.strip() for t in pred.split(delimiter) if t.strip()}
    gold_set = {t.strip() for t in gold.split(delimiter) if t.strip()}
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Inference + latency measurement for fine-tuned models.")
    parser.add_argument("--model_dir", required=True, help="Path to checkpoint (LoRA or full model).")
    parser.add_argument("--data_path", required=True, help="Input JSON file.")
    parser.add_argument("--input_field", default="think", help="Field to use as input (document | question | think).")
    parser.add_argument("--output_path", default=None, help="Where to save results JSON.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1536, help="Max input token length.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights before inference (slightly faster).")
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument("--hf_home", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_home, "hub")

    model_dir = Path(args.model_dir)
    assert model_dir.exists(), f"Model dir not found: {model_dir}"

    # ── detect model type ──
    t5 = is_t5_model(model_dir)
    LOGGER.info("Model type: %s", "T5 (Seq2Seq)" if t5 else "Causal LM")

    # ── load tokenizer ──
    tok_path = str(model_dir) if not is_lora_checkpoint(model_dir) else get_base_model_name(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── load model ──
    model = load_t5_model(model_dir, args) if t5 else load_causal_model(model_dir, args)
    generate_fn = generate_t5 if t5 else generate_causal

    # ── load data ──
    LOGGER.info("Loading data from %s (field=%s)", args.data_path, args.input_field)
    with open(args.data_path) as f:
        data = json.load(f)
    if args.max_samples:
        data = data[:args.max_samples]
    LOGGER.info("Running inference on %d samples", len(data))

    # ── inference loop ──
    results = []
    all_latency, all_tokens = [], []
    delimiter = "\n"

    total_start = time.perf_counter()
    for record in tqdm(data, desc="Inference"):
        source = str(record.get(args.input_field, "")).strip()
        gold_triples = record.get("triples", [])
        gold_text = delimiter.join(gold_triples) if isinstance(gold_triples, list) else str(gold_triples)

        if not source:
            continue

        prediction, latency, num_tokens = generate_fn(model, tokenizer, source, args)
        p, r, f1 = compute_triple_f1(prediction, gold_text, delimiter)

        all_latency.append(latency)
        all_tokens.append(num_tokens)
        results.append({
            "index": record.get("index"),
            "source": source[:200],
            "prediction": prediction,
            "gold": gold_text,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "latency_sec": round(latency, 4),
            "num_tokens": num_tokens,
        })

    total_elapsed = time.perf_counter() - total_start
    n = len(all_latency)
    mean_latency = sum(all_latency) / n
    mean_tokens = sum(all_tokens) / n
    throughput = sum(all_tokens) / total_elapsed
    mean_f1 = sum(r["f1"] for r in results) / n

    summary = {
        "model_dir": str(model_dir),
        "num_samples": n,
        "mean_f1": round(mean_f1, 4),
        "mean_latency_sec": round(mean_latency, 4),
        "mean_tokens_per_sample": round(mean_tokens, 1),
        "throughput_tok_per_sec": round(throughput, 1),
        "total_elapsed_sec": round(total_elapsed, 1),
    }

    LOGGER.info("=== Results ===")
    for k, v in summary.items():
        LOGGER.info("  %s: %s", k, v)

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved to %s", out)


if __name__ == "__main__":
    main()
