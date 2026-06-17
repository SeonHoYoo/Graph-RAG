#!/usr/bin/env python
import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import time

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model


LOGGER = logging.getLogger(__name__)

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
    "think": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step from a chain-of-thought, extract the factual triples it asserts. "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "think+search": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step and its search query, extract the factual triples it asserts. "
        "For facts already stated, use concrete values. For facts still being searched, use placeholders like (ENT1), (ENT2). "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
    "think+nosearch": (
        "You are a knowledge graph extraction expert. "
        "Given a reasoning step from a chain-of-thought, extract the factual triples it asserts. "
        "For facts already stated, use concrete values. For facts still being searched, use placeholders like (ENT1), (ENT2). "
        "Output one triple per line using the format: subject [SEP] relation [SEP] object"
    ),
}


def get_system_prompt(input_field: str) -> str:
    return SYSTEM_PROMPTS.get(input_field, SYSTEM_PROMPTS["document"])


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )


def set_all_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def parse_escaped_text(value: str) -> str:
    return value.encode("utf-8").decode("unicode_escape")


def get_default_lora_modules(model_name: str) -> str:
    """Return appropriate LoRA target modules based on model family."""
    name = model_name.lower()
    if "phi" in name:
        return "qkv_proj,o_proj,gate_up_proj,down_proj"
    return "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune causal LM with LoRA for document → triple extraction."
    )
    # Data
    parser.add_argument("--data_path", type=str, default="scripts/finetune/data/document/document.json")
    parser.add_argument("--input_field", type=str, default="document", help="Key to use as model input (e.g. 'document', 'question', 'think').")
    parser.add_argument("--target_field", type=str, default=None, help="Key for target triples. Defaults to 'triples', falls back to 'graph'.")
    parser.add_argument("--triple_delimiter", type=str, default="\\n")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune/qwen3-4b-instruct-document-triples")
    parser.add_argument("--run_name", type=str, default="qwen3-4b-instruct-document-triples")
    parser.add_argument("--flash_attention", action="store_true", help="Use flash_attention_2 (requires flash-attn).")
    # Sequence lengths
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--generation_max_new_tokens", type=int, default=512)
    parser.add_argument("--generation_num_beams", type=int, default=1, help="1=greedy, faster eval.")
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="",
        help="Comma-separated LoRA target modules. Auto-detected from model name if empty.",
    )
    # Training
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    # Eval
    parser.add_argument("--f1_eval_samples", type=int, default=50, help="Eval records used for auto-regressive triple F1.")
    parser.add_argument("--preview_samples", type=int, default=3)
    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="graph-document-triples")
    parser.add_argument("--wandb_entity", type=str, default=None)
    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def triples_to_text(triples: Sequence[str], delimiter: str) -> str:
    return delimiter.join(t.strip() for t in triples if t and t.strip())


def load_records(path: Path, delimiter: str, input_field: str = "document", target_field: Optional[str] = None, max_examples: Optional[int] = None) -> List[Dict]:
    LOGGER.info("Loading data from %s (input_field=%s)", path, input_field)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            raw_records = [json.loads(line) for line in f if line.strip()]
        else:
            raw_records = json.load(f)

    MIXED_MODES = {"question+think", "question+think+search", "question+think+nosearch"}
    include_search = "nosearch" not in input_field

    def _extract(row: Dict, mode: str):
        """Return (source_text, target_text, source_type) for a row."""
        source_type = row.get("source", mode) if mode in MIXED_MODES else mode

        if source_type in ("think+search", "think+nosearch"):
            think = str(row.get("think_text", "")).strip()
            search = str(row.get("search_query", "")).strip()
            src = f"{think}\n[Search] {search}" if (search and include_search and source_type == "think+search") else think
        elif source_type == "think":
            src = str(row.get("step_text", "")).strip()
        elif source_type == "question":
            src = str(row.get("question", "")).strip()
        else:
            src = str(row.get(source_type, "")).strip()

        if target_field:
            triples = row.get(target_field, [])
        elif source_type == "question":
            qg = row.get("question_graph", {})
            triples = qg.get("definition_triples", []) + qg.get("triples", [])
        else:
            triples = row.get("triples") or row.get("graph", [])
        tgt = triples_to_text(triples if isinstance(triples, list) else [], delimiter)
        return src, tgt, source_type

    records: List[Dict] = []
    iterator = raw_records[:max_examples] if max_examples else raw_records
    for row in tqdm(iterator, desc="Preparing examples"):
        src, tgt, source_type = _extract(row, input_field)
        if src and tgt:
            rec = {"source": src, "target": tgt}
            if input_field in MIXED_MODES:
                rec["source_type"] = source_type
            records.append(rec)

    if not records:
        raise ValueError(f"No usable records found in {path}")

    LOGGER.info("Loaded %d usable records", len(records))
    return records


def split_records(records: List[Dict], train_ratio: float, seed: int):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_size = max(1, int(len(shuffled) * train_ratio))
    train_records = shuffled[:train_size]
    eval_records = shuffled[train_size:]
    if not eval_records:
        eval_records = train_records[-1:]
        train_records = train_records[:-1] or train_records
    LOGGER.info("Train: %d | Eval: %d", len(train_records), len(eval_records))
    return train_records, eval_records


def apply_chat_template(tokenizer, messages: List[Dict], add_generation_prompt: bool) -> str:
    """Apply chat template. Disables thinking for Qwen3; falls back for other models."""
    try:
        # Qwen3 supports enable_thinking; other models raise TypeError
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


class QwenDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int, input_field: str = "source", system_prompt: str = ""):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_field = input_field
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        source_type = record.get("source_type")
        system_prompt = get_system_prompt(source_type) if source_type else self.system_prompt
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["source"]},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": record["target"]}]

        prompt_text = apply_chat_template(self.tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = apply_chat_template(self.tokenizer, full_messages, add_generation_prompt=False)

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + list(full_ids[prompt_len:])

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def compute_triple_f1(pred_text: str, gold_text: str, delimiter: str) -> tuple:
    pred_triples = {t.strip() for t in pred_text.split(delimiter) if t.strip()}
    gold_triples = {t.strip() for t in gold_text.split(delimiter) if t.strip()}
    if not gold_triples and not pred_triples:
        return 1.0, 1.0, 1.0
    tp = len(pred_triples & gold_triples)
    precision = tp / len(pred_triples) if pred_triples else 0.0
    recall = tp / len(gold_triples) if gold_triples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class GenerationF1Callback(TrainerCallback):
    def __init__(
        self,
        eval_records: Sequence[Dict],
        tokenizer,
        delimiter: str,
        max_length: int,
        generation_max_new_tokens: int,
        num_beams: int,
        f1_eval_samples: int,
        preview_samples: int,
        use_wandb: bool,
        system_prompt: str = "",
    ) -> None:
        # For question+think mode: guarantee at least 1 sample per source_type
        types = {r.get("source_type") for r in eval_records if r.get("source_type")}
        if len(types) > 1:
            by_type: Dict[str, List[Dict]] = {}
            for r in eval_records:
                by_type.setdefault(r["source_type"], []).append(r)
            guaranteed = [recs[0] for recs in by_type.values()]
            guaranteed_ids = {id(r) for r in guaranteed}
            remaining = [r for r in eval_records if id(r) not in guaranteed_ids]
            self.records = (guaranteed + remaining)[: max(f1_eval_samples, len(guaranteed))]
            preview_guaranteed = [recs[0] for recs in by_type.values()]
            preview_guaranteed_ids = {id(r) for r in preview_guaranteed}
            preview_remaining = [r for r in self.records if id(r) not in preview_guaranteed_ids]
            self.preview_records = (preview_guaranteed + preview_remaining)[:max(preview_samples, len(preview_guaranteed))]
        else:
            self.records = list(eval_records[:f1_eval_samples])
            self.preview_records = self.records[:preview_samples]
        self.tokenizer = tokenizer
        self.delimiter = delimiter
        self.max_length = max_length
        self.generation_max_new_tokens = generation_max_new_tokens
        self.num_beams = num_beams
        self.use_wandb = use_wandb
        self.system_prompt = system_prompt

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.records:
            return

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        all_p, all_r, all_f1, all_latency, all_tokens = [], [], [], [], []
        by_type_f1: Dict[str, List[float]] = {}
        preview_rows = []
        preview_record_ids = {id(r) for r in self.preview_records}

        eval_start = time.perf_counter()
        for record in self.records:
            source_type = record.get("source_type")
            system_prompt = get_system_prompt(source_type) if source_type else self.system_prompt
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record["source"]},
            ]
            prompt_text = apply_chat_template(self.tokenizer, prompt_messages, add_generation_prompt=True)
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.generation_max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            latency = time.perf_counter() - t0

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            p, r, f1 = compute_triple_f1(prediction, record["target"], self.delimiter)
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)
            all_latency.append(latency)
            all_tokens.append(len(generated_ids))
            if source_type:
                by_type_f1.setdefault(source_type, []).append(f1)  # only in question+think mode

            if id(record) in preview_record_ids:
                preview_rows.append((source_type, record["source"][:300], prediction, record["target"]))

        total_elapsed = time.perf_counter() - eval_start
        mean_p = sum(all_p) / len(all_p)
        mean_r = sum(all_r) / len(all_r)
        mean_f1 = sum(all_f1) / len(all_f1)
        mean_latency = sum(all_latency) / len(all_latency)
        mean_tokens = sum(all_tokens) / len(all_tokens)
        throughput = sum(all_tokens) / total_elapsed  # tokens/sec

        LOGGER.info(
            "Generation F1 @ step %d — P: %.4f | R: %.4f | F1: %.4f",
            state.global_step, mean_p, mean_r, mean_f1,
        )
        for stype, f1s in by_type_f1.items():
            LOGGER.info("  [%s] F1: %.4f (%d samples)", stype, sum(f1s) / len(f1s), len(f1s))
        LOGGER.info(
            "Latency @ step %d — mean: %.3fs | tokens/sample: %.1f | throughput: %.1f tok/s | total: %.1fs (%d samples)",
            state.global_step, mean_latency, mean_tokens, throughput, total_elapsed, len(self.records),
        )
        for i, (stype, doc, pred, gold) in enumerate(preview_rows):
            LOGGER.info("Preview %d [%s] | input: %s", i, stype, doc.replace("\n", " "))
            LOGGER.info("Preview %d [%s] | pred:\n%s", i, stype, pred)
            LOGGER.info("Preview %d [%s] | gold:\n%s", i, stype, gold)

        if self.use_wandb:
            try:
                import wandb
                table = wandb.Table(columns=["source_type", "input", "prediction", "target"])
                for stype, doc, pred, gold in preview_rows:
                    table.add_data(stype, doc, pred, gold)
                per_type_metrics = {f"gen_f1/{stype}_f1": sum(f1s) / len(f1s) for stype, f1s in by_type_f1.items()}
                wandb.log({
                    "gen_f1/precision": mean_p,
                    "gen_f1/recall": mean_r,
                    "gen_f1/f1": mean_f1,
                    **per_type_metrics,
                    "gen_f1/preview": table,
                    "latency/mean_sec": mean_latency,
                    "latency/tokens_per_sample": mean_tokens,
                    "latency/throughput_tok_per_sec": throughput,
                    "trainer/global_step": state.global_step,
                })
            except Exception as exc:
                LOGGER.warning("wandb logging failed: %s", exc)

        if was_training:
            model.train()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_all_seed(args.seed)

    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    triple_delimiter = parse_escaped_text(args.triple_delimiter)
    system_prompt = get_system_prompt(args.input_field)
    LOGGER.info("System prompt for mode '%s': %s", args.input_field, system_prompt)
    records = load_records(Path(args.data_path), triple_delimiter, args.input_field, args.target_field, args.max_examples)
    train_records, eval_records = split_records(records, args.train_ratio, args.seed)

    is_phi = "phi" in args.model_name_or_path.lower()
    is_mistral = any(kw in args.model_name_or_path.lower() for kw in ("mistral", "ministral"))

    LOGGER.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer_kwargs: Dict[str, Any] = dict(padding_side="right")
    if not is_phi:
        tokenizer_kwargs["trust_remote_code"] = True
    if is_mistral:
        tokenizer_kwargs["fix_mistral_regex"] = True
        tokenizer_kwargs["use_fast"] = False  # TokenizersBackend not available for Ministral
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model from %s", args.model_name_or_path)
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    model_kwargs = dict(dtype=model_dtype)
    if not is_phi:
        model_kwargs["trust_remote_code"] = True
    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    except TypeError:
        # older transformers versions use torch_dtype instead of dtype
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    if not args.lora_target_modules:
        args.lora_target_modules = get_default_lora_modules(args.model_name_or_path)
        LOGGER.info("Auto-detected LoRA target modules: %s", args.lora_target_modules)
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    train_dataset = QwenDataset(train_records, tokenizer, args.max_length, args.input_field, system_prompt)
    eval_dataset = QwenDataset(eval_records, tokenizer, args.max_length, args.input_field, system_prompt)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["wandb"] if args.wandb else [],
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            GenerationF1Callback(
                eval_records=eval_records,
                tokenizer=tokenizer,
                delimiter=triple_delimiter,
                max_length=args.max_length,
                generation_max_new_tokens=args.generation_max_new_tokens,
                num_beams=args.generation_num_beams,
                f1_eval_samples=args.f1_eval_samples,
                preview_samples=args.preview_samples,
                use_wandb=args.wandb,
                system_prompt=system_prompt,
            )
        ],
    )

    LOGGER.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    LOGGER.info("Running final evaluation")
    metrics = trainer.evaluate()
    LOGGER.info("Final metrics: %s", metrics)

    LOGGER.info("Saving LoRA adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
