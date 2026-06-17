#!/usr/bin/env python
import argparse
import inspect
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainerCallback,
)


LOGGER = logging.getLogger(__name__)


def import_trainer_stack():
    """Import Trainer classes after disabling broken optional PEFT auto-detection."""
    try:
        import transformers.utils as utils
        import transformers.utils.import_utils as import_utils

        import_utils.is_peft_available.cache_clear()
        import_utils.is_peft_available = lambda: False
        utils.is_peft_available = lambda: False
    except Exception as exc:
        LOGGER.warning("Could not disable PEFT auto-detection before Trainer import: %s", exc)

    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    return DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5/FLAN-T5 to generate triples from documents."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="scripts/finetune/data/document/document.json",
        help="JSON file with records containing an input field and `triples` fields.",
    )
    parser.add_argument(
        "--input_field",
        type=str,
        default="document",
        help="Key in each JSON record to use as the model input (e.g. 'document' or 'question').",
    )
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune/flan-t5-base-document-triples")
    parser.add_argument("--run_name", type=str, default="flan-t5-base-document-triples")
    parser.add_argument("--prompt_prefix", type=str, default="extract triples: ")
    parser.add_argument(
        "--triple_delimiter",
        type=str,
        default="\\n",
        help="Delimiter between generated triples. Use '\\n' for newline or e.g. '<TRIPLE>'.",
    )
    parser.add_argument(
        "--additional_special_tokens",
        type=str,
        default="[SEP],[PREP]",
        help="Comma-separated structural tokens to add to the tokenizer.",
    )
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--generation_max_length", type=int, default=512)
    parser.add_argument("--generation_num_beams", type=int, default=4)
    parser.add_argument("--preview_samples", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training.")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging. Requires wandb login/key in the environment.",
    )
    parser.add_argument("--wandb_project", type=str, default="graph-document-triples")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


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


def triples_to_text(triples: Sequence[str], delimiter: str) -> str:
    return delimiter.join(triple.strip() for triple in triples if triple and triple.strip())


def make_compute_metrics(tokenizer, delimiter: str):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_precision, all_recall, all_f1 = [], [], []
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_triples = {t.strip() for t in pred.split(delimiter) if t.strip()}
            gold_triples = {t.strip() for t in label.split(delimiter) if t.strip()}

            if not gold_triples and not pred_triples:
                all_precision.append(1.0)
                all_recall.append(1.0)
                all_f1.append(1.0)
                continue

            tp = len(pred_triples & gold_triples)
            precision = tp / len(pred_triples) if pred_triples else 0.0
            recall = tp / len(gold_triples) if gold_triples else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        return {
            "precision": round(sum(all_precision) / len(all_precision), 4),
            "recall": round(sum(all_recall) / len(all_recall), 4),
            "f1": round(sum(all_f1) / len(all_f1), 4),
        }

    return compute_metrics


def parse_special_tokens(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def add_structural_tokens(tokenizer, model, tokens: Sequence[str]) -> None:
    if not tokens:
        return

    existing_vocab = tokenizer.get_vocab()
    new_tokens = [token for token in tokens if token not in existing_vocab]
    if not new_tokens:
        LOGGER.info("All structural tokens already exist in the tokenizer: %s", list(tokens))
        return

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    LOGGER.info("Added %d structural special tokens: %s", num_added, new_tokens)


def load_records(
    path: Path, delimiter: str, input_field: str = "document", max_examples: int | None = None
) -> List[Dict[str, str]]:
    LOGGER.info("Loading data from %s (input_field=%s)", path, input_field)
    with path.open("r", encoding="utf-8") as f:
        raw_records = json.load(f)

    records: List[Dict[str, str]] = []
    iterator = raw_records[:max_examples] if max_examples else raw_records
    for row in tqdm(iterator, desc="Preparing examples"):
        source = str(row.get(input_field, "")).strip()
        triples = row.get("triples", [])
        target = triples_to_text(triples if isinstance(triples, list) else [], delimiter)
        if source and target:
            records.append({"source": source, "target": target})

    if not records:
        raise ValueError(f"No usable records found in {path}")

    LOGGER.info("Loaded %d usable records", len(records))
    return records


def split_records(
    records: List[Dict[str, str]], train_ratio: float, seed: int
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
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
    LOGGER.info("Train examples: %d | Eval examples: %d", len(train_records), len(eval_records))
    return train_records, eval_records


@dataclass
class DocumentTripleDataset(Dataset):
    records: List[Dict[str, str]]
    tokenizer: object
    prompt_prefix: str
    max_source_length: int
    max_target_length: int

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        source = self.prompt_prefix + record["source"]
        model_inputs = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
        )
        try:
            labels = self.tokenizer(
                text_target=record["target"],
                max_length=self.max_target_length,
                truncation=True,
            )
        except TypeError:
            labels = self.tokenizer(
                record["target"],
                max_length=self.max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class GenerationPreviewCallback(TrainerCallback):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        tokenizer,
        prompt_prefix: str,
        max_source_length: int,
        generation_max_length: int,
        num_beams: int,
        sample_count: int,
        use_wandb: bool,
    ) -> None:
        self.records = list(records[:sample_count])
        self.tokenizer = tokenizer
        self.prompt_prefix = prompt_prefix
        self.max_source_length = max_source_length
        self.generation_max_length = generation_max_length
        self.num_beams = num_beams
        self.sample_count = sample_count
        self.use_wandb = use_wandb

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.records or self.sample_count <= 0:
            return

        was_training = model.training
        model.eval()
        rows = []
        device = next(model.parameters()).device
        for i, record in enumerate(self.records):
            source = self.prompt_prefix + record["source"]
            inputs = self.tokenizer(
                source,
                max_length=self.max_source_length,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=self.generation_max_length,
                    num_beams=self.num_beams,
                )
            prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            rows.append((i, record["source"][:500], prediction, record["target"]))

        LOGGER.info("Generation preview at step %s", state.global_step)
        for i, source, prediction, target in rows:
            LOGGER.info("Sample %d source: %s", i, source.replace("\n", " "))
            LOGGER.info("Sample %d prediction:\n%s", i, prediction)
            LOGGER.info("Sample %d target:\n%s", i, target)

        if self.use_wandb:
            try:
                import wandb

                table = wandb.Table(columns=["sample_id", "document", "prediction", "target"])
                for row in rows:
                    table.add_data(*row)
                wandb.log({"generation_preview": table, "trainer/global_step": state.global_step})
            except Exception as exc:
                LOGGER.warning("Failed to log generation preview to wandb: %s", exc)

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
    additional_special_tokens = parse_special_tokens(args.additional_special_tokens)
    if triple_delimiter != "\n" and triple_delimiter.strip():
        additional_special_tokens.append(triple_delimiter)

    records = load_records(Path(args.data_path), triple_delimiter, args.input_field, args.max_examples)
    train_records, eval_records = split_records(records, args.train_ratio, args.seed)

    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments = import_trainer_stack()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    add_structural_tokens(tokenizer, model, additional_special_tokens)

    train_dataset = DocumentTripleDataset(
        train_records,
        tokenizer,
        args.prompt_prefix,
        args.max_source_length,
        args.max_target_length,
    )
    eval_dataset = DocumentTripleDataset(
        eval_records,
        tokenizer,
        args.prompt_prefix,
        args.max_source_length,
        args.max_target_length,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
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
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": make_compute_metrics(tokenizer, triple_delimiter),
        "callbacks": [
            GenerationPreviewCallback(
                eval_records,
                tokenizer,
                args.prompt_prefix,
                args.max_source_length,
                args.generation_max_length,
                args.generation_num_beams,
                args.preview_samples,
                args.wandb,
            )
        ],
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    LOGGER.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    LOGGER.info("Running final evaluation")
    metrics = trainer.evaluate()
    LOGGER.info("Final metrics: %s", metrics)

    LOGGER.info("Saving model and tokenizer to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
