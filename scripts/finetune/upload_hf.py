#!/usr/bin/env python
"""Upload a fine-tuned LoRA adapter to HuggingFace Hub.

Uploads only the final model files from output_dir, excluding checkpoint-* subdirs.
"""
import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the final model directory.")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo id, e.g. doupari/Llama-3.2-1B-Instruct-think.")
    parser.add_argument("--token", default=None, help="HuggingFace token (falls back to HF_TOKEN env var).")
    parser.add_argument("--private", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required: pass --token or set HF_TOKEN env var.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    api = HfApi(token=token)

    logger.info("Creating repo: %s (private=%s)", args.repo_id, args.private)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    logger.info("Uploading from %s (excluding checkpoint-*)", model_dir)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        ignore_patterns=["checkpoint-*/**", "checkpoint-*"],
    )

    logger.info("Done: https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
