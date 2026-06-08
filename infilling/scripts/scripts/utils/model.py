import os

import anthropic
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_library.llm_clients import GPT, Claude, Qwen


def load_model(model_name: str):
    model_name_lower = model_name.lower()

    if model_name_lower.startswith("qwen"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return Qwen(model, tokenizer)

    if model_name_lower.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT models.")
        client = OpenAI(api_key=api_key)
        return GPT(model_name, client)

    if model_name_lower.startswith("claude"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude models.")
        client = anthropic.Anthropic(api_key=api_key)
        return Claude(model_name, client)

    raise ValueError(f"Unsupported model_name: {model_name}")
