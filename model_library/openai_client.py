import os
from typing import Optional

import httpx
from openai import OpenAI


SKIML_API_BASE = "https://147.47.200.198:7861"


def create_openai_client(api_key: Optional[str] = None) -> OpenAI:
    os.environ.setdefault("LITELLM_LOG", "ERROR")

    resolved_api_key = api_key or os.getenv("SKIML_API_KEY")
    if not resolved_api_key:
        raise ValueError("SKIML_API_KEY is required for GPT models.")

    return OpenAI(
        api_key=resolved_api_key,
        base_url=os.getenv("SKIML_API_BASE", SKIML_API_BASE),
        http_client=httpx.Client(verify=False),
    )
