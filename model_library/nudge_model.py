from openai import OpenAI
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import anthropic
import json
import logging
import os

logger = logging.getLogger(__name__)

from model_library.llm_clients import GPT, Claude, Qwen

class NudgeModel:
    def __init__(
        self, 
        model_name: str,
        output_path: str,
        api_key: Optional[str] = None,
    ):
        self.output_path = output_path
        if os.path.exists(self.output_path):
            logger.info(f"Loaded existing generated data from {self.output_path}.")

            with open(self.output_path, 'r') as f:
                self.outputs = json.load(f)        
        else:
            self.outputs = {}
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        if model_name.startswith("gpt"):
            if api_key is None:
                client = None
                self.model = None
            else:
                client = OpenAI(api_key=api_key) #, organization="org-CsaDwJgaYH1LgSRXtOwVARVC")
                self.model = GPT(model_name, client)
        elif model_name.startswith("claude"):
            client = anthropic.Anthropic(api_key=api_key)
            self.model = Claude(model_name, client)
        elif model_name.startswith("Qwen"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = Qwen(model, tokenizer)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def get_user_message(
        self,
        input: str
    ) -> str:
        prompt = f"""Generate a thinking step that STRICTLY follows this exact format:

REQUIRED FORMAT:
- Start with "I need to find out"
- State what information needs to be found
- End with "I'll search for it."

Question: Houston Baptist University was founded in a date. What is the date?
Thinking: I need to find out when was Houston Baptist University founded. I'll search for it.

Question: What is the state?
Thinking: I need to find out what state is being referred to. I'll search for it.

Question: Ladakh found his guidance in religion in a place. What is the place?
Thinking: I need to find out where Ladakh found his guidance in religion. I'll search for it.

Question: a number theatre companies are in residence in the Seattle. What is the number?
Thinking: I need to find out how many theatre companies are in residence in Seattle. I'll search for it.

Question: an individual is the father of Roger Waters. What is the individual?
Thinking: I need to find out who is the father of Roger Waters. I'll search for it.

{input}
Thinking:
"""
        
        return prompt

    def generate_thinking(
        self,
        input: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0
    ) -> str:
        if self.outputs is not None:
            generated_thinking = self.outputs.get(input, None)
            if generated_thinking is not None:
                logger.info(f"Using cached thinking for prompt.")
                return generated_thinking
        
        if self.model is None:
            raise RuntimeError("Model is not initialized due to existing generated data.")
        
        prompt = self.get_user_message(input)
        
        response = self.model.generate(
            user_message=prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        self.outputs[input] = response
        with open(self.output_path, 'w') as f:
            json.dump(self.outputs, f, indent=4)
        
        return response