import anthropic
import json
import logging
from openai import OpenAI
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import *
from tqdm import tqdm

from model_library.prompt import construction_prompt, latent_detection_prompt_musique, latent_detection_prompt_hotpotqa, latent_detection_prompt_2wikimultihopqa, triplet_extraction_prompt_musique, triplet_extraction_prompt_hotpotqa, triplet_extraction_prompt_2wikimultihopqa

logger = logging.getLogger(__name__)


class GPT:
    def __init__(
        self, 
        construct_model_name: str,
        client: Any
    ):
        self.construct_model_name = construct_model_name
        self.client = client
    
    
    def generate(
        self, 
        user_message: str, 
        system_message: Optional[str] = None, 
        max_tokens: Optional[int] = 1024, 
        temperature: Optional[float] = 0.0, 
        top_p: Optional[float] = 1.0
    ) -> str:
        
        if not system_message:
            messages=[
                {"role": "user", "content": user_message}
            ]
        else:
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        response = self.client.chat.completions.create(
            model=self.construct_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    
    
    def wait_for_batch_completion(
        self, 
        batch_id: str
    ) -> str:
        
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            status = batch_status.status
            
            if status == "completed":
                return batch_status.output_file_id
            elif status in ["failed", "expired", "cancelled"]:
                self.check_batch_errors(batch_id)
                raise ValueError(f"Batch {batch_id} failed or was cancelled.")
            else:
                logger.info(f"Batch {batch_id} is still processing (status: {status}). Retrying in 30 seconds...")
                time.sleep(30)
    
    
    def check_batch_errors(
        self, 
        batch_id: str
    ) -> None:
        
        batch_status = self.client.batches.retrieve(batch_id)
        if batch_status.errors:
            logger.error(f"Batch {batch_id} errors: {batch_status.errors}")
        else:
            logger.error(f"No specific error details for batch {batch_id}.")
    
    
    def batch_generate(
        self, 
        user_message_list: List[str], 
        system_message_list: Optional[List[str]] = None, 
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.0, 
        top_p: Optional[float] = 1.0
    ) -> List[str]:
        
        if not system_message_list:
            message_list = [
                {
                    "custom_id": f"request-{idx + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.construct_model_name,
                        "messages": [
                            {"role": "user", "content": user_message}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                } for idx, user_message in enumerate(user_message_list)
            ]
        else:
            message_list = [
                {
                    "custom_id": f"request-{idx + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.construct_model_name,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                } for idx, (user_message, system_message) in enumerate(zip(user_message_list, system_message_list))
            ]
        
        # Write to a temporary file (in jsonl format)
        jsonl_file = "make_jsonl.jsonl"
        with open(jsonl_file, encoding="utf-8", mode="w") as file:
            for i in message_list:
                file.write(json.dumps(i) + "\n")
        
        # Create the batch
        batch_input_file = self.client.files.create(file=open(jsonl_file, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        response = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        if response is None:
            raise ValueError("Received no response from batches.create()")
        else:
            logger.info(f"Batch created successfully: {response}")
        
        # Wait for the batch to complete
        batch_id = response.id
        output_file_id = self.wait_for_batch_completion(batch_id)
        
        # Parse the output file content
        if output_file_id:
            file_response = self.client.files.content(output_file_id)
            batch_responses = file_response.text.strip().split("\n")
            content_list = []
            for response_str in batch_responses:
                try:
                    response_json = json.loads(response_str)
                    content = response_json["response"]["body"]["choices"][0]["message"]["content"]
                    content_list.append(content)
                except (KeyError, json.JSONDecodeError) as e:
                    logger.error(f"Error parsing response: {e}")
            
            return content_list
        
        else:
            raise ValueError(f"No output file generated for batch {batch_id}")


class Claude:
    def __init__(
        self, 
        construct_model_name: str,
        client: Any
    ):
        self.construct_model_name = construct_model_name
        self.client = client
    
    
    def generate(
        self, 
        user_message: str, 
        system_message: Optional[str] = None, 
        max_tokens: Optional[int] = 1024, 
        temperature: Optional[float] = 0.0, 
    ) -> str:
        
        if not system_message:
            system_message = "You are an API that strictly returns only what is requested. Do not add any explanation, greeting, or commentary."
        
        messages=[
            {"role": "user", "content": user_message}
        ]
        response = self.client.messages.create(
            model=self.construct_model_name,
            messages=messages,
            system=system_message,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.content[0].text


class Qwen:
    def __init__(
        self, 
        model: Any, 
        tokenizer: Any
    ):
        self.model = model
        self.tokenizer = tokenizer
    
    
    def generate(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0
    ) -> str:
        
        if not system_message:
            messages = [
                {"role": "user", "content": user_message}
            ]
        else:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Use temperature > 0 for sampling
        do_sample = temperature > 0

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample=do_sample,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
