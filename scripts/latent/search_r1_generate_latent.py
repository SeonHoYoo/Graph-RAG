import hashlib
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import transformers


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i] :], target):
                return True
        return False


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def summarize_vector(tensor: torch.Tensor) -> Dict[str, float]:
    flat = tensor.float().cpu()
    return {
        "l2_norm": torch.norm(flat, p=2).item(),
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "max_abs": flat.abs().max().item(),
    }


def sanitize_fragment(value: Optional[Any]) -> str:
    if value is None:
        return "none"
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return sanitized[:120] or "empty"


class SearchR1GenerateLatentInference:
    def __init__(
        self,
        model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        max_turns: int = 4,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        topk: int = 3,
        seed: int = 42,
        device: Optional[str] = None,
        latent_output_dir: Optional[str] = None,
        latent_boundaries: Optional[List[str]] = None,
        latent_layers: Optional[List[int]] = None,
        latent_dtype: str = "float16",
        latent_think_fixed_token_offsets: Optional[List[int]] = None,
        latent_dense_think_stride: Optional[int] = None,
    ):
        set_seed(seed)

        self.model_id = model_id
        self.retriever_url = retriever_url
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.topk = topk
        self.latent_output_dir = latent_output_dir
        self.latent_boundaries = latent_boundaries or ["<think>", "</think>", "<search>", "</search>", "<answer>", "</answer>"]
        self.latent_layers = latent_layers
        self.latent_dtype = latent_dtype
        self.save_dtype = resolve_dtype(latent_dtype)
        self.latent_think_fixed_token_offsets = latent_think_fixed_token_offsets or [1, 5, 10, 20]
        if latent_dense_think_stride is not None and latent_dense_think_stride <= 0:
            raise ValueError("latent_dense_think_stride must be positive when provided.")
        self.latent_dense_think_stride = latent_dense_think_stride

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.curr_eos = [151645, 151643]
        self.curr_search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"

        print(f"Loading model: {model_id}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, self.tokenizer)])
        self.boundary_token_ids = {
            boundary: self.tokenizer.encode(boundary, add_special_tokens=False)
            for boundary in self.latent_boundaries
        }

        print("Model loaded successfully!")

    def _get_query(self, text: str) -> Optional[str]:
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        return None

    def _search(self, query: str) -> list:
        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        try:
            results = requests.post(self.retriever_url, json=payload, timeout=30).json()["result"]
        except Exception as exc:
            print(f"Search error for query '{query}': {exc}")
            return []

        format_reference_list = []
        for doc_item in results[0]:
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0].strip('"\'')
            text = "\n".join(content.split("\n")[1:])
            format_reference_list.append(f"(Title: {title}) {text}")
        return format_reference_list

    def _extract_answer(self, text: str) -> str:
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
        return ""

    def _extract_reasoning(self, text: str) -> str:
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return "\n".join(match.strip() for match in matches)
        return ""

    def _extract_reasoning_steps(self, text: str) -> list:
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        return [match.strip() for match in pattern.findall(text) if match.strip()]

    def _build_result(
        self,
        question: str,
        prompt: str,
        full_response: str,
        cnt: int,
        retrieval_turns: list,
        total_search_results: list,
        last_search_results_list: list,
        latent_trace: list,
        latent_trace_dir: Optional[str],
        initial_thinking: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "prompt": prompt,
            "initial_thinking": initial_thinking,
            "full_response": full_response,
            "predicted_answer": self._extract_answer(full_response),
            "reasoning_path": self._extract_reasoning(full_response),
            "reasoning_steps": self._extract_reasoning_steps(full_response),
            "num_turns": cnt,
            "retrieval_turns": retrieval_turns,
            "total_search_results": total_search_results,
            "last_search_results_list": last_search_results_list,
            "latent_trace": latent_trace,
            "latent_trace_dir": latent_trace_dir,
        }

    def _build_trace_dir(self, question: str, trace_context: Optional[Dict[str, Any]]) -> Optional[str]:
        if not self.latent_output_dir:
            return None
        trace_context = trace_context or {}
        question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        trace_dir = os.path.join(
            self.latent_output_dir,
            f"sample_{sanitize_fragment(trace_context.get('sample_index'))}__uid_{sanitize_fragment(trace_context.get('sample_uid'))}__q_{question_hash}",
        )
        os.makedirs(trace_dir, exist_ok=True)
        return trace_dir

    def _find_boundary_token_spans_from_text(self, generated_text: str) -> List[Tuple[str, int, int]]:
        spans: List[Tuple[str, int, int]] = []
        seen = set()

        for boundary in sorted(self.latent_boundaries, key=len, reverse=True):
            char_start = 0
            while True:
                match_idx = generated_text.find(boundary, char_start)
                if match_idx == -1:
                    break

                prefix_text = generated_text[:match_idx]
                through_boundary_text = generated_text[: match_idx + len(boundary)]
                boundary_start = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))
                boundary_end = len(self.tokenizer.encode(through_boundary_text, add_special_tokens=False))
                key = (boundary, boundary_start, boundary_end)

                if boundary_end > boundary_start and key not in seen:
                    spans.append(key)
                    seen.add(key)

                char_start = match_idx + len(boundary)

        spans.sort(key=lambda item: (item[1], item[2], item[0]))
        return spans

    def _group_boundary_spans(self, generated_text: str) -> Dict[str, List[Tuple[int, int]]]:
        grouped: Dict[str, List[Tuple[int, int]]] = {}
        for boundary, boundary_start, boundary_end in self._find_boundary_token_spans_from_text(generated_text):
            grouped.setdefault(boundary, []).append((boundary_start, boundary_end))
        return grouped

    def _next_span_start(self, spans: List[Tuple[int, int]], min_start: int) -> Optional[Tuple[int, int]]:
        for start, end in spans:
            if start >= min_start:
                return start, end
        return None

    def _add_anchor_spec(
        self,
        specs: List[Dict[str, Any]],
        anchor: str,
        seq_index: int,
        generated_start: Optional[int],
        generated_end: Optional[int],
        capture_position: str,
        source_boundary: Optional[str] = None,
        block_index: Optional[int] = None,
        token_offset: Optional[int] = None,
    ) -> None:
        specs.append(
            {
                "anchor": anchor,
                "boundary": anchor,
                "source_boundary": source_boundary,
                "event_type": "event_anchor",
                "capture_position": capture_position,
                "capture_sequence_index": seq_index,
                "boundary_start_generated_token": generated_start,
                "boundary_end_generated_token": generated_end,
                "block_index": block_index,
                "token_offset": token_offset,
            }
        )

    def _build_event_anchor_specs(
        self,
        input_len: int,
        total_len: int,
        generated_text: str,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        grouped = self._group_boundary_spans(generated_text)

        self._add_anchor_spec(
            specs,
            anchor="pre_generation",
            seq_index=input_len - 1,
            generated_start=None,
            generated_end=None,
            capture_position="prompt_last_token",
        )

        think_spans = grouped.get("<think>", [])
        endthink_spans = grouped.get("</think>", [])
        for think_idx, (think_start, think_end) in enumerate(think_spans):
            self._add_anchor_spec(
                specs,
                anchor="think_marker",
                seq_index=input_len + think_end - 1,
                generated_start=think_start,
                generated_end=think_end,
                capture_position="marker_end",
                source_boundary="<think>",
                block_index=think_idx,
            )

            endthink = self._next_span_start(endthink_spans, think_end)
            content_start = think_end
            content_end = endthink[0] if endthink is not None else total_len - input_len

            if content_start < content_end:
                self._add_anchor_spec(
                    specs,
                    anchor="think_first_content",
                    seq_index=input_len + content_start,
                    generated_start=content_start,
                    generated_end=content_start + 1,
                    capture_position="first_content_token",
                    source_boundary="<think>",
                    block_index=think_idx,
                    token_offset=1,
                )

            for offset in self.latent_think_fixed_token_offsets:
                token_index = content_start + offset - 1
                if token_index < content_end:
                    self._add_anchor_spec(
                        specs,
                        anchor=f"think_token_{offset}",
                        seq_index=input_len + token_index,
                        generated_start=token_index,
                        generated_end=token_index + 1,
                        capture_position="fixed_think_token",
                        source_boundary="<think>",
                        block_index=think_idx,
                        token_offset=offset,
                    )

            if self.latent_dense_think_stride is not None:
                offset = self.latent_dense_think_stride
                while content_start + offset - 1 < content_end:
                    token_index = content_start + offset - 1
                    self._add_anchor_spec(
                        specs,
                        anchor=f"think_dense_token_{offset}",
                        seq_index=input_len + token_index,
                        generated_start=token_index,
                        generated_end=token_index + 1,
                        capture_position="dense_think_token",
                        source_boundary="<think>",
                        block_index=think_idx,
                        token_offset=offset,
                    )
                    offset += self.latent_dense_think_stride

            if endthink is not None and endthink[0] > content_start:
                self._add_anchor_spec(
                    specs,
                    anchor="think_end",
                    seq_index=input_len + endthink[0] - 1,
                    generated_start=endthink[0] - 1,
                    generated_end=endthink[0],
                    capture_position="before_boundary",
                    source_boundary="</think>",
                    block_index=think_idx,
                )

        for search_idx, (search_start, search_end) in enumerate(grouped.get("<search>", [])):
            if search_start > 0:
                self._add_anchor_spec(
                    specs,
                    anchor="search_before",
                    seq_index=input_len + search_start - 1,
                    generated_start=search_start - 1,
                    generated_end=search_start,
                    capture_position="before_boundary",
                    source_boundary="<search>",
                    block_index=search_idx,
                )

        for search_idx, (endsearch_start, endsearch_end) in enumerate(grouped.get("</search>", [])):
            if endsearch_start > 0:
                self._add_anchor_spec(
                    specs,
                    anchor="search_query_end",
                    seq_index=input_len + endsearch_start - 1,
                    generated_start=endsearch_start - 1,
                    generated_end=endsearch_start,
                    capture_position="before_boundary",
                    source_boundary="</search>",
                    block_index=search_idx,
                )

        for answer_idx, (answer_start, answer_end) in enumerate(grouped.get("<answer>", [])):
            if answer_start > 0:
                self._add_anchor_spec(
                    specs,
                    anchor="answer_before",
                    seq_index=input_len + answer_start - 1,
                    generated_start=answer_start - 1,
                    generated_end=answer_start,
                    capture_position="before_boundary",
                    source_boundary="<answer>",
                    block_index=answer_idx,
                )

        for answer_idx, (endanswer_start, endanswer_end) in enumerate(grouped.get("</answer>", [])):
            if endanswer_start > 0:
                self._add_anchor_spec(
                    specs,
                    anchor="answer_end",
                    seq_index=input_len + endanswer_start - 1,
                    generated_start=endanswer_start - 1,
                    generated_end=endanswer_start,
                    capture_position="before_boundary",
                    source_boundary="</answer>",
                    block_index=answer_idx,
                )

        specs = [
            spec
            for spec in specs
            if 0 <= spec["capture_sequence_index"] < total_len
        ]
        specs.sort(
            key=lambda spec: (
                spec["capture_sequence_index"],
                spec["anchor"],
                -1 if spec["block_index"] is None else spec["block_index"],
            )
        )
        return specs

    def _collect_layer_vectors(
        self,
        last_hidden_states: Tuple[torch.Tensor, ...],
        target_layers: List[int],
        seq_index: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        layer_vectors = {}
        layer_summaries = {}
        for layer_idx in target_layers:
            if layer_idx < 0 or layer_idx >= len(last_hidden_states):
                raise ValueError(f"Requested layer {layer_idx} is out of range 0..{len(last_hidden_states)-1}")
            vector = last_hidden_states[layer_idx][0, seq_index, :].detach().to("cpu", dtype=self.save_dtype)
            layer_vectors[str(layer_idx)] = vector
            layer_summaries[str(layer_idx)] = summarize_vector(vector)
        return layer_vectors, layer_summaries

    def _capture_turn_latents(self, prompt: str, input_ids: torch.Tensor, generated_ids: torch.Tensor, turn_dir: Optional[str]) -> List[Dict[str, Any]]:
        if turn_dir is not None:
            os.makedirs(turn_dir, exist_ok=True)

        attention_mask = torch.ones_like(generated_ids)
        with torch.no_grad():
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        last_hidden_states = outputs.hidden_states
        target_layers = self.latent_layers if self.latent_layers is not None else [len(last_hidden_states) - 1]
        generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
        snapshots: List[Dict[str, Any]] = []
        anchor_specs = self._build_event_anchor_specs(input_ids.shape[1], generated_ids.shape[1], generated_text)

        for snapshot_idx, spec in enumerate(anchor_specs):
            seq_index = spec["capture_sequence_index"]
            layer_vectors, layer_summaries = self._collect_layer_vectors(last_hidden_states, target_layers, seq_index)
            tensor_path = None
            if turn_dir is not None:
                tensor_path = os.path.join(turn_dir, f"snapshot_{snapshot_idx:03d}.pt")
                torch.save(
                    {
                        **spec,
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "layers": layer_vectors,
                    },
                    tensor_path,
                )

            generated_end = spec["boundary_end_generated_token"]
            token_end = input_ids.shape[1] + generated_end if generated_end is not None else input_ids.shape[1]
            token_end = min(max(token_end, 0), generated_ids.shape[1])
            anchor_token_text = self.tokenizer.decode(
                generated_ids[0][seq_index : seq_index + 1],
                skip_special_tokens=True,
            )
            snapshots.append(
                {
                    "snapshot_index": snapshot_idx,
                    **spec,
                    "anchor_token_text": anchor_token_text,
                    "decoded_text_through_anchor": self.tokenizer.decode(
                        generated_ids[0][input_ids.shape[1] : token_end],
                        skip_special_tokens=True,
                    ),
                    "tensor_path": tensor_path,
                    "layer_summaries": layer_summaries,
                }
            )

        if turn_dir is not None:
            metadata_path = os.path.join(turn_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "boundaries": self.latent_boundaries,
                        "anchor_scheme": {
                            "event_anchors": [
                                "pre_generation",
                                "think_marker",
                                "think_first_content",
                                "think_end",
                                "search_before",
                                "search_query_end",
                                "answer_before",
                                "answer_end",
                            ],
                            "think_fixed_token_offsets": self.latent_think_fixed_token_offsets,
                            "dense_think_stride": self.latent_dense_think_stride,
                        },
                        "layers": self.latent_layers,
                        "num_snapshots": len(snapshots),
                        "snapshots": snapshots,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
        return snapshots

    def _generate_one_turn(self, prompt: str, trace_dir: Optional[str], turn_idx: int) -> Dict[str, Any]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.temperature,
        )

        generated_tokens = outputs[0][input_ids.shape[1] :]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        turn_dir = os.path.join(trace_dir, f"turn_{turn_idx:02d}") if trace_dir is not None else None
        snapshots = self._capture_turn_latents(prompt, input_ids, outputs, turn_dir)
        return {
            "generated_ids": outputs[0],
            "output_text": output_text,
            "hit_eos": outputs[0][-1].item() in self.curr_eos,
            "turn_dir": turn_dir,
            "snapshots": snapshots,
        }

    def _build_prompt(self, question: str, initial_thinking: Optional[str] = None) -> str:
        question = question.strip()
        if question and question[-1] != "?":
            question += "?"

        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        if initial_thinking:
            prompt += f"\n\n<think>{initial_thinking}</think>\n\n"
        return prompt

    def infer(self, question: str, verbose: bool = False, trace_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = self._build_prompt(question)
        if verbose:
            print("\n\n################# [Start Reasoning + Searching] ##################\n\n")
            print(prompt)

        full_response = ""
        cnt = 0
        retrieval_turns = []
        total_search_results = []
        last_search_results_list = []
        latent_trace = []
        trace_dir = self._build_trace_dir(question, trace_context)

        while cnt < self.max_turns:
            generation = self._generate_one_turn(prompt, trace_dir, cnt)
            output_text = generation["output_text"]
            latent_trace.append(
                {
                    "turn": cnt,
                    "turn_dir": generation["turn_dir"],
                    "snapshots": generation["snapshots"],
                }
            )

            if generation["hit_eos"]:
                full_response += output_text
                if verbose:
                    print(output_text)
                break

            tmp_query = self._get_query(self.tokenizer.decode(generation["generated_ids"], skip_special_tokens=True))
            if tmp_query:
                last_search_results_list = self._search(tmp_query)
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)
                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "model_output": output_text,
                    "search_results": last_search_results_list.copy(),
                    "retrieved_docs": [],
                }
                for result in last_search_results_list:
                    if result.startswith("(Title: "):
                        title_end = result.find(")")
                        if title_end != -1:
                            turn_info["retrieved_docs"].append(result[8:title_end])
                retrieval_turns.append(turn_info)
                search_results = "\n".join([f"Doc {idx+1}{result}" for idx, result in enumerate(last_search_results_list)])
            else:
                search_results = ""

            search_text = self.curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            full_response += search_text
            cnt += 1

            if verbose:
                print(search_text)

        print("Question:", question)
        print("Full Response:", full_response)
        print("\n===\n")

        return self._build_result(
            question=question,
            prompt=prompt,
            full_response=full_response,
            cnt=cnt,
            retrieval_turns=retrieval_turns,
            total_search_results=total_search_results,
            last_search_results_list=last_search_results_list,
            latent_trace=latent_trace,
            latent_trace_dir=trace_dir,
        )
