import time
import transformers
import torch
import random
import numpy as np
import requests
import re
from typing import Callable, Optional, Dict, Any


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StopOnSequence(transformers.StoppingCriteria):
    """Custom stopping criterion for generation"""
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False)
                          for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device)
                  for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


class SearchR1Inference:
    """SearchR1 inference class for question answering with iterative search"""

    def __init__(
        self,
        model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        max_turns: int = 4,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        topk: int = 3,
        seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize SearchR1 inference model

        Args:
            model_id: HuggingFace model ID
            retriever_url: URL for retrieval service
            max_turns: Maximum number of search iterations
            max_new_tokens: Maximum tokens to generate per turn
            temperature: Sampling temperature
            topk: Number of documents to retrieve per search
            seed: Random seed for reproducibility
            device: Device to use (auto-detected if None)
        """
        set_seed(seed)

        self.model_id = model_id
        self.retriever_url = retriever_url
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.topk = topk

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model-specific settings
        self.curr_eos = [151645, 151643]  # for Qwen2.5 series models
        self.curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

        # Initialize tokenizer and model
        print(f"Loading model: {model_id}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None

        # Initialize stopping criteria
        target_sequences = ["</search>", " </search>", "</search>\n",
                          " </search>\n", "</search>\n\n", " </search>\n\n"]
        self.stopping_criteria = transformers.StoppingCriteriaList([
            StopOnSequence(target_sequences, self.tokenizer)
        ])

        print("Model loaded successfully!")

    def _get_query(self, text: str) -> Optional[str]:
        """Extract search query from text"""
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        return None

    def _search(self, query: str) -> list:
        """Perform search via retriever API"""
        payload = {
            "queries": [query],
            "topk": self.topk,
            "return_scores": True
        }
        try:
            results = requests.post(self.retriever_url, json=payload, timeout=30).json()['result']
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
            return []

        def _passages2string(retrieval_result):
            format_reference_list = []
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                # Remove surrounding quotes if present
                title = title.strip('"\'')
                text = "\n".join(content.split("\n")[1:])
                format_reference_list.append(f"(Title: {title}) {text}")
            return format_reference_list

        return _passages2string(results[0])

    def _extract_answer(self, text: str) -> str:
        """Extract answer from <answer> tags"""
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
        return ""
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from <think> tags"""
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # 모든 reasoning을 합쳐서 반환
            return "\n".join(match.strip() for match in matches)
        return ""

    def _extract_reasoning_steps(self, text: str) -> list:
        """Extract each reasoning step from <think> tags in order."""
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
        question_latency_sec: float = 0.0,
        initial_thinking: Optional[str] = None,
    ) -> Dict[str, Any]:
        turn_generate_latencies = [
            t["generate_latency_sec"] for t in retrieval_turns if "generate_latency_sec" in t
        ]
        total_tokens = sum(
            t.get("num_generated_tokens", 0) for t in retrieval_turns
        )
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
            "latency": {
                "question_latency_sec": round(question_latency_sec, 4),
                "mean_turn_generate_latency_sec": round(sum(turn_generate_latencies) / len(turn_generate_latencies), 4) if turn_generate_latencies else 0.0,
                "total_generated_tokens": total_tokens,
            },
        }

    def infer(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform inference on a question

        Args:
            question: Input question
            verbose: Whether to print intermediate outputs

        Returns:
            Dictionary containing:
                - full_response: Complete response with all search iterations
                - predicted_answer: Extracted answer
                - num_turns: Number of search iterations performed
                - retrieval_turns: List of retrieval information per turn
        """
        # Prepare question
        question = question.strip()
        if question[-1] != '?':
            question += '?'

        # Prepare prompt
        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        if verbose:
            print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
            print(prompt)

        full_response = ""
        cnt = 0
        retrieval_turns = []
        total_search_results = []
        last_search_results_list = []
        question_start = time.perf_counter()

        # Iterative search loop
        while cnt < self.max_turns:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate text with stopping criteria
            t0 = time.perf_counter()
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature
            )
            generate_latency = time.perf_counter() - t0
            num_generated_tokens = len(outputs[0]) - input_ids.shape[1]

            # Check if generation finished
            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_response += output_text
                if verbose:
                    print(output_text)
                retrieval_turns.append({
                    "turn": cnt,
                    "query": None,
                    "model_output": output_text,
                    "search_results": [],
                    "retrieved_docs": [],
                    "generate_latency_sec": round(generate_latency, 4),
                    "num_generated_tokens": num_generated_tokens,
                })
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract and perform search
            tmp_query = self._get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                search_start = time.perf_counter()
                last_search_results_list = self._search(tmp_query)
                search_latency = time.perf_counter() - search_start
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)

                # Track retrieval information for this turn
                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "model_output": output_text,
                    "search_results": last_search_results_list.copy(),
                    "retrieved_docs": [],
                    "generate_latency_sec": round(generate_latency, 4),
                    "search_latency_sec": round(search_latency, 4),
                    "num_generated_tokens": num_generated_tokens,
                }

                # Extract document titles from search results
                for result in last_search_results_list:
                    # Extract title from "(Title: {title}) {text}" format
                    if result.startswith("(Title: "):
                        title_end = result.find(")")
                        if title_end != -1:
                            title = result[8:title_end]  # Skip "(Title: " and get title
                            turn_info["retrieved_docs"].append(title)

                retrieval_turns.append(turn_info)

                search_results = "\n".join([f"Doc {idx+1}{result}"
                                          for idx, result in enumerate(last_search_results_list)])
            else:
                search_results = ''

            search_text = self.curr_search_template.format(
                output_text=output_text,
                search_results=search_results
            )
            prompt += search_text
            full_response += search_text
            cnt += 1

            if verbose:
                print(search_text)

        question_latency = time.perf_counter() - question_start

        if verbose:
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
            question_latency_sec=question_latency,
        )

    def infer_with_nudge(self, question: str, thinking: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform inference on a question

        Args:
            question: Input question
            verbose: Whether to print intermediate outputs

        Returns:
            Dictionary containing:
                - full_response: Complete response with all search iterations
                - predicted_answer: Extracted answer
                - num_turns: Number of search iterations performed
                - retrieval_turns: List of retrieval information per turn
        """
        # Prepare question
        question = question.strip()
        if question[-1] != '?':
            question += '?'

        # Prepare prompt
        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        prompt += (f"\n\n<think>{thinking}</think>\n\n" if thinking else "")

        if verbose:
            print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
            print(prompt)

        full_response = ""
        cnt = 0
        retrieval_turns = []
        total_search_results = []
        last_search_results_list = []
        question_start = time.perf_counter()

        # Iterative search loop
        while cnt < self.max_turns:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate text with stopping criteria
            t0 = time.perf_counter()
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature
            )
            generate_latency = time.perf_counter() - t0
            num_generated_tokens = len(outputs[0]) - input_ids.shape[1]

            # Check if generation finished
            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_response += output_text
                if verbose:
                    print(output_text)
                retrieval_turns.append({
                    "turn": cnt,
                    "query": None,
                    "model_output": output_text,
                    "search_results": [],
                    "retrieved_docs": [],
                    "generate_latency_sec": round(generate_latency, 4),
                    "num_generated_tokens": num_generated_tokens,
                })
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract and perform search
            tmp_query = self._get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                search_start = time.perf_counter()
                last_search_results_list = self._search(tmp_query)
                search_latency = time.perf_counter() - search_start
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)

                # Track retrieval information for this turn
                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "model_output": output_text,
                    "search_results": last_search_results_list.copy(),
                    "retrieved_docs": [],
                    "generate_latency_sec": round(generate_latency, 4),
                    "search_latency_sec": round(search_latency, 4),
                    "num_generated_tokens": num_generated_tokens,
                }

                # Extract document titles from search results
                for result in last_search_results_list:
                    # Extract title from "(Title: {title}) {text}" format
                    if result.startswith("(Title: "):
                        title_end = result.find(")")
                        if title_end != -1:
                            title = result[8:title_end]  # Skip "(Title: " and get title
                            turn_info["retrieved_docs"].append(title)

                retrieval_turns.append(turn_info)

                search_results = "\n".join([f"Doc {idx+1}{result}"
                                          for idx, result in enumerate(last_search_results_list)])
            else:
                search_results = ''

            search_text = self.curr_search_template.format(
                output_text=output_text,
                search_results=search_results
            )
            prompt += search_text
            full_response += search_text
            cnt += 1

            if verbose:
                print(search_text)

        question_latency = time.perf_counter() - question_start

        #print("Question:", question)
        print(prompt)
        # print("Full Response:", full_response)
        print("\n===\n")

        return self._build_result(
            question=question,
            prompt=prompt,
            full_response=full_response,
            cnt=cnt,
            retrieval_turns=retrieval_turns,
            total_search_results=total_search_results,
            last_search_results_list=last_search_results_list,
            question_latency_sec=question_latency,
            initial_thinking=thinking,
        )

    def infer_with_graph_hint(self, question: str, graph_hint: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform SearchR1 inference with a filled question graph as an answer hint.

        The graph is treated as a hint, not a replacement for retrieval.  SearchR1
        can still issue <search> queries and should answer inside <answer> tags.
        """
        question = question.strip()
        if question and question[-1] != '?':
            question += '?'

        graph_hint = (graph_hint or "").strip() or "(empty)"
        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
You are also given a filled question graph hint. Use it as a guide for the relationships to verify, but prefer retrieved evidence if the hint is incomplete or inconsistent. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

Filled question graph hint:
{graph_hint}

Question: {question}\n"""

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        if verbose:
            print('\n\n################# [Start Reasoning + Searching with Graph Hint] ##################\n\n')
            print(prompt)

        full_response = ""
        cnt = 0
        retrieval_turns = []
        total_search_results = []
        last_search_results_list = []

        while cnt < self.max_turns:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature
            )

            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_response += output_text
                if verbose:
                    print(output_text)
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tmp_query = self._get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                last_search_results_list = self._search(tmp_query)
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)

                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "retrieved_docs": [],
                }
                for result in last_search_results_list:
                    if result.startswith("(Title: "):
                        title_end = result.find(")")
                        if title_end != -1:
                            turn_info["retrieved_docs"].append(result[8:title_end])
                retrieval_turns.append(turn_info)

                search_results = "\n".join([f"Doc {idx+1}{result}"
                                          for idx, result in enumerate(last_search_results_list)])
            else:
                search_results = ''

            search_text = self.curr_search_template.format(
                output_text=output_text,
                search_results=search_results
            )
            prompt += search_text
            full_response += search_text
            cnt += 1

            if verbose:
                print(search_text)

        predicted_answer = self._extract_answer(full_response)
        reasoning_path = self._extract_reasoning(full_response)

        return {
            "full_response": full_response,
            "predicted_answer": predicted_answer,
            "reasoning_path": reasoning_path,
            "num_turns": cnt,
            "retrieval_turns": retrieval_turns,
            "total_search_results": total_search_results,
            "last_search_results_list": last_search_results_list
        }

    def infer_with_subgoal(
        self,
        question: str,
        current_graph_hint: str,
        target_triple: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform SearchR1 inference guided by one unresolved query-graph triple.

        This is used by the Q-guided online Veri-Graph path: the question graph
        chooses the current reasoning subgoal, and SearchR1 searches for
        evidence that can fill the UNKNOWN slot(s) in that selected triple.
        """
        question = question.strip()
        if question and question[-1] != '?':
            question += '?'

        current_graph_hint = (current_graph_hint or "").strip() or "(empty)"
        target_triple = (target_triple or "").strip() or "(none)"
        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
You are also given the current question graph and one selected unresolved query triple. Treat the selected triple as the current reasoning subgoal: focus your thinking and search query on finding evidence that can fill the UNKNOWN placeholder(s) in that triple. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

Current question graph state:
{current_graph_hint}

Selected unresolved query triple:
{target_triple}

Question: {question}\n"""

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        if verbose:
            print('\n\n################# [Start Q-Graph Subgoal Reasoning + Searching] ##################\n\n')
            print(prompt)

        full_response = ""
        cnt = 0
        retrieval_turns = []
        total_search_results = []
        last_search_results_list = []

        while cnt < self.max_turns:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature
            )

            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_response += output_text
                if verbose:
                    print(output_text)
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tmp_query = self._get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                last_search_results_list = self._search(tmp_query)
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)

                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "retrieved_docs": [],
                    "target_triple": target_triple,
                }
                for result in last_search_results_list:
                    if result.startswith("(Title: "):
                        title_end = result.find(")")
                        if title_end != -1:
                            turn_info["retrieved_docs"].append(result[8:title_end])
                retrieval_turns.append(turn_info)

                search_results = "\n".join([f"Doc {idx+1}{result}"
                                          for idx, result in enumerate(last_search_results_list)])
            else:
                search_results = ''

            search_text = self.curr_search_template.format(
                output_text=output_text,
                search_results=search_results
            )
            prompt += search_text
            full_response += search_text
            cnt += 1

            if verbose:
                print(search_text)

        predicted_answer = self._extract_answer(full_response)
        reasoning_path = self._extract_reasoning(full_response)

        return {
            "full_response": full_response,
            "predicted_answer": predicted_answer,
            "reasoning_path": reasoning_path,
            "num_turns": cnt,
            "retrieval_turns": retrieval_turns,
            "total_search_results": total_search_results,
            "last_search_results_list": last_search_results_list,
            "target_triple": target_triple,
        }

    def infer_with_observer(
        self,
        question: str,
        on_turn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        verbose: bool = False,
        max_turns_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform ordinary SearchR1 inference while exposing each search turn.

        Unlike infer_with_subgoal, the prompt contains only the original
        question.  The optional observer callback runs outside SearchR1 after a
        reasoning/search chunk and its retrieved documents are available.

        Callback return dict supports:
          - "stop": True to halt further generation
          - "abstain": True (paired with stop) marks abstain
          - "reason": string label
          - "prompt_injection": string appended to the prompt before the next
            generation pass (this is what makes the observer act as a
            reasoning *corrector*: it can feed verigraph-derived guidance back
            into SearchR1 between thinking steps without giving away the
            answer).

        max_turns_override lets the caller use a larger reasoning budget for
        verigraph-corrected runs without re-initialising the model.
        """
        question = question.strip()
        if question and question[-1] != '?':
            question += '?'

        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        if verbose:
            print('\n\n################# [Start Reasoning + Searching with Observer] ##################\n\n')
            print(prompt)

        full_response = ""
        cnt = 0
        retrieval_turns = []
        total_search_results = []
        last_search_results_list = []
        observer_events = []
        observer_stop_reason = ""
        observer_abstained = False
        max_turns_local = int(max_turns_override) if max_turns_override is not None else self.max_turns

        while cnt < max_turns_local:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature
            )

            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_response += output_text
                if verbose:
                    print(output_text)
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tmp_query = self._get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                last_search_results_list = self._search(tmp_query)
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)

                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "retrieved_docs": [],
                }
                for result in last_search_results_list:
                    if result.startswith("(Title: "):
                        title_end = result.find(")")
                        if title_end != -1:
                            turn_info["retrieved_docs"].append(result[8:title_end])
                retrieval_turns.append(turn_info)

                search_results = "\n".join([f"Doc {idx+1}{result}"
                                          for idx, result in enumerate(last_search_results_list)])
            else:
                search_results = ''

            search_text = self.curr_search_template.format(
                output_text=output_text,
                search_results=search_results
            )
            prompt += search_text
            full_response += search_text

            observer_action: Dict[str, Any] = {}
            if on_turn is not None:
                event = {
                    "turn": cnt,
                    "output_text": output_text,
                    "search_text": search_text,
                    "query": tmp_query or "",
                    "search_results": list(last_search_results_list or []),
                    "full_response": full_response,
                }
                observer_action = on_turn(event) or {}
                event["observer_action"] = dict(observer_action)
                observer_events.append(event)

            injection = str(observer_action.get("prompt_injection", "") or "")
            if injection:
                prompt += injection
                full_response += injection
                if verbose:
                    print(injection)

            cnt += 1

            if verbose:
                print(search_text)

            if observer_action.get("stop"):
                observer_stop_reason = str(observer_action.get("reason", "observer_stop") or "observer_stop")
                observer_abstained = bool(observer_action.get("abstain", False))
                break

        predicted_answer = self._extract_answer(full_response)
        reasoning_path = self._extract_reasoning(full_response)

        return {
            "full_response": full_response,
            "predicted_answer": predicted_answer,
            "reasoning_path": reasoning_path,
            "num_turns": cnt,
            "retrieval_turns": retrieval_turns,
            "total_search_results": total_search_results,
            "last_search_results_list": last_search_results_list,
            "observer_events": observer_events,
            "observer_stop_reason": observer_stop_reason,
            "observer_abstained": observer_abstained,
        }


# Example usage when run as script
if __name__ == "__main__":
    # Initialize inference class
    inferencer = SearchR1Inference()

    # Example question
    question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

    # Run inference
    result = inferencer.infer(question, verbose=True)

    print("\n\n################# [Results] ##################\n")
    print(f"Predicted Answer: {result['predicted_answer']}")
    print(f"Number of Turns: {result['num_turns']}")
