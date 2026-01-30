import transformers
import torch
import random
import numpy as np
import requests
import re
from typing import Optional, Dict, Any


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

        # Iterative search loop
        while cnt < self.max_turns:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate text with stopping criteria
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature
            )

            # Check if generation finished
            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_response += output_text
                if verbose:
                    print(output_text)
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract and perform search
            tmp_query = self._get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                last_search_results_list = self._search(tmp_query)
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)

                # Track retrieval information for this turn
                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "retrieved_docs": [],
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

        # Extract final answer and reasoning
        predicted_answer = self._extract_answer(full_response)
        reasoning_path = self._extract_reasoning(full_response)

        print("Question:", question)
        print("Full Response:", full_response)
        print("\n===\n")

        return {
            "full_response": full_response,
            "predicted_answer": predicted_answer,
            "reasoning_path": reasoning_path,
            "num_turns": cnt,
            "retrieval_turns": retrieval_turns,
            "total_search_results": total_search_results,
            "last_search_results_list": last_search_results_list
        }

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

        # Iterative search loop
        while cnt < self.max_turns:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate text with stopping criteria
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature
            )

            # Check if generation finished
            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_response += output_text
                if verbose:
                    print(output_text)
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract and perform search
            tmp_query = self._get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                last_search_results_list = self._search(tmp_query)
                for res in last_search_results_list:
                    if res not in total_search_results:
                        total_search_results.append(res)

                # Track retrieval information for this turn
                turn_info = {
                    "turn": cnt,
                    "query": tmp_query,
                    "retrieved_docs": [],
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

        # Extract final answer and reasoning
        predicted_answer = self._extract_answer(full_response)
        reasoning_path = self._extract_reasoning(full_response)

        #print("Question:", question)
        print(prompt)
        # print("Full Response:", full_response)
        print("\n===\n")

        return {
            "full_response": full_response,
            "predicted_answer": predicted_answer,
            "reasoning_path": reasoning_path,
            "num_turns": cnt,
            "retrieval_turns": retrieval_turns,
            "total_search_results": total_search_results,
            "last_search_results_list": last_search_results_list
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
