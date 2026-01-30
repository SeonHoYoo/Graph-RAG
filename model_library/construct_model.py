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

from model_library.prompt import construction_prompt, latent_detection_prompt_musique, latent_detection_prompt_hotpotqa, latent_detection_prompt_2wikimultihopqa, triplet_extraction_prompt_musique, triplet_extraction_prompt_hotpotqa, triplet_extraction_prompt_2wikimultihopqa, cot_reasoning_triplet_extraction_prompt, document_triplet_extraction_prompt, cot_reasoning_generation_prompt, cot_reasoning_with_triplets_prompt
from model_library.llm_clients import GPT, Claude, Qwen

logger = logging.getLogger(__name__)


class ConstructModel():
    def __init__(
        self, 
        construct_model_name: str,
        dataset_name: str,
        api_key: Optional[Any] = None, 
        batch_size: Optional[int] = 1,
    ):  
        
        if construct_model_name.lower().startswith("gpt"):
            client = OpenAI(api_key=api_key)
            self.construct_model = GPT(construct_model_name, client)
        
        elif construct_model_name.lower().startswith("claude"):
            client = anthropic.Anthropic(api_key=api_key)
            self.construct_model = Claude(construct_model_name, client)
        
        elif construct_model_name.lower().startswith("qwen"):
            model = AutoModelForCausalLM.from_pretrained(
                construct_model_name,
                dtype="auto",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(construct_model_name)
            self.construct_model = Qwen(model, tokenizer)

        self.batch_size = batch_size        
        
        dataset_name = dataset_name.lower()
        self.construction_prompt = construction_prompt
        if dataset_name == "musique":
            self.latent_detection_prompt = latent_detection_prompt_musique
            self.triplet_extraction_prompt = triplet_extraction_prompt_musique
        elif dataset_name == "hotpotqa":
            self.latent_detection_prompt = latent_detection_prompt_hotpotqa
            self.triplet_extraction_prompt = triplet_extraction_prompt_hotpotqa
        elif dataset_name == "2wikimultihopqa":
            self.latent_detection_prompt = latent_detection_prompt_2wikimultihopqa
            self.triplet_extraction_prompt = triplet_extraction_prompt_2wikimultihopqa
    
    
    def parse_graph(
        self, 
        generated_text: str
    ) -> Tuple[List[str], List[str]]:
        
        first_section, second_section = [], []
        flag = 0
        
        lines = [line.strip() for line in generated_text.split("\n")]
        for line in lines:
            if not line:
                continue
            if line.startswith("# Question"):
                break
            if "no latent entities identified" in line.lower():
                continue
            if "(no latent entities needed)" in line.lower():
                continue
            if line.lower().strip() == "none":
                continue
            if line.startswith("# Latent Entities"):
                continue
            if line.startswith("# Triples"):
                flag = 1
                continue
            if "[SEP]" not in line:
                continue
            if not line.startswith("(ENT"):
                flag = 1
            
            if flag == 0:
                first_section.append(line)
            elif flag == 1:
                second_section.append(line)
        
        def_triples = []
        for idx, line in enumerate(first_section.copy()):
            expected_prefix = f"(ENT{idx+1}) [SEP] is [SEP]"
            if line.startswith(expected_prefix):
                def_triples.append(line)
                first_section.remove(line)
        
        triples = first_section + second_section
        
        return def_triples, triples
    
    
    def parse_latent_entities(
        self, 
        generated_text: str
    ) -> List[str]:
        
        latent_entities = []
        lines = [line.strip() for line in generated_text.split("\n")]
        for line in lines:
            if not line:
                continue
            if line.startswith("# Question"):
                break
            if "no latent entities identified" in line.lower():
                continue
            if "(no latent entities needed)" in line.lower():
                continue
            if line.lower().strip() == "none":
                continue
            if line.startswith("# Latent Entities"):
                continue
            if "[SEP]" not in line:
                continue
            # 정의 triplets만 추출: (ENTX) [SEP] is [SEP] 형태만
            if line.startswith("(ENT") and " [SEP] is [SEP] " in line:
                latent_entities.append(line)
        
        return latent_entities


    def parse_triplets(
        self, 
        generated_text: str,
        def_triples: Optional[List[str]] = None
    ) -> List[str]:
        
        triplets = []
        lines = [line.strip() for line in generated_text.split("\n")]
        for line in lines:
            if not line:
                continue
            if line.startswith("# Question"):
                break
            if line.startswith("# Triples"):
                continue
            if "[SEP]" not in line:
                continue
            if line not in def_triples and line not in triplets:
                triplets.append(line)
        
        return triplets

    
    def process_sample(
        self,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        # prompt = self.construction_prompt.replace("<<target_question>>", sample["question"])
        # answer = self.construct_model.generate(prompt)
        
        # def_triples, triples = self.parse_graph(answer)

        # Updated to two-step generation for latent entity detection and triplet extraction
        latent_prompt = self.latent_detection_prompt.replace("<<target_question>>", sample["question"])
        latent_answer = self.construct_model.generate(latent_prompt)
        def_triples = self.parse_latent_entities(latent_answer)

        latent_entities_str = "\n".join(def_triples) if def_triples else "(no latent entities)"
        triplet_prompt = self.triplet_extraction_prompt.replace("<<target_question>>", sample["question"])
        triplet_prompt = triplet_prompt.replace("<<target_latent_entities>>", latent_entities_str)
        triplet_answer = self.construct_model.generate(triplet_prompt)
        triples = self.parse_triplets(triplet_answer, def_triples)
        
        sample.update({
            "definition_triples": def_triples,
            "triples": triples
        })
        
        return sample
    
    
    # TODO: Update to two-step processing
    def process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        batch_prompt = [
            self.construction_prompt.replace("<<target_question>>", sample["question"]) for sample in batch
        ]
        batch_answer = self.construct_model.batch_generate(batch_prompt)
        
        for sample, answer in zip(batch, batch_answer):
            def_triples, triples = self.parse_graph(answer)
            sample.update({
                "definition_triples": def_triples,
                "triples": triples
            })
        
        return batch
    
    
    def construct_graph(
        self,
        input_list: List[Dict[str, Any]],
        graph_path: str,
        force_new_construction: Optional[bool] = False
    ) -> List[Dict[str, Any]]:
        
        graph_list = []
        if os.path.exists(graph_path) and not force_new_construction:
            with open(graph_path, "r") as f:
                graph_list = json.load(f)
                
        input_indices = {sample["index"] for sample in input_list}
            
        existing_count = len([sample for sample in graph_list if sample["index"] in input_indices])
        total_count = len(input_list)
        
        if existing_count > 0:
            remaining_count = total_count - existing_count
            logger.info(f"{existing_count}/{total_count} samples already have graphs. Constructing graphs for remaining {remaining_count} samples...")
            
            existing_indices = {sample["index"] for sample in graph_list}
            input_list = [sample for sample in input_list if sample["index"] not in existing_indices]
        
        else:
            remaining_count = total_count
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            logger.info(f"Constructing graphs for {len(input_list)} samples...")
        
        new_count = 0

        # Use OpenAI Batch API
        if self.batch_size > 1 and isinstance(self.construct_model, GPT): 
            for i in tqdm(range(0, len(input_list), self.batch_size)):
                try:
                    batch = input_list[i : i + self.batch_size]
                    batch_graph = self.process_batch(batch)
                    graph_list.extend(batch_graph)
                    
                    new_count += len(batch_graph)
                    if new_count % (self.batch_size * 5) == 0 and new_count < remaining_count:
                        graph_list = sorted(graph_list, key=lambda x: x["index"])
                        with open(graph_path, "w") as f:
                            json.dump(graph_list, f, indent=4)
                            logger.info(f"Checkpoint: {new_count} new graphs saved at '{graph_path}'")
                    
                except Exception:
                    logger.warning("Failed to construct graphs. Skipping this batch.")
        else:
            for sample in tqdm(input_list):
                try:
                    graph = self.process_sample(sample)
                    graph_list.append(graph)
                    
                    new_count += 1
                    if new_count % 10 == 0 and new_count < remaining_count:
                        graph_list = sorted(graph_list, key=lambda x: x["index"])
                        with open(graph_path, "w") as f:
                            json.dump(graph_list, f, indent=4)
                            logger.info(f"Checkpoint: {new_count} new graphs saved at '{graph_path}'")
                
                except Exception:
                    logger.warning("Failed to construct a graph. Skipping this sample.")
        
        logger.info(f"Graph construction completed!")
        
        graph_list = sorted(graph_list, key=lambda x: x["index"])
        with open(graph_path, "w") as f:
            json.dump(graph_list, f, indent=4)
            logger.info(f"Graphs saved at '{graph_path}'")
        
        graph_list = [sample for sample in graph_list if sample["index"] in input_indices]

        if isinstance(self.construct_model, Qwen):
            try:
                del self.construct_model
                torch.cuda.empty_cache()
                logger.info("Released construct_model from GPU memory.")
            except Exception as e:
                logger.warning(f"Failed to release construct_model from memory: {e}")
        
        return graph_list
    
    
    def extract_triplets_from_cot_reasoning(
        self,
        reasoning_path: str
    ) -> Tuple[List[str], List[str]]:
        """
        CoT reasoning path에서 triplet을 추출합니다.
        
        Args:
            reasoning_path: Chain-of-Thought reasoning 텍스트
            
        Returns:
            (def_triples, triples): definition triples와 일반 triples 리스트
        """
        prompt = cot_reasoning_triplet_extraction_prompt.replace("<<target_reasoning_path>>", reasoning_path)
        answer = self.construct_model.generate(prompt)
        
        # latent entities가 있는지 확인
        def_triples = self.parse_latent_entities(answer)
        triples = self.parse_triplets(answer, def_triples)
        
        return def_triples, triples
    
    
    def extract_triplets_from_document(
        self,
        document: str
    ) -> Tuple[List[str], List[str]]:
        """
        Retrieved document에서 triplet을 추출합니다.
        
        Args:
            document: Retrieved document 텍스트
            
        Returns:
            (def_triples, triples): definition triples와 일반 triples 리스트
        """
        prompt = document_triplet_extraction_prompt.replace("<<target_document>>", document)
        answer = self.construct_model.generate(prompt)
        
        # documents에서는 보통 latent entities가 없으므로 빈 리스트
        def_triples = []
        triples = self.parse_triplets(answer, def_triples)
        
        return def_triples, triples
    
    
    def generate_cot_reasoning(
        self,
        question: str
    ) -> str:
        """
        질문에 대한 CoT reasoning을 생성합니다.
        
        Args:
            question: 질문 텍스트
            
        Returns:
            CoT reasoning 텍스트
        """
        prompt = cot_reasoning_generation_prompt.replace("<<target_question>>", question)
        reasoning = self.construct_model.generate(prompt)
        
        # 프롬프트에서 "# Reasoning:" 이후 부분만 추출
        if "# Reasoning:" in reasoning:
            reasoning = reasoning.split("# Reasoning:")[-1].strip()
        elif "Reasoning:" in reasoning:
            reasoning = reasoning.split("Reasoning:")[-1].strip()
        
        return reasoning
    
    
    def generate_cot_reasoning_with_triplets(
        self,
        question: str
    ) -> Tuple[str, List[str], List[str]]:
        """
        질문에서 CoT reasoning과 triplet을 한 번에 추출합니다 (더 효율적).
        
        Args:
            question: 질문 텍스트
            
        Returns:
            (reasoning, def_triples, triples): CoT reasoning과 triplet 리스트
        """
        prompt = cot_reasoning_with_triplets_prompt.replace("<<target_question>>", question)
        answer = self.construct_model.generate(prompt)
        
        # Reasoning 추출 - 여러 패턴 시도
        reasoning = ""
        
        # 패턴 1: "# Reasoning:" 구분자 사용
        if "# Reasoning:" in answer:
            reasoning_part = answer.split("# Reasoning:")[-1]
            if "# Latent Entities:" in reasoning_part:
                reasoning = reasoning_part.split("# Latent Entities:")[0].strip()
            elif "# Triples:" in reasoning_part:
                reasoning = reasoning_part.split("# Triples:")[0].strip()
            else:
                reasoning = reasoning_part.strip()
        
        # 패턴 2: "Reasoning:" 구분자 사용 (앵글 브래킷 없음)
        elif "Reasoning:" in answer and not reasoning:
            reasoning_part = answer.split("Reasoning:")[-1]
            if "# Latent Entities:" in reasoning_part:
                reasoning = reasoning_part.split("# Latent Entities:")[0].strip()
            elif "# Triples:" in reasoning_part:
                reasoning = reasoning_part.split("# Triples:")[0].strip()
            else:
                reasoning = reasoning_part.strip()
        
        # 패턴 3: "# Question:" 이전 부분이 reasoning일 수 있음
        if not reasoning or not reasoning.strip():
            if "# Question:" in answer:
                before_question = answer.split("# Question:")[0].strip()
                # 프롬프트 텍스트 제거
                if "Given a question" in before_question:
                    # 프롬프트 부분 제거
                    if "# Output Format:" in before_question:
                        before_question = before_question.split("# Output Format:")[-1].strip()
                if before_question and len(before_question) > 20:  # 의미있는 길이
                    reasoning = before_question
        
        # 패턴 4: "# Latent Entities:" 이전 부분
        if not reasoning or not reasoning.strip():
            if "# Latent Entities:" in answer:
                before_entities = answer.split("# Latent Entities:")[0].strip()
                # "# Question:" 이후 부분만 추출
                if "# Question:" in before_entities:
                    after_question = before_entities.split("# Question:")[-1]
                    if "Reasoning:" in after_question:
                        reasoning = after_question.split("Reasoning:")[-1].strip()
                    elif len(after_question.strip()) > 20:
                        reasoning = after_question.strip()
        
        # 패턴 5: "# Triples:" 이전 부분
        if not reasoning or not reasoning.strip():
            if "# Triples:" in answer:
                before_triples = answer.split("# Triples:")[0].strip()
                if "# Question:" in before_triples:
                    after_question = before_triples.split("# Question:")[-1]
                    if "Reasoning:" in after_question:
                        reasoning = after_question.split("Reasoning:")[-1].strip()
                    elif "# Latent Entities:" in after_question:
                        reasoning = after_question.split("# Latent Entities:")[0].strip()
                    elif len(after_question.strip()) > 20:
                        reasoning = after_question.strip()
        
        # 패턴 6: 모델이 다른 형식으로 출력한 경우 - 첫 번째 의미있는 문단 추출
        if not reasoning or not reasoning.strip():
            lines = answer.split("\n")
            for line in lines:
                line = line.strip()
                # 프롬프트 관련 라인 제외
                if line and not line.startswith("#") and "Question:" not in line and "Output Format:" not in line:
                    if len(line) > 30:  # 의미있는 길이
                        reasoning = line
                        break
        
        # 최종 검증: reasoning이 너무 짧거나 의미없으면 비움
        if reasoning and len(reasoning.strip()) < 20:
            reasoning = ""
        
        if not reasoning or not reasoning.strip():
            logger.warning(f"Failed to extract reasoning from model output. Output length: {len(answer)}")
            # reasoning 추출 실패 시 모델 출력의 일부를 로깅 (디버깅용)
            logger.warning(f"Model output preview (first 500 chars): {answer[:500]}")
            if len(answer) > 500:
                logger.warning(f"Model output preview (last 500 chars): {answer[-500:]}")
        
        # Triplet 추출
        def_triples = self.parse_latent_entities(answer)
        triples = self.parse_triplets(answer, def_triples)
        
        return reasoning, def_triples, triples