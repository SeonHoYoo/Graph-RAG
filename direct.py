import argparse
import json
import logging
import os
import random
import requests
import unicodedata
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import *

from model_library.base_model import BaseModel
from utils.metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths
from search_r1 import SearchR1Inference

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)
random.seed(26)


def parse_args(
) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--input_filename", type=str, 
        help="Input JSON filename"
    )    
    parser.add_argument("--direct_filename", type=str, default=None, 
        help="Output JSON filename for Direct answering results"
    )
    parser.add_argument("--base_model_name", type=str, default="google/flan-t5-xl", 
        help="Model used for Direct, text infilling, triple verification and strategy selection."
    )
    parser.add_argument("--setting", type=str, default="open-book",
        choices=["open-book", "open-book+gold", "gold"], 
        help="Retrieval setting mode - options: [open-book, open-book+gold, gold]"
    )
    parser.add_argument("--bm25_top_k", type=int, default=5,
        help="Number of top documents to retrieve using BM25"
    )
    parser.add_argument("--use_searchr1", action="store_true",
        help="Use SearchR1 model for retrieval"
    )
    parser.add_argument("--searchr1_top_k", type=int, default=3,
        help="Number of top documents to retrieve using SearchR1"
    )
    parser.add_argument("--searchr1_max_turns", type=int, default=3,
        help="Maximum number of turns for SearchR1 inference"
    )
    parser.add_argument("--use_total_search_results", action="store_true",
        help="Use total search results from all turns of SearchR1 for retrieval instead of only the last turn"
    )    
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve",
        help="URL for retrieval server"
    )

    return parser.parse_args()


class Direct:
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        self.dataset = args.dataset
        self.input_path = os.path.join("datasets", self.dataset, "claims", args.input_filename)

        self.base_model_name = args.base_model_name
        self.base_model = None

        self.setting = args.setting
        self.bm25_top_k = args.bm25_top_k
        self.use_searchr1 = args.use_searchr1
        self.searchr1_top_k = args.searchr1_top_k
        self.searchr1_max_turns = args.searchr1_max_turns   
        self.use_total_search_results = args.use_total_search_results
        self.retriever_url = args.retriever_url

        self.retriever = None
        if "open-book" in self.setting:            
            if self.use_searchr1:
                # Import and initialize SearchR1Inference                
                logger.info("Initializing SearchR1 inference model...")
                self.searchr1_model = SearchR1Inference(
                    retriever_url=self.retriever_url,
                    max_turns=self.searchr1_max_turns,
                    topk=self.searchr1_top_k
                )
            else:
                self.searchr1_model = None        
        else:
            self.searchr1_model = None
        self.evidence_max_len = 40000

        if args.direct_filename:
            direct_filename = args.direct_filename
        else:
            direct_filename = args.input_filename
        self.direct_path = os.path.join(
            "results", self.dataset, "answering", "direct", self.base_model_name.split("/")[-1], 
            self.setting, direct_filename
        )
    
    
    def check_retrieval(
        self, 
        doc_id: str, 
        gold_id_list: List[str]
    ) -> int:
        
        return 1 if doc_id in gold_id_list else 0
    
    
    def truncate(
        self, 
        text: str
    ) -> str:
        
        if len(text) > self.evidence_max_len:            
            text = text[:self.evidence_max_len]
            logger.warning(f"Evidence length exceeds {self.evidence_max_len} characters and has been truncated to prevent GPU memory overflow.")
        
        return text

    
    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = 10,
            max_words: Optional[int] = 500,
            use_searchr1: bool = False,
            use_total_search_results: bool = False,
            nudge_searchr1: bool = False,            
            thinking: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        if use_searchr1:
            # Use SearchR1Inference to get last search results
            try:
                # Run SearchR1 inference with the query
                if nudge_searchr1:
                    search_info = self.searchr1_model.infer_with_nudge(
                        query, thinking=thinking, verbose=False
                    )
                else:
                    search_info = self.searchr1_model.infer(query, verbose=False)
                
                if use_total_search_results:
                    search_results = search_info.get("total_search_results", [])
                else:
                    search_results = search_info.get("last_search_results_list", [])

                results = []
                for search_text in search_results:
                    # Parse the "(Title: {title}) {text}" format
                    if search_text.startswith("(Title: "):
                        title_end = search_text.find(")")
                        if title_end != -1:
                            title = search_text[8:title_end]  # Skip "(Title: "
                            text = search_text[title_end + 2:].strip()  # Skip ") "

                            title = unicodedata.normalize('NFC', title)

                            # Truncate by words if needed
                            words = text.split()
                            if max_words is not None and len(words) > max_words:
                                text = " ".join(words[:max_words])

                            results.append({
                                "doc_id": title,
                                "text": text,
                                "score": 1.0  # SearchR1 doesn't provide scores
                            })

                # If using total search results, don't limit by top_k (already limited by SearchR1's topk per turn)
                # Otherwise, limit to top_k
                if use_total_search_results:
                    return results, search_info
                else:
                    return results[:top_k], search_info

            except Exception as e:
                logger.error(f"SearchR1 retrieval error: {e}")
                return [], {}

        else:
            # Directly use retrieval server
            payload = {
                "queries": [query],
                "topk": top_k,
                "return_scores": True
            }
            try:
                response = requests.post(self.retriever_url, json=payload, timeout=30)
                response.raise_for_status()
                retrieval_results = response.json()['result'][0]

                results = []
                for doc_item in retrieval_results:
                    content = doc_item['document']['contents']
                    # Extract title from first line
                    lines = content.split("\n")
                    
                    title = lines[0] if lines else ""
                    title = title.strip('"\'')
                    title = unicodedata.normalize('NFC', title)
                    
                    text = "\n".join(lines[1:]) if len(lines) > 1 else ""

                    # Truncate by words if needed
                    words = text.split()
                    if max_words is not None and len(words) > max_words:
                        text = " ".join(words[:max_words])

                    results.append({
                        "doc_id": title,
                        "text": text,
                        "score": doc_item.get('score', 0.0)
                    })

                return results, {}

            except Exception as e:
                logger.error(f"SearchR1 retrieval error: {e}")
                return [], {}

    def retrieve_evidence(
        self,
        query: str,
        gold_id_list: List[str],
        gold_evidence_list: List[str],
        top_k: int = 10,
        use_searchr1: bool = False,
        use_total_search_results: bool = False,
        nudge_searchr1: bool = False,
        thinking: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        
        doc_id_list, is_gold_list, evidence_list = [], [], []
        gold_id_list = [unicodedata.normalize('NFC', ti).strip() for ti in gold_id_list]
        gold_evidence_list = [unicodedata.normalize('NFC', te).strip() for te in gold_evidence_list]

        if "gold" in self.setting:
            doc_id_list.extend(gold_id_list)
            is_gold_list.extend([1] * len(gold_id_list))
            for doc_id, text in zip(gold_id_list, gold_evidence_list):
                evidence_list.append(f"(Title: {doc_id}) {text}")

        if "open-book" in self.setting:
            hit_list, search_info = self.retrieve(
                query, 
                top_k, 
                use_searchr1=use_searchr1, 
                use_total_search_results=use_total_search_results,
                nudge_searchr1=nudge_searchr1, 
                thinking=thinking, 
            )

            for hit in hit_list:
                # Only apply top_k limit when NOT using total search results
                if not use_total_search_results and len(doc_id_list) >= top_k:
                    break

                doc_id = unicodedata.normalize('NFC', hit["doc_id"]).strip()
                text = unicodedata.normalize('NFC', hit["text"]).strip()
                if text and f"(Title: {doc_id}) {text}" not in evidence_list:
                    doc_id_list.append(doc_id)
                    is_gold_list.append(self.check_retrieval(doc_id, gold_id_list))
                    evidence_list.append(f"(Title: {doc_id}) {text}")


        retrieval_info = {
            "query": query,
            "doc_id_list": doc_id_list,
            "is_gold_list": is_gold_list
        }

        if "open-book" in self.setting and use_searchr1:            
            retrieval_info["full_response"] = search_info.get("full_response", "")
            retrieval_info["searchr1_answer"] = search_info.get("predicted_answer", "")
            retrieval_info["num_turns"] = search_info.get("num_turns", 0)
            retrieval_info["retrieval_turns"] = search_info.get("retrieval_turns", [])

        return evidence_list, retrieval_info


    def verify_claim(
        self,
        claim: str,
        gold_id_list: List[str],
        gold_evidence_list: List[str],
        top_k: Optional[int] = 10,
        document_level: Optional[str] = "concat",
        retrieval_result: Optional[Tuple[List[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        if retrieval_result is None:
            evidence_list, retrieval_info = self.retrieve_evidence(
                claim, gold_id_list, gold_evidence_list, top_k
            )
        else:
            evidence_list, retrieval_info = retrieval_result
            
        answer, supported_doc_id = False, None
        conf_list = []
        
        if document_level == "concat":
            evidence_concat = self.truncate('\n'.join(evidence_list))
            answer, conf = self.base_model.verify(claim, evidence_concat)
            conf_list.append(conf)
            if answer:
                supported_doc_id = "[concatenated_documents]"
            
        elif document_level == "each":
            
            for idx, evidence in enumerate(evidence_list):
                evidence = self.truncate(evidence)
                answer, conf = self.base_model.verify(claim, evidence)
                conf_list.append(conf)
                if answer:
                    supported_doc_id = retrieval_info["doc_id_list"][idx]
                    break

        elif document_level == "concat+each":
            evidence_concat = self.truncate('\n'.join(evidence_list))
            answer, conf = self.base_model.verify(claim, evidence_concat)
            conf_list.append(conf)
            if answer:
                supported_doc_id = "[concatenated_documents]"
            else:
                for idx, evidence in enumerate(evidence_list):
                    evidence = self.truncate(evidence)
                    answer, conf = self.base_model.verify(claim, evidence)
                    conf_list.append(conf)
                    if answer:
                        supported_doc_id = retrieval_info["doc_id_list"][idx]
                        break
        
        prediction = "SUPPORTED" if answer else "NOT_SUPPORTED"
        avg_conf = sum(conf_list) / len(conf_list) if conf_list else None
        
        return {
            "prediction": prediction,
            "average_verification_confidence": avg_conf,
            "last_verification_confidence": conf,
            "retrieval_info": retrieval_info,
            "supported_doc_id": supported_doc_id,
        }


    def get_answer(
        self, 
        question: str, 
        evidence: Optional[str] = None,
        **generator_args
    ) -> Tuple[str, str]:
        
        evidence_concat = self.truncate(evidence)        
        answer, conf = self.base_model.generate_answer(question, evidence_concat, **generator_args)
        
        return answer, conf

    
    def process_sample_direct(
        self,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:

        try:
            if self.use_searchr1:
                
                evidence_list, retrieval_info = self.retrieve_evidence(
                    sample["question"], 
                    sample["gold_id_list"], 
                    sample["gold_evidence_list"], 
                    self.searchr1_top_k, 
                    use_searchr1=self.use_searchr1,
                    use_total_search_results=self.use_total_search_results
                )
            else:
                evidence_list, retrieval_info = self.retrieve_evidence(
                    sample["question"], sample["gold_id_list"], sample["gold_evidence_list"], self.bm25_top_k
                )

            evidence = '\n'.join(evidence_list)
            answer, conf = self.get_answer(sample["question"], evidence)

            # Compute evaluation metrics
            ground_truth_answers = [sample["answer"]] + sample.get("answer_aliases", [])
            em_score = metric_max_over_ground_truths(compute_exact, answer, ground_truth_answers)
            f1_score = metric_max_over_ground_truths(compute_f1, answer, ground_truth_answers)

            sample.update({
                "predicted_answer": answer,
                "answering_confidence": conf,
                "em_score": em_score,
                "f1_score": f1_score,
                "retrieval_info": {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None,
            })

        except Exception as e:
            logger.warning(f"Failed to generate answer. Skipping this sample. Error: {e}")

            sample.update({
                "predicted_answer": "",
                "answering_confidence": None,
                "em_score": 0.0,
                "f1_score": 0.0,    
                "retrieval_info": None,
            })

        return sample
    
    
    def run_direct(
        self,
        input_indices: Set[int] = None
    ) -> List[Dict[str, Any]]:
        
        with open(self.input_path, "r") as f:
            input_list = json.load(f)

        if input_indices:
            input_list = [sample for sample in input_list if sample["index"] in input_indices]

        logger.info("Starting Direct answering...")

        if self.base_model is None:
            tokenizer = T5Tokenizer.from_pretrained(self.base_model_name)
            model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, device_map="auto")
            self.base_model = BaseModel(model, tokenizer)
            logger.info(f"Base model '{self.base_model_name}' initialized successfully.")        

        result_list = []
        for sample in tqdm(input_list):
            result = self.process_sample_direct(sample)
            result_list.append(result)
                
        logger.info("Direct answering completed!")
        
        return result_list

    
    def save_result(
        self,
        result_list: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(result_list, f, indent=4)
        logger.info(f"Results saved at '{output_path}'.")
    
    
    def has_complete_results(
        self,
        file_path: str,
        input_indices: Optional[Set[int]] = None
    ) -> bool:
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                result_list = json.load(f)
            result_indices = {sample["index"] for sample in result_list}

            if not input_indices:
                with open(self.input_path, "r") as f:
                    input_list = json.load(f)
                input_indices = {sample["index"] for sample in input_list}
            
            missing_indices = input_indices - result_indices

            if not missing_indices:
                return True
        
        return False
    

if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        logger.info(f"- {key}: {value}")
    
    verifier = Direct(args)
    result_list = verifier.run_direct()
    verifier.save_result(result_list, verifier.direct_path)        

