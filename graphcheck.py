import argparse
import json
import logging
import signal
import os
import random
import re
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import *

from direct import Direct
from model_library.base_model import BaseModel
from model_library.construct_model import ConstructModel
from model_library.nudge_model import NudgeModel
from utils.graph import Graph
from utils.metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths

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
        help="Output JSON filename for Direct verification results"
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
    parser.add_argument("--graph_filename", type=str, default=None, 
        help="Output JSON filename for graph construction results"
    )    
    parser.add_argument("--graphcheck_filename", type=str, default=None, 
        help="Output JSON filename for GraphCheck verification results"
    )
    parser.add_argument("--force_new_construction", action="store_true",
        help="Force new graph construction even if the graph file is already available"
    )
    parser.add_argument("--construct_model_name", type=str, default="gpt-4o-2024-08-06", 
        help=(
            "Model version used for graph construction.\n"
            "Recommended options: [gpt-4o-2024-08-06, claude-3-7-sonnet-20250219, Qwen/Qwen2.5-72B-Instruct]"
        )
    )
    parser.add_argument("--api_key", type=str, default=None,
        help="API key required for using OpenAI (GPT) or Anthropic (Claude) models during graph construction."
    )
    parser.add_argument("--construct_batch_size", type=int, default=1,
        help="Batch size for graph construction. Set greater than 1 to use OpenAI Batch APIs for cost-efficient processing (up to 50% cheaper but slower generation)."
    )
    parser.add_argument("--path_limit", type=int, default=5, 
        help="Maximum number of identification paths"
    )
    parser.add_argument("--document_level", type=str, default="concat",
        choices=["concat", "each", "concat+each"], 
        help="Document-level setting mode - options: [concat, each, concat+each]"
    )    
    parser.add_argument("--nudge_searchr1", action="store_true",
        help="Use SearchR1 model for retrieval with nudge thinking"
    )
    parser.add_argument("--nudge_model_name", type=str, default="gpt-4o-2024-08-06",
        help="Model used for generating initial thinking for SearchR1"
    )
    parser.add_argument("--nudge_filename", type=str, default=None, 
        help="Output JSON filename for nudge model results"
    )

    return parser.parse_args()


class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Function took more than 5 minutes!")


class GraphCheck(Direct):
    def __init__(
        self, 
        args: argparse.Namespace,
    ):
        super().__init__(args)
        
        self.force_new_construction = args.force_new_construction
        self.construct_model_name = args.construct_model_name
        self.api_key = args.api_key
        self.construct_batch_size = args.construct_batch_size

        self.path_limit = args.path_limit
        self.document_level = args.document_level

        self.nudge_searchr1 = args.nudge_searchr1
        
        if args.graph_filename:
            graph_filename = args.graph_filename
        else:
            graph_filename = args.input_filename
        self.graph_path = os.path.join(
            "results", self.dataset, "graphs", self.construct_model_name.split("/")[-1], graph_filename
        )
                
        if args.graphcheck_filename:
            graphcheck_filename = args.graphcheck_filename
        else:
            graphcheck_filename = args.input_filename
        
        if self.use_total_search_results:
            graphcheck_filename = graphcheck_filename.replace(".json", "_total_search_results.json")

        self.graphcheck_path = os.path.join(
            "results", self.dataset, "answering", "graphcheck", self.construct_model_name.split("/")[-1], self.base_model_name.split("/")[-1], 
            self.setting, f"top_{str(self.bm25_top_k)}", graphcheck_filename
        )

        if args.nudge_filename:
            nudge_filename = args.nudge_filename
        else:
            nudge_filename = graph_filename
        self.nudge_path = os.path.join(
            "results", self.dataset, "thinking", args.nudge_model_name.split("/")[-1], nudge_filename
        )

        if self.nudge_searchr1:
            self.nudge_model = NudgeModel(
                model_name=args.nudge_model_name,
                output_path=self.nudge_path,
                api_key=args.api_key,
            )
        else:
            self.nudge_model = None


    def get_infilling_retrieval_query(
        self, 
        graph: Graph, 
        target_la_ent: str,
        use_searchr1: Optional[bool] = None
    ) -> str:
        """
        Construct a retrieval query for latent entity infilling.
        
        The retrieval query is formed by concatenating all triples (except the definition triple) 
        that include the target latent entity and exclude any unidentified latent entities. 
        Each latent entity in the query is replaced with its corresponding reference mapped in the definition triples, 
        forming a nearly complete sentence.
        
        ----------
        Example:
        
        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] a musician
        (ENT2) [SEP] is [SEP] a band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington
        
        [Infilling retrieval query for (ENT1)]
        a musician is part of Tall Birds. 
        """

        sub_graph = [f"{triple.sentence}." for triple in graph.la_ent_2_sub_triples[target_la_ent]]
        query = " ".join(
            [triple_sent for triple_sent in sub_graph if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {target_la_ent}]
        )
        
        # Handle edge case where no relevant triples exist
        if query == "":
            query = f"{graph.la_ent_2_def_triple[target_la_ent].sentence}."
            
        if use_searchr1:        
            replaced = set()
            while re.search(r"\(ENT\d+\)", query):
                for la_ent, definition in graph.la_ent_2_def.items():                
                    if la_ent in replaced:
                        definition = definition.replace("a ", "the ", 1).replace("an ", "the ", 1)

                    query = query.replace(la_ent, definition, 1)
                    replaced.add(la_ent)

                    definition = definition.replace("a ", "the ", 1).replace("an ", "the ", 1)
                    query = query.replace(la_ent, definition)                
                if graph.has_la_ent_w_no_def == 1: # Edge case
                    break
            
            target_def = graph.la_ent_2_def[target_la_ent].replace("a ", "the ", 1).replace("an ", "the ", 1)
            query += f" What is {target_def}?"
            query = re.sub(r'\b(.+?)\s+is\s+\1\b.', '', query).strip()

            return query
        else:
            while re.search(r"\(ENT\d+\)", query):
                for la_ent, definition in graph.la_ent_2_def.items():
                    query = query.replace(la_ent, definition)
                if graph.has_la_ent_w_no_def == 1: # Edge case
                    break
            
            return query
    
    
    def get_infilling_query(
        self, 
        graph: Graph, 
        target_la_ent: str
    ) -> str:
        """
        Construct an infilling query for latent entity infilling.
        
        The infilling query is formed by concatenating all triples that include the target latent entity exclude any other unidentified latent entities. 
        The target latent entity is replaced with the special token to indicate that it should be infilled.
        
        ----------
        Example:
        
        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] a musician
        (ENT2) [SEP] is [SEP] a band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington
        
        [Infilling query for (ENT1)]
        <extra_id_0> is part of Tall Birds. <extra_id_0> is a musician.        
        """
        
        sub_graph = [f"{triple.sentence}." for triple in graph.la_ent_2_sub_triples[target_la_ent]]
        sub_graph.append(f"{graph.la_ent_2_def_triple[target_la_ent].sentence}.")

        query = " ".join(
            [triple_sent for triple_sent in sub_graph if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {target_la_ent}]
        )

        # Handle edge case where no relevant triples exist
        if query == "":
            query = f"{graph.la_ent_2_def_triple[target_la_ent].sentence}."
        
        variable_name = "<extra_id_0>"
        query = query.strip().replace(target_la_ent, variable_name)
        
        while re.search(r"\(ENT\d+\)", query):
            for la_ent, definition in graph.la_ent_2_def.items():
                query = query.replace(la_ent, definition)
            if graph.has_la_ent_w_no_def == 1: # Edge case
                break                   
        
        return query
    
    
    def infill_graph(
        self, 
        graph: Graph, 
        path: List[str], 
        gold_id_list: List[str], 
        gold_evidence_list: List[str],
        use_searchr1: Optional[bool] = None,
        use_total_search_results: bool = False,
        nudge_searchr1: bool = False,            
        searchr1_evidence_list: Optional[List[str]] = None
    ) -> Graph:
        
        infilled_def_triple_sents = [def_triple.triple_sent for def_triple in graph.def_triples]
        infilled_triple_sents = [triple.triple_sent for triple in graph.triples]
        infilling_log = []

        for i, target_la_ent in enumerate(path):
            conf_list = []
            retrieval_query = self.get_infilling_retrieval_query(graph, target_la_ent, use_searchr1=use_searchr1)
            
            if use_searchr1 and nudge_searchr1 and searchr1_evidence_list is not None:
                evidence_list = searchr1_evidence_list
                retrieval_info = None
            elif use_searchr1:
                evidence_list, retrieval_info = self.retrieve_evidence(
                    retrieval_query, 
                    gold_id_list, 
                    gold_evidence_list, 
                    self.searchr1_top_k, 
                    use_searchr1=use_searchr1,
                    use_total_search_results=use_total_search_results,
                )
            else:
                evidence_list, retrieval_info = self.retrieve_evidence(
                    retrieval_query, gold_id_list, gold_evidence_list, self.bm25_top_k
                )
            
            evidence = self.truncate('\n'.join(evidence_list))    
            
            infilling_query = self.get_infilling_query(graph, target_la_ent)
            answer, conf = self.base_model.infill(infilling_query, evidence)
            conf_list.append(conf)
            
            if not answer.strip():
                logger.error(f"No answer generated for {target_la_ent} in path {path}.")
                answer = graph.la_ent_2_def[target_la_ent]
            else:
                answer = answer.split("\n")[0].strip()
            
            infilled_def_triple_sents = [
                sent.replace(target_la_ent, answer) for sent in infilled_def_triple_sents
            ]
            infilled_triple_sents = [
                sent.replace(target_la_ent, answer) for sent in infilled_triple_sents
            ]
            remained_def_triple_sents = [
                sent for sent in infilled_def_triple_sents if re.search(r"\(ENT\d+\)", sent.split()[0])
            ]
            if remained_def_triple_sents:
                graph = Graph(remained_def_triple_sents, infilled_triple_sents)
            infilling_log.append({
                "infilling_index": i, 
                "target_latent_entity": target_la_ent, 
                "infilling_query": infilling_query, 
                "infilling_answer": answer, 
                "infilling_confidence": conf,
                "infilling_retrieval_info": {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None
            })
        
        avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0

        return Graph(infilled_def_triple_sents, infilled_triple_sents), infilling_log, avg_conf


    def verify_graph(
        self, 
        graph: Graph, 
        gold_id_list: List[str], 
        gold_evidence_list: List[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        
        path_prediction = "SUPPORTED"
        path_verification = []
        num_supported_subclaims = 0
        # subclaim_prediction, retrieval_info, supported_doc_id
        for triple in graph.total_triples:
            verdict = self.verify_claim(
                triple.sentence, 
                gold_id_list, 
                gold_evidence_list, 
                self.bm25_top_k, 
                self.document_level, 
            )
            subclaim_prediction = verdict["prediction"]
            avg_conf = verdict["average_verification_confidence"]
            last_conf = verdict["last_verification_confidence"]
            retrieval_info = verdict["retrieval_info"]
            supported_doc_id = verdict["supported_doc_id"]
            
            path_verification.append({
                "subclaim": triple.sentence,
                "subclaim_prediction": subclaim_prediction,
                "average_verification_confidence": avg_conf,
                "last_verification_confidence": last_conf,
                "retrieval_info": {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None,
                "supported_doc_id": supported_doc_id
            })
            if subclaim_prediction == "SUPPORTED":
                num_supported_subclaims += 1
            if subclaim_prediction == "NOT_SUPPORTED":
                path_prediction = "NOT_SUPPORTED"
        
        path_confidence = num_supported_subclaims / len(graph.total_triples) if graph.total_triples else 0.0
        
        return path_prediction, path_verification, path_confidence

    
    def process_sample_graphcheck(
        self, 
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:        
        
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(300)

            prediction = "NOT_SUPPORTED"
            verification_process = []
            
            graph = Graph(sample["definition_triples"], sample["triples"])
            

            if graph.num_la_ent > 0:
                if self.use_searchr1 and self.nudge_searchr1:
                    la_ent_list = graph.la_ent_list[:self.path_limit]
                    start_ent_to_retrieval = {}
                    for la_ent in la_ent_list:
                        sub_question = self.get_infilling_retrieval_query(graph, la_ent, use_searchr1=self.use_searchr1)
                        nudge_input = f"Question: {sub_question}"
                        
                        print("Nudge Input:\n", nudge_input)
                        thinking = self.nudge_model.generate_thinking(nudge_input)

                        evidence_list, retrieval_info = self.retrieve_evidence(
                            sample["question"], 
                            sample["gold_id_list"], 
                            sample["gold_evidence_list"], 
                            self.searchr1_top_k, 
                            use_searchr1=self.use_searchr1, 
                            use_total_search_results=self.use_total_search_results,
                            nudge_searchr1=self.nudge_searchr1, 
                            thinking=thinking, 
                        )
                        start_ent_to_retrieval[la_ent] = (evidence_list, retrieval_info)                    
                    
                    path_list = graph.get_paths_with_various_start(self.path_limit)
                else:
                    path_list = graph.get_valid_paths(self.path_limit)
            else:
                path_list = [[]]  # No latent entities to infill; use empty path
            
            for path_idx, path in enumerate(path_list):
                if self.use_searchr1 and self.nudge_searchr1:
                    start_ent = path[0]
                    evidence_list, retrieval_info = start_ent_to_retrieval[start_ent]
                else:
                    evidence_list, retrieval_info = None, None
                
                infilled_graph, infilling_log, avg_infilling_conf = self.infill_graph(
                    graph, 
                    path, 
                    sample["gold_id_list"], 
                    sample["gold_evidence_list"], 
                    self.use_searchr1, 
                    self.use_total_search_results,
                    self.nudge_searchr1,
                    evidence_list
                )

                path_prediction, path_verification, path_confidence = self.verify_graph(
                    infilled_graph, sample["gold_id_list"], sample["gold_evidence_list"]
                )
                
                path_info = f"{' - '.join(path)}" if path else None
                    
                verification_process.append({
                    "path_index": path_idx,
                    "path": path_info,
                    "path_prediction": path_prediction,
                    "path_confidence": path_confidence,
                    "infilled_definition_triples": infilled_graph.def_triple_sents,
                    "infilled_triples": infilled_graph.triple_sents,
                    "average_infilling_confidence": avg_infilling_conf,
                })

                if self.use_searchr1 and self.nudge_searchr1:
                    verification_process[-1]["searchr1_retrieval_info"] = {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None

                verification_process[-1]["infilling_log"] = infilling_log
                verification_process[-1]["path_verification"] = path_verification

                if path_prediction == "SUPPORTED":
                    prediction = "SUPPORTED"
                    break
            
            best_path_idx = max(
                range(len(verification_process)), 
                key=lambda idx: verification_process[idx]["path_confidence"]
            )
            
            infilled_triples = verification_process[best_path_idx]["infilled_definition_triples"] + verification_process[best_path_idx]["infilled_triples"]
            graph_evidence = "\n".join(infilled_triples)
            graph_evidence = graph_evidence.replace("[SEP] ", "").replace("[PREP] ", "")
            answer, conf = self.get_answer(sample["question"], graph_evidence)
            
            # Compute evaluation metrics
            ground_truth_answers = [sample["answer"]] + sample.get("answer_aliases", [])
            em_score = metric_max_over_ground_truths(compute_exact, answer, ground_truth_answers)
            f1_score = metric_max_over_ground_truths(compute_f1, answer, ground_truth_answers)

            best_infilling_conf_path_idx = max(
                range(len(verification_process)), 
                key=lambda idx: verification_process[idx]["average_infilling_confidence"]
            )

            sample.update({
                "predicted_answer": answer, 
                "answering_confidence": conf,
                "em_score": em_score,
                "f1_score": f1_score,
                "prediction": prediction,
                "best_path_index": best_path_idx,
                "best_path_confidence": verification_process[best_path_idx]["path_confidence"],
                "best_path_infilling_confidence": verification_process[best_path_idx]["average_infilling_confidence"],
                "best_infilling_conf_path_index": best_infilling_conf_path_idx,
                "best_infilling_conf_path_confidence": verification_process[best_infilling_conf_path_idx]["path_confidence"],
                "verification_process": verification_process
            })

            signal.alarm(0)

        except TimeoutException as e:
            logger.warning(f"Timeout: Function took more than 5 minutes. Skipping this sample.")
            
            prediction = random.choice(["SUPPORTED", "NOT_SUPPORTED"])
            sample.update({
                "predicted_answer": "", 
                "answering_confidence": None,
                "em_score": 0.0,
                "f1_score": 0.0,
                "prediction": prediction,
                "best_path_index": None,
                "best_path_confidence": None,
                "best_path_infilling_confidence": None,
                "best_infilling_conf_path_index": None,
                "best_infilling_conf_path_confidence": None,
                "verification_process": verification_process
            })
        
        except Exception as e:
            logger.warning(f"Failed to verify sample. Skipping this sample. Error: {e}")
            
            prediction = random.choice(["SUPPORTED", "NOT_SUPPORTED"])
            sample.update({
                "predicted_answer": "", 
                "answering_confidence": None,
                "em_score": 0.0,
                "f1_score": 0.0,
                "prediction": prediction,
                "best_path_index": None,
                "best_path_confidence": None,
                "best_path_infilling_confidence": None,
                "best_infilling_conf_path_index": None,
                "best_infilling_conf_path_confidence": None,
                "verification_process": verification_process
            })
        
        return sample
    
    
    def run_graphcheck(
        self,
        input_indices: Set[int] = None
    ) -> List[Dict[str, Any]]:

        if self.has_complete_results(self.graph_path, input_indices) and not self.force_new_construction:
            logger.info(f"Graphs for all samples are already available. Skipping construction.")

            with open(self.graph_path, "r") as f:
                graph_list = json.load(f)
            if input_indices:
                graph_list = [sample for sample in graph_list if sample["index"] in input_indices]
        
        else:
            self.construct_model = ConstructModel(
                construct_model_name=self.construct_model_name,
                dataset_name=self.dataset,
                api_key=self.api_key,
                batch_size=self.construct_batch_size
            )
            with open(self.input_path, "r") as f:
                input_list = json.load(f)
            if input_indices:
                input_list = [sample for sample in input_list if sample["index"] in input_indices]
            
            graph_list = self.construct_model.construct_graph(input_list, self.graph_path, self.force_new_construction)        

        logger.info("Starting GraphCheck answering...")

        if self.base_model is None:
            tokenizer = T5Tokenizer.from_pretrained(self.base_model_name)
            model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, device_map="auto")
            self.base_model = BaseModel(model, tokenizer)
            logger.info(f"Base model '{self.base_model_name}' initialized successfully.")        
            
        result_list = []
        for i, sample in enumerate(graph_list):
            graph_list[i] = {
                "index": sample["index"],
                "uid": sample["uid"],
                "num_hops": sample["num_hops"],
                "gold_id_list": sample["gold_id_list"],
                "gold_evidence_list": sample["gold_evidence_list"],
                "question": sample["question"],
                "answer": sample["answer"],
                "predicted_answer": None, 
                "answering_confidence": None,
                "em_score": None,
                "f1_score": None,
                "prediction": None,
                "best_path_index": None,
                "best_path_confidence": None,
                "best_path_infilling_confidence": None,
                "best_infilling_conf_path_index": None,
                "best_infilling_conf_path_confidence": None,
                "definition_triples": sample["definition_triples"],
                "triples": sample["triples"],
                "verification_process": None
            }
        
        for i, sample in enumerate(tqdm(graph_list)):
            result = self.process_sample_graphcheck(sample)
            result_list.append(result)

            if (i + 1) % 20 == 0:
                self.save_result(result_list, self.graphcheck_path)
                
        logger.info("GraphCheck answering completed!")
        
        return result_list


if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        logger.info(f"- {key}: {value}")
    
    verifier = GraphCheck(args)
    result_list = verifier.run_graphcheck()
    verifier.save_result(result_list, verifier.graphcheck_path)

