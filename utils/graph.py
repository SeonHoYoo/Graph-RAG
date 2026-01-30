from collections import defaultdict
import itertools
import logging
import numpy as np
import random
import re
from typing import *

logger = logging.getLogger(__name__)
random.seed(26)


class Triple:
    def __init__(
        self, 
        triple_sent: str
    ):
        self.triple_sent = triple_sent
        self.elements = self.split_all(triple_sent)
        self.sentence = " ".join(self.elements) if self.elements else None
    
    
    def split_all(
        self, 
        triple: str
    ) -> Optional[List[str]]:
        
        if "[SEP]" not in triple:
            logger.error(f"Invalid triple format: '{triple}'")            
            return None
        
        elements = re.split(r"\[SEP\]|\[PREP\]", triple)
        elements = [ele.strip() for ele in elements if ele.strip()]
        
        return elements


class DefinitionTriple(Triple):
    def __init__(
        self, 
        def_triple_sent: str
    ):
        super().__init__(def_triple_sent)
        
        if not self.elements:
            self.latent_entity = None
            self.definition = None
        else:
            self.latent_entity = self.elements[0]
            self.definition = " ".join(self.elements[2:]) if len(self.elements) > 2 else ""


class Graph:
    def __init__(
        self, 
        def_triple_sents: List[str], 
        triple_sents: List[str]
    ):
        self.def_triple_sents = def_triple_sents
        self.triple_sents = triple_sents
        
        self.def_triples = [
            def_triple for sent in def_triple_sents if (def_triple := DefinitionTriple(sent)).elements is not None]
        self.triples: List[Triple] = [
            triple for sent in triple_sents if (triple := Triple(sent)).elements is not None]
        self.total_triples = self.def_triples + self.triples
        
        self.la_ent_2_def = self.get_la_ent_2_def()
        self.la_ent_2_def_triple = self.get_la_ent_2_def_triple()
        
        self.la_ent_list = list(self.la_ent_2_def.keys())
        self.la_ent_index = {la_ent: idx for idx, la_ent in enumerate(self.la_ent_list)}
        self.num_la_ent = len(self.la_ent_list)
                
        self.has_la_ent_w_no_def = 0
        self.la_ent_2_sub_triples = self.get_la_ent_2_sub_triples()
        self.adjacent_la_ent_pairs = None
        
        self.count_2_triple = defaultdict(list)
        for triple in self.triples:
            count = len(re.findall(r"\(ENT\d+\)", triple.triple_sent))
            self.count_2_triple[count].append(triple)
    
    def get_la_ent_2_def(
        self
    ) -> Dict[str, str]:
        
        la_ent_2_def = {}
        for def_triple in self.def_triples:
            la_ent_2_def[def_triple.latent_entity] = def_triple.definition
        
        return la_ent_2_def
    
    
    def get_la_ent_2_def_triple(
        self
    ) -> Dict[str, DefinitionTriple]:
        
        la_ent_2_def_triple = {}
        for def_triple in self.def_triples:
            la_ent_2_def_triple[def_triple.latent_entity] = def_triple
        
        return la_ent_2_def_triple
    
    
    def get_la_ent_2_sub_triples(
        self
    ) -> DefaultDict[str, List[Triple]]:
        
        la_ent_2_sub_triples = defaultdict(list)
        
        for triple in self.triples:
            la_ents = set(re.findall(r"\(ENT\d+\)", triple.sentence))
            
            for la_ent in la_ents:
                if la_ent in self.la_ent_list:
                    la_ent_2_sub_triples[la_ent].append(triple)
                else:
                    self.has_la_ent_w_no_def = 1
        
        return la_ent_2_sub_triples
    
    
    def get_adjacent_la_ent_pairs(
        self
    ) -> List[Tuple[str, str]]:
        
        adjacency_matrix = np.zeros((self.num_la_ent, self.num_la_ent), dtype=int)
        
        for triple in self.total_triples:
            la_ents = re.findall(r"\(ENT\d+\)", triple.sentence)
            
            for i in range(len(la_ents)):
                for j in range(i + 1, len(la_ents)):
                    if la_ents[i] in self.la_ent_list and la_ents[j] in self.la_ent_list:
                        idx1, idx2 = self.la_ent_index[la_ents[i]], self.la_ent_index[la_ents[j]]
                        adjacency_matrix[idx1][idx2] = 1
                        adjacency_matrix[idx2][idx1] = 1
        
        pair_list = []
        for idx1 in range(self.num_la_ent):
            for idx2 in range(idx1 + 1, self.num_la_ent):
                if adjacency_matrix[idx1][idx2] == 1:
                    pair_list.append((self.la_ent_list[idx1], self.la_ent_list[idx2]))
        
        return pair_list
    
    
    def backtrack(
        self, 
        rule: List[Tuple[str, str]], 
        path: List[str], 
        used_ent: List[str]
    ) -> Optional[List[str]]:
        
        if len(path) == self.num_la_ent:
            return path
        
        for ent in self.la_ent_list:
            if ent not in used_ent:
                follow_rule = True
                updated_path = path + [ent]
                
                # Check if the new entity maintains the rule constraints
                for (a, b) in rule:
                    if a in updated_path and b in updated_path:
                        if updated_path.index(a) > updated_path.index(b):
                            follow_rule = False
                            break
                
                if follow_rule:
                    used_ent.add(ent)
                    result = self.backtrack(rule, updated_path, used_ent)
                    used_ent.remove(ent)
                    
                    if result:
                        return result # Return the first valid sequence found
        
        return None
    
    
    def get_valid_paths(
        self, 
        path_limit: int = 5
    ) -> List[List[str]]:
        """
        Generate latent entity sequences where order variations may lead to different results in latent entity identification.
        
        - The order of adjacent nodes (i.e., latent entities with direct connections) affects the outcome.
        - The order of non-adjacent nodes (i.e., latent entities without direct connections) does not affect the outcome.
        
        Based on this, the function generates sequences with different orderings of adjacent nodes.
        """
        
        if not self.adjacent_la_ent_pairs:
            self.adjacent_la_ent_pairs = self.get_adjacent_la_ent_pairs()
        
        # Generate all possible adjacency pair permutations
        rule_list = []
        for do_flip in itertools.product([False, True], repeat=len(self.adjacent_la_ent_pairs)):
            rule = []
            for (left_ent, right_ent), flip in zip(self.adjacent_la_ent_pairs, do_flip):
                if flip:
                    rule.append((right_ent, left_ent))
                else:
                    rule.append((left_ent, right_ent))
            rule_list.append(rule)
        
        # Shuffle rules if there are more than path_limit to introduce randomness
        if len(rule_list) > path_limit:
            random.shuffle(rule_list)
        
        valid_paths = []
        
        for rule in rule_list:
            path = self.backtrack(rule, [], set())
            if path and path not in valid_paths:
                valid_paths.append(path)
                
            if len(valid_paths) >= path_limit:
                break
        
        valid_paths = [list(seq) for seq in valid_paths]
        return valid_paths
    

    def get_paths_with_various_start(self, path_limit: int) -> List[List[str]]:
        # Estimate needed paths: assume uniform distribution across start entities
        # Request more paths to ensure coverage, but cap at reasonable limit
        estimated_paths = min(path_limit * 10, 1000)
        all_paths = self.get_valid_paths(estimated_paths)

        if not all_paths:
            return []

        # Group paths by starting entity
        start_ent_2_paths = defaultdict(list)
        for path in all_paths:
            start_ent_2_paths[path[0]].append(path)

        # Round-robin selection to ensure variety in starting entities
        filtered_paths = []
        path_lists = list(start_ent_2_paths.values())

        while len(filtered_paths) < path_limit and any(path_lists):
            for path_list in path_lists:
                if path_list:
                    filtered_paths.append(path_list.pop(0))
                    if len(filtered_paths) >= path_limit:
                        break

        return filtered_paths[:path_limit]
    
    
    def compare_with(
        self,
        other_graph: 'Graph'
    ) -> Dict[str, Any]:
        """
        현재 그래프와 다른 그래프를 비교합니다.
        
        Args:
            other_graph: 비교할 다른 Graph 객체
            
        Returns:
            비교 결과 딕셔너리 (overlap, precision, recall, f1 등)
        """
        # Triplet 문자열을 정규화하여 비교
        def normalize_triplet(triplet_sent: str) -> str:
            # 공백 정규화 및 소문자 변환
            return " ".join(triplet_sent.lower().split())
        
        # 현재 그래프의 triplet들
        self_triplets = set(normalize_triplet(t.triple_sent) for t in self.total_triples)
        other_triplets = set(normalize_triplet(t.triple_sent) for t in other_graph.total_triples)
        
        # 겹치는 triplet들
        overlap = self_triplets & other_triplets
        
        # Precision, Recall, F1 계산
        if len(self_triplets) == 0:
            precision = 0.0
        else:
            precision = len(overlap) / len(self_triplets)
        
        if len(other_triplets) == 0:
            recall = 0.0
        else:
            recall = len(overlap) / len(other_triplets)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # Entity overlap 계산
        self_entities = set(self.la_ent_2_def.keys()) if hasattr(self, 'la_ent_2_def') else set()
        other_entities = set(other_graph.la_ent_2_def.keys()) if hasattr(other_graph, 'la_ent_2_def') else set()
        entity_overlap = len(self_entities & other_entities)
        
        return {
            "triplet_overlap": len(overlap),
            "triplet_precision": precision,
            "triplet_recall": recall,
            "triplet_f1": f1,
            "self_triplet_count": len(self_triplets),
            "other_triplet_count": len(other_triplets),
            "overlapping_triplets": list(overlap),
            "self_only_triplets": list(self_triplets - other_triplets),
            "other_only_triplets": list(other_triplets - self_triplets),
            "entity_overlap": entity_overlap,
            "self_entity_count": len(self_entities),
            "other_entity_count": len(other_entities)
        }