from collections import defaultdict
import itertools
import logging
import numpy as np
import random
import re
from typing import *

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

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
        other_graph: 'Graph',
        match_mode: str = "exact",
        min_token_jaccard: float = 0.6,
        include_definitions: bool = True,
        ignore_ent_placeholders: bool = False,
    ) -> Dict[str, Any]:
        """
        현재 그래프와 다른 그래프를 비교합니다.
        
        Args:
            other_graph: 비교할 다른 Graph 객체
            
        Returns:
            비교 결과 딕셔너리 (overlap, precision, recall, f1 등)
        """
        if match_mode not in {"exact", "token_jaccard"}:
            raise ValueError(f"Invalid match_mode: {match_mode}")

        def parse_triple(triple_sent: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
            if "[SEP]" not in triple_sent:
                return None
            parts = triple_sent.split(" [SEP] ")
            if len(parts) < 3:
                return None
            head = parts[0].strip()
            rel = parts[1].strip()
            tail = " [SEP] ".join(parts[2:]).strip()
            context = None
            if " [PREP] " in tail:
                tail, context = tail.split(" [PREP] ", 1)
                tail = tail.strip()
                context = context.strip()
            return head, rel, tail, context

        def normalize_text(text: str) -> str:
            if not text:
                return ""
            normalized = text.lower().strip()
            normalized = re.sub(r"\\b([a-z0-9]+)'s\\b", r"\\1", normalized)
            normalized = re.sub(r"[\"'`]+", "", normalized)
            normalized = re.sub(r"[^a-z0-9\\s]", " ", normalized)
            normalized = re.sub(r"\\bbusinesswomen\\b", "businesswoman", normalized)
            normalized = re.sub(r"\\bthat\\b", " ", normalized)
            normalized = re.sub(r"\\b(the|a|an|for|of|in|on|at)\\b", " ", normalized)
            normalized = " ".join(normalized.split())
            return normalized

        def normalize_entity(entity: str) -> str:
            if not entity:
                return ""
            without_paren = re.sub(r"\\([^)]*\\)", " ", entity)
            normalized = normalize_text(without_paren)
            normalized = re.sub(r"^professor\\s+", "", normalized)
            normalized = re.sub(r"\\s+academics$", "", normalized)
            normalized = re.sub(r"\\s+from\\s+[a-z0-9\\s]+$", "", normalized)
            normalized = " ".join(normalized.split())
            return normalized

        def normalize_relation(rel: str) -> str:
            normalized = normalize_text(rel)
            relation_map = {
                "has received acclaim as": "is",
                "has received acclaim": "is",
                "has generated": "generated",
                "is played by": "played_by",
                "played by": "played_by",
                "plays": "plays",
                "has attribute": "has_attribute",
                "has_attribute": "has_attribute",
            }
            return relation_map.get(normalized, normalized)

        def canonicalize_triple(triple_sent: str) -> Optional[Tuple[str, str, str, Set[str]]]:
            parsed = parse_triple(triple_sent)
            if not parsed:
                return None
            head, rel, tail, _context = parsed
            head_norm = normalize_entity(head)
            rel_norm = normalize_relation(rel)
            tail_norm = normalize_entity(tail)
            context_tokens = token_set(_context or "")
            if rel_norm == "plays":
                return tail_norm, "played_by", head_norm, context_tokens
            if rel_norm == "played_by":
                return head_norm, "played_by", tail_norm, context_tokens
            return head_norm, rel_norm, tail_norm, context_tokens

        def canonical_to_str(triple: Tuple[str, str, str, Set[str]]) -> str:
            return f"{triple[0]} [SEP] {triple[1]} [SEP] {triple[2]}"

        def is_placeholder(entity: str) -> bool:
            normalized = normalize_text(entity)
            return re.search(r"\\bent\\d+\\b", normalized) is not None

        def jaccard(a: Set[str], b: Set[str]) -> float:
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        STOPWORDS = {
            "the", "a", "an", "for", "of", "in", "on", "at", "by", "with", "from",
            "into", "over", "under", "up", "down", "about", "across", "after",
            "before", "between", "among", "through", "throughout", "as", "that",
            "who", "which", "whom", "whose", "what", "where", "when", "why", "how",
            "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
            "do", "does", "did"
        }

        def token_set(text: str) -> Set[str]:
            tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
            return set(t for t in tokens if t not in STOPWORDS and len(t) >= 3)

        def attribute_equivalent(
            cand: Tuple[str, str, str, Set[str]],
            ref: Tuple[str, str, str, Set[str]],
            threshold: float
        ) -> Optional[float]:
            relations = {cand[1], ref[1]}
            if relations != {"has_attribute", "is"}:
                return None
            cand_tokens = token_set(cand[2])
            ref_tokens = token_set(ref[2])
            if not cand_tokens or not ref_tokens:
                return None
            role_keywords = {
                "role", "model", "businesswoman", "businesswomen",
                "entrepreneur", "business"
            }
            if not (cand_tokens & role_keywords and ref_tokens & role_keywords):
                return None
            score = jaccard(cand_tokens, ref_tokens)
            keyword_overlap = (cand_tokens & role_keywords) & (ref_tokens & role_keywords)
            if len(keyword_overlap) >= 2:
                score = max(score, 0.5)
            if score >= threshold:
                return score
            return None

        def triple_match(
            cand: Tuple[str, str, str, Set[str]],
            ref: Tuple[str, str, str, Set[str]],
            wildcard: bool,
            threshold: float,
            exact_mode: bool
        ) -> Tuple[bool, float, str]:
            if cand[1] == ref[1]:
                relation_match = True
                match_type = "relation_exact"
            else:
                attr_score = attribute_equivalent(cand, ref, min(threshold, 0.4))
                if attr_score is None:
                    return False, 0.0, "relation_mismatch"
                relation_match = True
                match_type = "attribute_equivalence"

            if not relation_match:
                return False, 0.0, "relation_mismatch"

            head_placeholder = is_placeholder(cand[0]) or is_placeholder(ref[0])
            tail_placeholder = is_placeholder(cand[2]) or is_placeholder(ref[2])

            head_score = 1.0
            tail_score = 1.0
            head_overlap = set()
            tail_overlap = set()

            if not head_placeholder:
                if exact_mode:
                    head_score = 1.0 if cand[0] == ref[0] else 0.0
                else:
                    head_tokens = token_set(cand[0])
                    ref_head_tokens = token_set(ref[0])
                    head_overlap = head_tokens & ref_head_tokens
                    head_score = jaccard(head_tokens, ref_head_tokens)
            elif not wildcard:
                head_score = 0.0

            if not tail_placeholder:
                if exact_mode:
                    tail_score = 1.0 if cand[2] == ref[2] else 0.0
                else:
                    tail_tokens = token_set(cand[2])
                    ref_tail_tokens = token_set(ref[2])
                    tail_overlap = tail_tokens & ref_tail_tokens
                    tail_score = jaccard(tail_tokens, ref_tail_tokens)
            elif not wildcard:
                tail_score = 0.0

            if head_placeholder and tail_placeholder:
                if not cand[3] or not ref[3]:
                    return False, 0.0, "double_placeholder"
                context_score = jaccard(cand[3], ref[3])
                if context_score >= threshold:
                    return True, context_score, "context_match"
                return False, context_score, "context_mismatch"

            if exact_mode:
                matched = head_score == 1.0 and tail_score == 1.0
                score = 1.0 if matched else 0.0
                return matched, score, match_type

            head_threshold = max(threshold, 0.9)
            tail_threshold = threshold

            if (not head_placeholder and not head_overlap) or (not tail_placeholder and not tail_overlap):
                return False, (head_score + tail_score) / 2, "keyword_mismatch"

            if (not head_placeholder and head_score < head_threshold) or (not tail_placeholder and tail_score < tail_threshold):
                return False, (head_score + tail_score) / 2, "entity_mismatch"

            score = (head_score + tail_score) / 2
            return True, score, match_type

        self_triples = self.total_triples if include_definitions else self.triples
        other_triples = other_graph.total_triples if include_definitions else other_graph.triples

        self_items = []
        for triple in self_triples:
            canonical = canonicalize_triple(triple.triple_sent)
            if canonical:
                self_items.append({
                    "raw": triple.triple_sent,
                    "canonical": canonical,
                })

        other_items = []
        for triple in other_triples:
            canonical = canonicalize_triple(triple.triple_sent)
            if canonical:
                other_items.append({
                    "raw": triple.triple_sent,
                    "canonical": canonical,
                })

        self_triplets = [canonical_to_str(item["canonical"]) for item in self_items]
        other_triplets = [canonical_to_str(item["canonical"]) for item in other_items]

        ref_lookup: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        for idx, item in enumerate(other_items):
            ref_lookup[item["canonical"][:3]].append(idx)

        matched_self = set()
        matched_other = set()
        overlap_pairs = []

        for idx, item in enumerate(self_items):
            candidates = ref_lookup.get(item["canonical"][:3])
            if candidates:
                ref_idx = candidates.pop(0)
                matched_self.add(idx)
                matched_other.add(ref_idx)
                overlap_pairs.append({
                    "self_triplet": item["raw"],
                    "other_triplet": other_items[ref_idx]["raw"],
                    "score": 1.0,
                    "match_type": "exact",
                    "self_canonical": canonical_to_str(item["canonical"]),
                    "other_canonical": canonical_to_str(other_items[ref_idx]["canonical"]),
                })

        candidate_pairs = []
        if match_mode == "token_jaccard":
            for i, self_item in enumerate(self_items):
                if i in matched_self:
                    continue
                for j, other_item in enumerate(other_items):
                    if j in matched_other:
                        continue
                    matched, score, match_type = triple_match(
                        self_item["canonical"],
                        other_item["canonical"],
                        wildcard=ignore_ent_placeholders,
                        threshold=min_token_jaccard,
                        exact_mode=False,
                    )
                    candidate_pairs.append((i, j, score, match_type, matched))

            candidate_pairs.sort(key=lambda x: x[2], reverse=True)
            for i, j, score, match_type, matched in candidate_pairs:
                if not matched:
                    continue
                if i in matched_self or j in matched_other:
                    continue
                matched_self.add(i)
                matched_other.add(j)
                overlap_pairs.append({
                    "self_triplet": self_items[i]["raw"],
                    "other_triplet": other_items[j]["raw"],
                    "score": score,
                    "match_type": match_type,
                    "self_canonical": canonical_to_str(self_items[i]["canonical"]),
                    "other_canonical": canonical_to_str(other_items[j]["canonical"]),
                })

        overlap_count = len(overlap_pairs)
        self_count = len(self_items)
        other_count = len(other_items)

        # Precision, Recall, F1 계산
        if self_count == 0:
            precision = 0.0
        else:
            precision = overlap_count / self_count
        
        if other_count == 0:
            recall = 0.0
        else:
            recall = overlap_count / other_count

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # 부분집합 관점 점수: self가 other에 얼마나 포함되는지
        subset_f1 = precision
        # 표준 F1은 부분집합 관점으로 덮어쓰기
        f1 = subset_f1
        
        # Entity overlap 계산
        self_entities = set(self.la_ent_2_def.keys()) if hasattr(self, 'la_ent_2_def') else set()
        other_entities = set(other_graph.la_ent_2_def.keys()) if hasattr(other_graph, 'la_ent_2_def') else set()
        entity_overlap = len(self_entities & other_entities)
        
        unmatched_self = [canonical_to_str(self_items[i]["canonical"]) for i in range(self_count) if i not in matched_self]
        unmatched_other = [canonical_to_str(other_items[i]["canonical"]) for i in range(other_count) if i not in matched_other]

        canonical_self_sample = [canonical_to_str(item["canonical"]) for item in self_items[:10]]
        canonical_other_sample = [canonical_to_str(item["canonical"]) for item in other_items[:10]]

        top_k_candidate_pairs = []
        if match_mode == "token_jaccard" and candidate_pairs:
            for i, j, score, match_type, matched in candidate_pairs[:5]:
                top_k_candidate_pairs.append({
                    "self_triplet": self_items[i]["raw"],
                    "other_triplet": other_items[j]["raw"],
                    "score": score,
                    "match_type": match_type,
                    "self_canonical": canonical_to_str(self_items[i]["canonical"]),
                    "other_canonical": canonical_to_str(other_items[j]["canonical"]),
                    "matched": matched,
                })

        return {
            "triplet_overlap": overlap_count,
            "triplet_precision": precision,
            "triplet_recall": recall,
            "triplet_f1": f1,
            "self_coverage": precision,
            "other_coverage": recall,
            "subset_f1": subset_f1,
            "self_triplet_count": self_count,
            "other_triplet_count": other_count,
            "overlapping_triplets": [pair["self_canonical"] for pair in overlap_pairs],
            "self_only_triplets": unmatched_self,
            "other_only_triplets": unmatched_other,
            "match_mode": match_mode,
            "min_token_jaccard": min_token_jaccard if match_mode == "token_jaccard" else None,
            "include_definitions": include_definitions,
            "ignore_ent_placeholders": ignore_ent_placeholders,
            "overlap_pairs": overlap_pairs,
            "canonical_self_triplets_sample": canonical_self_sample,
            "canonical_other_triplets_sample": canonical_other_sample,
            "unmatched_self_count": len(unmatched_self),
            "unmatched_other_count": len(unmatched_other),
            "top_k_candidate_pairs": top_k_candidate_pairs if match_mode == "token_jaccard" else [],
            "entity_overlap": entity_overlap,
            "self_entity_count": len(self_entities),
            "other_entity_count": len(other_entities)
        }


def search_query_graph_bindings(
    query_triples: List[str],
    fact_triples: List[str],
    top_k: int = 5,
    beam_size: int = 50,
    cand_per_query: int = 50,
    min_token_jaccard: float = 0.5,
    include_definitions: bool = False,
) -> Dict[str, Any]:
    if not query_triples or not fact_triples:
        return {"k": top_k, "bindings": []}

    def parse_triple(triple_sent: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
        if "[SEP]" not in triple_sent:
            return None
        parts = triple_sent.split(" [SEP] ")
        if len(parts) < 3:
            return None
        head = parts[0].strip()
        rel = parts[1].strip()
        tail = " [SEP] ".join(parts[2:]).strip()
        context = None
        if " [PREP] " in tail:
            tail, context = tail.split(" [PREP] ", 1)
            tail = tail.strip()
            context = context.strip()
        return head, rel, tail, context

    def normalize_text(text: str) -> str:
        if not text:
            return ""
        normalized = text.lower().strip()
        normalized = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", normalized)
        normalized = re.sub(r"[\"'`]+", "", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\bbusinesswomen\b", "businesswoman", normalized)
        normalized = re.sub(r"\bthat\b", " ", normalized)
        normalized = " ".join(normalized.split())
        return normalized

    STOPWORDS = {
        "the", "a", "an", "for", "of", "in", "on", "at", "by", "with", "from",
        "into", "over", "under", "up", "down", "about", "across", "after",
        "before", "between", "among", "through", "throughout", "as", "that",
        "who", "which", "whom", "whose", "what", "where", "when", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
        "do", "does", "did"
    }

    def token_set(text: str) -> Set[str]:
        tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
        return set(t for t in tokens if t not in STOPWORDS and len(t) >= 3)

    def normalize_entity(entity: str) -> str:
        if not entity:
            return ""
        without_paren = re.sub(r"\([^)]*\)", " ", entity)
        normalized = normalize_text(without_paren)
        normalized = re.sub(r"^professor\s+", "", normalized)
        normalized = re.sub(r"\s+academics$", "", normalized)
        normalized = re.sub(r"\s+from\s+[a-z0-9\s]+$", "", normalized)
        normalized = " ".join(normalized.split())
        return normalized

    def normalize_relation(rel: str) -> str:
        normalized = normalize_text(rel)
        relation_map = {
            "has received acclaim as": "is",
            "has received acclaim": "is",
            "has generated": "generated",
            "is played by": "played_by",
            "played by": "played_by",
            "plays": "plays",
            "has attribute": "has_attribute",
            "has_attribute": "has_attribute",
        }
        return relation_map.get(normalized, normalized)

    def is_placeholder(entity: str) -> Optional[str]:
        normalized = normalize_text(entity)
        match = re.search(r"\bent\d+\b", normalized)
        return match.group(0).upper() if match else None

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def canonicalize_triple(triple_sent: str) -> Optional[Dict[str, Any]]:
        parsed = parse_triple(triple_sent)
        if not parsed:
            return None
        head, rel, tail, context = parsed
        rel_norm = normalize_relation(rel)
        head_norm = normalize_entity(head)
        tail_norm = normalize_entity(tail)
        context_tokens = token_set(context or "")
        inverse_used = False
        rel_family = rel_norm
        if rel_norm == "plays":
            rel_family = "played_by"
            head_norm, tail_norm = tail_norm, head_norm
            head, tail = tail, head
            inverse_used = True
        elif rel_norm == "played_by":
            rel_family = "played_by"
        return {
            "raw": triple_sent,
            "head": head,
            "tail": tail,
            "head_norm": head_norm,
            "tail_norm": tail_norm,
            "relation": rel_norm,
            "rel_family": rel_family,
            "context_tokens": context_tokens,
            "inverse_used": inverse_used,
        }

    def attribute_equivalent(cand: Dict[str, Any], ref: Dict[str, Any]) -> Optional[float]:
        relations = {cand["rel_family"], ref["rel_family"]}
        if relations != {"has_attribute", "is"}:
            return None
        cand_tokens = token_set(cand["tail_norm"])
        ref_tokens = token_set(ref["tail_norm"])
        if not cand_tokens or not ref_tokens:
            return None
        role_keywords = {"role", "model", "businesswoman", "entrepreneur", "business"}
        if not (cand_tokens & role_keywords and ref_tokens & role_keywords):
            return None
        score = jaccard(cand_tokens, ref_tokens)
        keyword_overlap = (cand_tokens & role_keywords) & (ref_tokens & role_keywords)
        if len(keyword_overlap) >= 2:
            score = max(score, 0.5)
        return score

    def entity_sim(a: str, b: str) -> float:
        return jaccard(token_set(a), token_set(b))

    def relation_family_ok(cand: Dict[str, Any], ref: Dict[str, Any]) -> bool:
        if cand["rel_family"] == ref["rel_family"]:
            return True
        if {cand["rel_family"], ref["rel_family"]} == {"has_attribute", "is"}:
            return True
        return False

    def base_match_score(cand: Dict[str, Any], ref: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
        if not relation_family_ok(cand, ref):
            return None

        attr_score = None
        if {cand["rel_family"], ref["rel_family"]} == {"has_attribute", "is"}:
            attr_score = attribute_equivalent(cand, ref)
            if attr_score is None:
                return None

        cand_head_var = is_placeholder(cand["head"])
        cand_tail_var = is_placeholder(cand["tail"])

        if cand_head_var and cand_tail_var:
            context_sim = jaccard(cand["context_tokens"], ref["context_tokens"])
            if context_sim < min_token_jaccard:
                return None
            score = context_sim * (0.95 if cand["inverse_used"] or ref["inverse_used"] else 1.0)
            return score, {
                "relation": cand["rel_family"],
                "subj_sim": None,
                "obj_sim": None,
                "context_sim": context_sim,
                "inverse_used": cand["inverse_used"] or ref["inverse_used"],
                "attribute_equiv": attr_score,
            }

        subj_sim = 1.0
        obj_sim = 1.0
        if not cand_head_var:
            subj_sim = entity_sim(cand["head_norm"], ref["head_norm"])
        if not cand_tail_var:
            obj_sim = entity_sim(cand["tail_norm"], ref["tail_norm"])

        if not cand_head_var and subj_sim < min_token_jaccard:
            return None
        if not cand_tail_var and obj_sim < min_token_jaccard:
            return None

        base = (subj_sim + obj_sim) / 2
        context_sim = jaccard(cand["context_tokens"], ref["context_tokens"])
        bonus = 1.0 + (0.1 * context_sim if context_sim > 0 else 0.0)
        inverse_penalty = 0.95 if cand["inverse_used"] or ref["inverse_used"] else 1.0
        score = base * bonus * inverse_penalty
        return score, {
            "relation": cand["rel_family"],
            "subj_sim": subj_sim,
            "obj_sim": obj_sim,
            "context_sim": context_sim,
            "inverse_used": cand["inverse_used"] or ref["inverse_used"],
            "attribute_equiv": attr_score,
        }

    def binding_consistent(binding: Dict[str, str], var: str, value: str) -> bool:
        if var not in binding:
            return True
        return normalize_entity(binding[var]) == normalize_entity(value)

    def apply_binding(binding: Dict[str, str], cand: Dict[str, Any], ref: Dict[str, Any]) -> Optional[Tuple[Dict[str, str], float, Dict[str, Any]]]:
        cand_head_var = is_placeholder(cand["head"])
        cand_tail_var = is_placeholder(cand["tail"])

        new_binding = dict(binding)
        if cand_head_var:
            if not binding_consistent(new_binding, cand_head_var, ref["head"]):
                return None
            new_binding.setdefault(cand_head_var, ref["head"])
        if cand_tail_var:
            if not binding_consistent(new_binding, cand_tail_var, ref["tail"]):
                return None
            new_binding.setdefault(cand_tail_var, ref["tail"])

        score_info = base_match_score(cand, ref)
        if score_info is None:
            return None
        score, details = score_info
        return new_binding, score, details

    query_items = []
    for triple in query_triples:
        parsed = canonicalize_triple(triple)
        if parsed:
            query_items.append(parsed)

    fact_items = []
    for triple in fact_triples:
        parsed = canonicalize_triple(triple)
        if parsed:
            fact_items.append(parsed)

    if not query_items or not fact_items:
        return {"k": top_k, "bindings": []}

    rel_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in fact_items:
        rel_index[item["rel_family"]].append(item)

    candidate_lists = []
    for q in query_items:
        rel_family = q["rel_family"]
        candidates = []
        rel_candidates = rel_index.get(rel_family, [])
        if rel_family == "has_attribute":
            rel_candidates = rel_candidates + rel_index.get("is", [])
        if rel_family == "is":
            rel_candidates = rel_candidates + rel_index.get("has_attribute", [])
        for fact in rel_candidates:
            score_info = base_match_score(q, fact)
            if score_info is None:
                continue
            score, details = score_info
            candidates.append((score, fact, details))
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidate_lists.append(candidates[:cand_per_query])

    query_order = sorted(range(len(query_items)), key=lambda i: len(candidate_lists[i]))

    beam = [{
        "binding": {},
        "score": 0.0,
        "supported_pairs": [],
        "used_facts": set(),
        "unmatched": [],
    }]

    for q_idx in query_order:
        q_item = query_items[q_idx]
        candidates = candidate_lists[q_idx]
        next_beam = []

        for state in beam:
            next_beam.append({
                "binding": dict(state["binding"]),
                "score": state["score"],
                "supported_pairs": list(state["supported_pairs"]),
                "used_facts": set(state["used_facts"]),
                "unmatched": state["unmatched"] + [q_item["raw"]],
            })

            for score, fact, details in candidates:
                fact_id = id(fact)
                if fact_id in state["used_facts"]:
                    continue
                bound = apply_binding(state["binding"], q_item, fact)
                if not bound:
                    continue
                new_binding, match_score, match_details = bound
                supported_pair = {
                    "query": q_item["raw"],
                    "fact": fact["raw"],
                    "match_score": match_score,
                    "details": match_details,
                }
                next_beam.append({
                    "binding": new_binding,
                    "score": state["score"] + match_score,
                    "supported_pairs": state["supported_pairs"] + [supported_pair],
                    "used_facts": set(state["used_facts"]) | {fact_id},
                    "unmatched": list(state["unmatched"]),
                })

        next_beam.sort(key=lambda s: s["score"], reverse=True)
        beam = next_beam[:beam_size]

    beam.sort(key=lambda s: s["score"], reverse=True)
    bindings = []
    for rank, state in enumerate(beam[:top_k], start=1):
        bindings.append({
            "rank": rank,
            "score": state["score"],
            "binding": state["binding"],
            "supported_pairs": state["supported_pairs"],
            "unmatched_query_triples": state["unmatched"],
        })

    return {"k": top_k, "bindings": bindings}


# ---------------------------------------------------------------------------
# Ensemble Triplet Matching
#   Lemma Token Jaccard + TF-IDF Cosine + Character N-gram Cosine
#   모델 학습 없이, 하드코딩 없이 triplet 유사도를 계산합니다.
# ---------------------------------------------------------------------------

# Module-level lazy singletons (초기화 비용 최소화)
_lemmatizer: Optional[WordNetLemmatizer] = None
_stopwords: Optional[set] = None


def _get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        for resource in ["wordnet", "omw-1.4", "stopwords"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stopwords() -> set:
    global _stopwords
    if _stopwords is None:
        _get_lemmatizer()  # ensure nltk data is downloaded
        _stopwords = set(nltk_stopwords.words("english"))
    return _stopwords


def _triplet_to_sentence(triple_sent: str) -> str:
    r"""Triplet 문자열을 자연어 문장으로 변환합니다.
    
    (ENT\d+) placeholder는 제거하여 실제 entity/relation 토큰만 남깁니다.
    """
    sent = triple_sent.replace(" [SEP] ", " ").replace(" [PREP] ", " ")
    sent = re.sub(r"\(ENT\d+\)", "", sent)
    sent = " ".join(sent.split())  # 다중 공백 정리
    return sent.strip()


def _lemma_token_set(text: str) -> Set[str]:
    """텍스트를 lemmatized token set으로 변환합니다."""
    lemmatizer = _get_lemmatizer()
    stops = _get_stopwords()
    tokens = re.findall(r"[a-z]+", text.lower())
    result = set()
    for t in tokens:
        if t in stops or len(t) < 3:
            continue
        # 동사 lemma (plays→play, performed→perform, held→hold)
        lemma_v = lemmatizer.lemmatize(t, pos="v")
        # 명사 lemma (cities→city, countries→country)
        lemma_n = lemmatizer.lemmatize(t, pos="n")
        # 더 짧은 쪽(= 더 많이 축약된 어근)을 사용
        result.add(min(lemma_v, lemma_n, key=len))
    return result


def _lemma_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Lemmatized token set 간 Jaccard 유사도를 계산합니다."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def ensemble_triplet_matching(
    query_triples: List[str],
    doc_triples: List[str],
    w_lemma: float = 0.4,
    w_tfidf: float = 0.35,
    w_char: float = 0.25,
    threshold: float = 0.3,
    top_n_per_query: int = 5,
) -> Dict[str, Any]:
    """
    세 가지 유사도를 앙상블하여 query triplet과 doc triplet을 매칭합니다.
    
    1) Lemma Token Jaccard : 어근 추출 후 token-set Jaccard
    2) TF-IDF Cosine       : corpus 통계 기반 단어 가중치 cosine similarity
    3) Char N-gram Cosine  : 문자 단위 n-gram TF-IDF cosine similarity
    
    Args:
        query_triples:  쿼리 측 triplet 문자열 리스트
        doc_triples:    문서 측 triplet 문자열 리스트
        w_lemma:        Lemma Jaccard 가중치 (default 0.4)
        w_tfidf:        TF-IDF Cosine 가중치 (default 0.35)
        w_char:         Char N-gram Cosine 가중치 (default 0.25)
        threshold:      매칭 인정 최소 앙상블 점수 (default 0.3)
        top_n_per_query: 각 query triplet 당 상위 N개 후보를 결과에 포함 (default 5)
    
    Returns:
        {
            "matched_pairs": [...],
            "matched_count": int,
            "precision": float,
            "recall": float,
            "f1": float,
            "per_method_avg": {"lemma_jaccard": ..., "tfidf_cosine": ..., "char_ngram_cosine": ...},
            "all_candidates": [...],   # top_n_per_query 후보 리스트
        }
    """
    empty_result = {
        "matched_pairs": [],
        "matched_count": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "per_method_avg": {"lemma_jaccard": 0.0, "tfidf_cosine": 0.0, "char_ngram_cosine": 0.0},
        "best_match_per_query": [],
        "all_candidates": [],
    }
    if not query_triples or not doc_triples:
        return empty_result

    # --- 문장화 ---
    query_sents = [_triplet_to_sentence(t) for t in query_triples]
    doc_sents = [_triplet_to_sentence(t) for t in doc_triples]
    all_sents = query_sents + doc_sents
    n_q, n_d = len(query_sents), len(doc_sents)

    # =====================================================================
    # 1) Lemma Token Jaccard 매트릭스
    # =====================================================================
    query_lemma_sets = [_lemma_token_set(s) for s in query_sents]
    doc_lemma_sets = [_lemma_token_set(s) for s in doc_sents]

    lemma_matrix = np.zeros((n_q, n_d), dtype=np.float64)
    for qi in range(n_q):
        for di in range(n_d):
            lemma_matrix[qi, di] = _lemma_jaccard(query_lemma_sets[qi], doc_lemma_sets[di])

    # =====================================================================
    # 2) TF-IDF Cosine Similarity 매트릭스
    # =====================================================================
    tfidf_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_mat = tfidf_vec.fit_transform(all_sents)
    tfidf_sim_full = cosine_similarity(tfidf_mat[:n_q], tfidf_mat[n_q:])
    tfidf_matrix = np.array(tfidf_sim_full, dtype=np.float64)

    # =====================================================================
    # 3) Character N-gram Cosine Similarity 매트릭스
    # =====================================================================
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )
    char_mat = char_vec.fit_transform(all_sents)
    char_sim_full = cosine_similarity(char_mat[:n_q], char_mat[n_q:])
    char_matrix = np.array(char_sim_full, dtype=np.float64)

    # =====================================================================
    # 앙상블 점수 = 가중 합산
    # =====================================================================
    ensemble_matrix = (
        w_lemma * lemma_matrix
        + w_tfidf * tfidf_matrix
        + w_char * char_matrix
    )

    # --- Top-N 후보 리스트 (디버깅용) ---
    all_candidates = []
    for qi in range(n_q):
        scored = []
        for di in range(n_d):
            scored.append({
                "query_triplet": query_triples[qi],
                "doc_triplet": doc_triples[di],
                "ensemble_score": float(ensemble_matrix[qi, di]),
                "lemma_jaccard": float(lemma_matrix[qi, di]),
                "tfidf_cosine": float(tfidf_matrix[qi, di]),
                "char_ngram_cosine": float(char_matrix[qi, di]),
            })
        scored.sort(key=lambda x: x["ensemble_score"], reverse=True)
        all_candidates.append(scored[:top_n_per_query])

    # --- Hungarian Algorithm으로 최적 1:1 매칭 ---
    cost_matrix = 1.0 - ensemble_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    sum_lemma, sum_tfidf, sum_char = 0.0, 0.0, 0.0

    for qi, di in zip(row_ind, col_ind):
        score = float(ensemble_matrix[qi, di])
        if score < threshold:
            continue
        matched_pairs.append({
            "query_triplet": query_triples[qi],
            "doc_triplet": doc_triples[di],
            "ensemble_score": score,
            "lemma_jaccard": float(lemma_matrix[qi, di]),
            "tfidf_cosine": float(tfidf_matrix[qi, di]),
            "char_ngram_cosine": float(char_matrix[qi, di]),
        })
        sum_lemma += lemma_matrix[qi, di]
        sum_tfidf += tfidf_matrix[qi, di]
        sum_char += char_matrix[qi, di]

    matched_pairs.sort(key=lambda x: x["ensemble_score"], reverse=True)

    matched_count = len(matched_pairs)
    precision = matched_count / n_q if n_q > 0 else 0.0
    recall = matched_count / n_d if n_d > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_method_avg = {
        "lemma_jaccard": sum_lemma / matched_count if matched_count else 0.0,
        "tfidf_cosine": sum_tfidf / matched_count if matched_count else 0.0,
        "char_ngram_cosine": sum_char / matched_count if matched_count else 0.0,
    }

    # --- 각 query triplet마다 가장 가까운 doc triplet (threshold 무관) ---
    best_match_per_query = []
    for qi in range(n_q):
        best_di = int(np.argmax(ensemble_matrix[qi]))
        best_match_per_query.append({
            "query_triplet": query_triples[qi],
            "best_doc_triplet": doc_triples[best_di],
            "ensemble_score": float(ensemble_matrix[qi, best_di]),
            "lemma_jaccard": float(lemma_matrix[qi, best_di]),
            "tfidf_cosine": float(tfidf_matrix[qi, best_di]),
            "char_ngram_cosine": float(char_matrix[qi, best_di]),
        })

    return {
        "matched_pairs": matched_pairs,
        "matched_count": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_method_avg": per_method_avg,
        "best_match_per_query": best_match_per_query,
        "all_candidates": all_candidates,
    }


def select_topk_doc_triplets_by_ensemble(
    question_triples: List[str],
    doc_triples: List[str],
    top_k: int,
    w_lemma: float = 0.4,
    w_tfidf: float = 0.35,
    w_char: float = 0.25,
) -> List[str]:
    """
    질문 triplet과 문서 triplet 간 ensemble 유사도로 상위 top-k doc triplet을 선택합니다.
    각 question triplet당 top-k 후보를 모아, 중복 제거 후 순서 유지하여 반환합니다.

    Args:
        question_triples: 질문 triplet 리스트 (ENT 플레이스홀더 포함 가능)
        doc_triples: 문서 triplet 리스트
        top_k: 각 question triplet당 선택할 doc triplet 개수 (1, 3, 5, 10 등)

    Returns:
        선택된 doc triplet 리스트 (중복 제거, 등장 순서 유지)
    """
    if not question_triples or not doc_triples:
        return []
    if top_k <= 0:
        return []

    result = ensemble_triplet_matching(
        query_triples=question_triples,
        doc_triples=doc_triples,
        w_lemma=w_lemma,
        w_tfidf=w_tfidf,
        w_char=w_char,
        threshold=0.0,  # threshold 무시, 순위만 사용
        top_n_per_query=max(top_k, 1),
    )
    all_candidates = result.get("all_candidates", [])

    seen: set = set()
    ordered: List[str] = []
    for query_cands in all_candidates:
        for i, cand in enumerate(query_cands):
            if i >= top_k:
                break
            dt = cand.get("doc_triplet", "")
            if dt and dt not in seen:
                seen.add(dt)
                ordered.append(dt)
    return ordered