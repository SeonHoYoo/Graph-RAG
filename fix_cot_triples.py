"""
기존 JSON 파일에서 cot_def_triples에 잘못 포함된 관계 triplets를 
cot_triples로 이동시키는 스크립트
"""

import json
import os
from typing import List, Tuple

def is_definition_triple(triple: str) -> bool:
    """
    정의 triple인지 확인: (ENTX) [SEP] is [SEP] 형태만 정의 triple
    """
    if not triple.startswith("(ENT"):
        return False
    if " [SEP] is [SEP] " not in triple:
        return False
    return True

def separate_def_and_relation_triples(triples: List[str]) -> Tuple[List[str], List[str]]:
    """
    triplets 리스트를 정의 triplets와 관계 triplets로 분리
    """
    def_triples = []
    relation_triples = []
    
    for triple in triples:
        if is_definition_triple(triple):
            def_triples.append(triple)
        else:
            relation_triples.append(triple)
    
    return def_triples, relation_triples

def fix_sample(sample: dict) -> dict:
    """
    샘플의 cot_def_triples와 cot_triples를 수정
    """
    cot_def_triples = sample.get("cot_def_triples", [])
    cot_triples = sample.get("cot_triples", [])
    
    # 기존 cot_triples와 합침
    all_triples = cot_def_triples + cot_triples
    
    # 정의와 관계로 분리
    new_def_triples, new_relation_triples = separate_def_and_relation_triples(all_triples)
    
    # 업데이트
    sample["cot_def_triples"] = new_def_triples
    sample["cot_triples"] = new_relation_triples
    
    # cot_graph도 업데이트
    if "cot_graph" in sample and sample["cot_graph"] is not None:
        cot_graph_def = sample["cot_graph"].get("definition_triples", [])
        cot_graph_triples = sample["cot_graph"].get("triples", [])
        
        all_graph_triples = cot_graph_def + cot_graph_triples
        new_graph_def, new_graph_triples = separate_def_and_relation_triples(all_graph_triples)
        
        sample["cot_graph"]["definition_triples"] = new_graph_def
        sample["cot_graph"]["triples"] = new_graph_triples
        sample["cot_graph"]["num_triplets"] = len(new_graph_def) + len(new_graph_triples)
    
    return sample

def main():
    input_path = "/data3/seonhoyoo/graphcheck-qa/results/musique/graph_comparison/Qwen2.5-14B-Instruct/compare_graphs_train_sampled_open_book.json"
    
    if not os.path.exists(input_path):
        print(f"파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"파일 로딩 중: {input_path}")
    with open(input_path, "r") as f:
        data = json.load(f)
    
    print(f"총 {len(data)}개 샘플 처리 중...")
    
    fixed_count = 0
    for sample in data:
        original_def_count = len(sample.get("cot_def_triples", []))
        original_triples_count = len(sample.get("cot_triples", []))
        
        sample = fix_sample(sample)
        
        new_def_count = len(sample.get("cot_def_triples", []))
        new_triples_count = len(sample.get("cot_triples", []))
        
        if original_def_count != new_def_count or original_triples_count != new_triples_count:
            fixed_count += 1
            if fixed_count <= 5:  # 처음 5개만 출력
                print(f"Sample {sample.get('index')}: "
                      f"def_triples {original_def_count} -> {new_def_count}, "
                      f"triples {original_triples_count} -> {new_triples_count}")
    
    print(f"\n총 {fixed_count}개 샘플 수정됨")
    
    # 백업 생성
    backup_path = input_path + ".backup"
    print(f"\n백업 생성 중: {backup_path}")
    with open(backup_path, "w") as f:
        json.dump(data, f, indent=4)
    
    # 수정된 파일 저장
    print(f"수정된 파일 저장 중: {input_path}")
    with open(input_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print("완료!")

if __name__ == "__main__":
    main()

