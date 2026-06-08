# TASI: Triplet Alignment with Structural Importance

LLM 호출 없이 그래프 비교만으로 multi-hop QA를 수행하는 시스템.

## 디렉터리 구조

```
graphqa/
├── data/                    # GraphSample 로더 (Q, T, Sr, D)
│   ├── schema.py            # Triple, GraphSample 데이터 클래스
│   └── loader.py            # 4-그래프를 uid로 묶어 GraphSample 리스트 반환
├── tasi/                    # Module 1: TASI 코어
│   ├── embedding.py         # SentenceEncoder (sentence-transformers + 캐시)
│   ├── align.py             # align_triple, free_matching, pairwise_alignment_matrix
│   ├── ppr.py               # build_nx_graph, compute_ppr, triple_weight
│   ├── consistency.py       # propagation_consistency
│   └── core.py              # tasi(...) = WA × PC
├── pipeline.py              # Module 2: TASIPipeline (5가지 비교)
├── qa.py                    # Module 3: answer_question (slot filling QA)
├── evaluate.py              # Module 4: evaluate_single, evaluate_all + pandas 표
├── scripts/
│   ├── check_loader.py
│   ├── check_tasi_sample.py
│   ├── show_single.py       # 단일 샘플 디버깅
│   └── run_eval.py          # 전체 평가 진입점
├── tests/
│   └── test_tasi_core.py    # Module 1 단위 테스트 (assert 기반)
├── outputs/                 # 결과 CSV / JSON / PNG
└── run_tasi.sh              # 통합 실행 스크립트
```

## 빠른 시작

```bash
# 단위 테스트
bash graphqa/run_tasi.sh test

# 단일 샘플 디버깅 (2wiki x 3개)
bash graphqa/run_tasi.sh single

# 빠른 평가 (각 데이터셋 30개)
bash graphqa/run_tasi.sh quick

# 전체 평가 (2wiki 500 + hotpotqa 500 + musique 1000 = 2000개)
bash graphqa/run_tasi.sh full
```

또는 직접:

```bash
PY=/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3
${PY} graphqa/scripts/run_eval.py \
    --datasets 2wikimultihopqa hotpotqa musique \
    --limit 100 \
    --output-dir graphqa/outputs/exp01 \
    --save-plots
```

### Online / open-book-only triplets로 실행

기본 `triplets_train_sampled.json`는 gold evidence가 섞인 파일일 수 있다. BM25로 검색한 문서만
쓴 triplets 파일이 있으면 아래처럼 파일명을 지정한다.

```bash
PY=/data3/seonhoyoo/.conda/envs/graphcheck/bin/python3
${PY} graphqa/scripts/run_eval.py \
    --datasets 2wikimultihopqa musique \
    --triplets-filename triplets_train_sampled_open-book_top10.json \
    --output-dir graphqa/outputs/online_openbook_top10 \
    --save-plots
```

새 online triplets 파일이 필요하면 먼저 retriever 서버를 켠 뒤:

```bash
sbatch extract_triplets.sh 2wikimultihopqa Qwen/Qwen2.5-7B-Instruct train_sampled 10 open-book triplets_train_sampled_open-book_top10.json
```

위 6번째 인자가 결과 파일명이다.

## 데이터 위치

| 그래프 | 출처 |
|---|---|
| Q (Query, UNKNOWN 포함) | `results/{ds}/triplets/Qwen2.5-7B-Instruct/triplets_train_sampled.json` (`question_graph.triples`) |
| D (Document) | 위 동일 파일 (`doc_graph.triples`) |
| T (Think) | `graph_data/searchr1/0407(open-book)/reasoning_graph/reasoning_graph_{ds}_*.json` |
| Sr (Search) | `graph_data/searchr1/0407(open-book)/{ds}_vanilla_searchr1_*.json` 의 `retrieval_info.retrieval_turns[].query` 를 패턴 매칭으로 triple 화 |

uid 기준으로 정확히 매칭됨 (모든 데이터셋에서 100% 매칭, skip=0).

## 알고리즘 요약

```
TASI(A, B) = WA(A, B) × PC(A)

WA = Σ w(τ_A) · max_{τ_B ∈ B} align(τ_A, τ_B)              # weighted alignment
w(τ) = sqrt(PPR(h) · PPR(t))                                # 구조 중요도 (PPR)
align(τ_A, τ_B) = w_S sim(S) + w_R sim(R) + w_O sim(O)      # UNKNOWN 위치 제외
                ∨ 위 식에서 B의 head/tail swap 한 결과       # 방향 자동 반전
PC(A) = Π_t  |E_t ∩ used_in(T_{t+1})| / |E_t|               # multi-hop entity 연속성
```

5가지 비교:

| 이름 | 측정 대상 | 의미 |
|---|---|---|
| relevance     | TASI(Q, D)  | Document가 Query를 커버하는가 |
| consistency   | TASI(T, D)  | LLM 추론이 Document와 일치하는가 |
| alignment     | TASI(T, Q)  | LLM 추론이 Query 방향과 맞는가 |
| search_quality| TASI(Sr, Q) | 검색 키워드가 Query 의도를 반영했는가 |
| retrieval     | TASI(Sr, D) | 검색이 유용한 Document를 가져왔는가 |
| total         | 5개의 곱 | |

## QA (Module 3)

LLM 호출 없이 동작:

1. Q에서 UNKNOWN 슬롯 (`(ENTk)`) 식별
2. 각 Q triple과 D triple 사이 alignment 계산 → 슬롯 위치에 D entity 후보 누적
3. 동일 entity 토큰이 Q에 이미 known으로 등장하면 페널티 (자기 자신 채우기 방지)
4. Q_def 의 type 정의 (`(ENT1) is a country`)에 맞는 후보에 type bonus
5. yes/no 질문은 같은 relation 의 슬롯 값들을 비교하여 same/diff 판정

## 검증 결과 (각 데이터셋 30개)

```
=== Combined (n=90) ===
  accuracy = 0.100 | em = 0.100 | f1 = 0.126
  yes/no acc = 0.231, open acc = 0.078
  -- Mean scores by group (correct vs incorrect) --
       relevance_score : correct=0.756  incorrect=0.683  Δ=+0.073
     consistency_score : correct=0.249  incorrect=0.108  Δ=+0.140
       alignment_score : correct=0.253  incorrect=0.123  Δ=+0.130
  search_quality_score : correct=0.745  incorrect=0.728  Δ=+0.017
       retrieval_score : correct=0.620  incorrect=0.588  Δ=+0.031
      total_tasi_score : correct=0.057  incorrect=0.018  Δ=+0.039
  -- Pearson correlation with is_correct --
       relevance_score : r = +0.280
     consistency_score : r = +0.195
       alignment_score : r = +0.171
  search_quality_score : r = +0.079
       retrieval_score : r = +0.172
      total_tasi_score : r = +0.197
```

→ 모든 score가 정답 여부와 양의 상관관계. relevance(Q-D)와 consistency(T-D)가 가장 강한 신호.
QA accuracy 자체는 낮지만 (LLM 미사용 한계), TASI score가 정답성과 통계적으로 의미 있는 차이를 보임.

## 환경

```
conda env: graphcheck
- python 3.10
- networkx 3.4.2
- scipy 1.9.3, numpy
- pandas 1.5.2
- transformers 5.3.0, torch 2.6.0
- sentence-transformers 5.4.1 (all-MiniLM-L6-v2)
- matplotlib 3.10
```
