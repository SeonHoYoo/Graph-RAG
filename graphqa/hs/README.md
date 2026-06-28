# VeriGraph HS Experiments

이 폴더는 VeriGraph 기반 SearchR1 실험을 한 곳에서 실행하기 위한 harness입니다.
baseline과 새 `online_feedback` runner를 모두 `graphqa/hs` 아래에 둡니다.

핵심 목적은 새 방법이 실제로 좋아졌는지 판단할 기준을 먼저 고정하는 것입니다.
따라서 아래 세 방법은 같은 dataset slice, SearchR1 모델, retriever, graph model,
cosine threshold를 공유하도록 맞춰서 실행합니다.

## Baseline 종류

### `vanilla_searchr1`

VeriGraph를 전혀 쓰지 않고 SearchR1만 실행하는 기준선입니다.

흐름:

```text
question
 -> SearchR1 think/search 반복
 -> retrieved documents
 -> final answer
```

여기에는 question graph, document graph, think graph, Q-D/Q-T cosine 검증,
`<verigraph_check>` feedback, slot filling이 모두 없습니다. 즉 SearchR1 단독 성능을
보기 위한 baseline입니다.

실제 결과는 `fallback_verigraph` 실행 결과 안의 `vanilla_*` 컬럼에서 가져옵니다.
`run_online_corrector.py`가 항상 먼저 vanilla SearchR1을 실행하고 그 결과를 저장하기
때문입니다.

주요 컬럼:

```text
vanilla_predicted_answer
vanilla_em
vanilla_f1
vanilla_num_turns
```

### `analyze_verigraph`

기존 `searchr1_first` 모드입니다. SearchR1 실행 중에는 VeriGraph가 개입하지 않고,
SearchR1이 끝난 뒤에 결과를 사후 분석합니다.

흐름:

```text
question
 -> SearchR1 전체 실행
 -> SearchR1 trajectory 수집
    - think steps
    - search queries
    - retrieved documents
    - final answer
 -> question graph Q 추출
 -> document graph D 추출
 -> think graph T 추출
 -> Q-D / Q-T cosine 비교
 -> UNKNOWN slot filling 및 alignment 분석
```

이 방식은 다음 SearchR1 turn에 feedback을 넣지 않습니다. 따라서 새 online feedback
방법이 아니라, "SearchR1 결과를 나중에 VeriGraph로 분석했을 때 어떤 신호가 나오는가"를
보는 baseline입니다.

실행 코드는:

```text
graphqa/scripts/run_online_eval.py --trajectory-mode searchr1_first
```

출력 디렉터리:

```text
analyze_verigraph/
```

### `fallback_verigraph`

현재 repo에 있는 `run_online_corrector.py` 기반 baseline입니다. 완전한 single-pass
online feedback은 아니고, 먼저 vanilla SearchR1을 실행한 뒤 조건에 걸린 샘플에 대해
VeriGraph feedback corrector를 다시 실행합니다.

흐름:

```text
1. vanilla SearchR1 실행
2. vanilla가 짧게 끝나고 답도 있으면 그대로 사용
3. vanilla가 오래 걸렸거나 답이 없으면 VeriGraph feedback corrector로 재실행
4. corrector run에서는 turn마다 Q-D/Q-T alignment report를 <verigraph_check>로 주입
```

즉 `fallback_verigraph`는 SearchR1 본 실행에 처음부터 붙는 방식이 아니라,
vanilla가 부족해 보이는 경우 fallback으로 VeriGraph corrector를 사용하는 방식입니다.

실행 코드는:

```text
graphqa/scripts/run_online_corrector.py
```

출력 디렉터리:

```text
fallback_verigraph/
```

## 실행 방법

기본 실행:

```bash
bash graphqa/hs/run_baseline.sh
```

Slurm으로 제출하려면:

```bash
sbatch graphqa/hs/run_baseline.sh
```

기본 실행은 아래 두 baseline을 차례로 돌린 뒤 summary를 만듭니다.

```text
analyze_verigraph
fallback_verigraph
```

두 baseline은 실행 의존성이 없습니다. `fallback_verigraph`는
`analyze_verigraph`의 결과 파일을 읽지 않고, 자기 내부에서 vanilla SearchR1과
VeriGraph corrector를 다시 실행합니다. 다만 전체 baseline 비교표를 만들 때는 두
결과가 모두 필요하므로 기본 실행에서는 `analyze_verigraph`를 먼저 돌리고
`fallback_verigraph`를 이어서 돌립니다.

원하는 baseline만 실행하려면 `BASELINE_MODES`를 지정합니다.

```bash
BASELINE_MODES="analyze_verigraph" \
sbatch graphqa/hs/run_baseline.sh
```

```bash
BASELINE_MODES="fallback_verigraph" \
sbatch graphqa/hs/run_baseline.sh
```

기본값은 `2wikimultihopqa` 일부 샘플입니다. 보통은 아래처럼 세 데이터셋에 같은
limit을 주고 시작하면 됩니다.

```bash
DATASETS="2wikimultihopqa hotpotqa musique" \
LIMIT=100 \
RETRIEVER_URL=http://127.0.0.1:8003/retrieve \
sbatch graphqa/hs/run_baseline.sh
```

데이터셋별 limit을 다르게 주려면 `LIMIT=0`과 `DATASET_LIMITS`를 같이 씁니다.

```bash
DATASETS="2wikimultihopqa hotpotqa musique" \
LIMIT=0 \
DATASET_LIMITS="100 100 100" \
RETRIEVER_URL=http://127.0.0.1:8003/retrieve \
sbatch graphqa/hs/run_baseline.sh
```

SearchR1 retrieval server가 먼저 떠 있어야 합니다. 기본 URL은:

```text
http://127.0.0.1:8003/retrieve
```

다른 포트를 쓰면 `RETRIEVER_URL`을 바꾸면 됩니다.

## Online Feedback 실행

`online_feedback`은 vanilla SearchR1 prepass나 trigger 없이, 처음 SearchR1 실행부터
turn마다 VeriGraph observer를 켭니다.

흐름:

```text
question
 -> SearchR1 turn k: think/search
 -> documents retrieve
 -> Q/D/T graph 추출
 -> Q-D/Q-T cosine 비교
 -> verification_labels 생성
    - Q-D: document가 질문 requirement를 support하는가
    - Q-T: think가 질문 requirement를 다루는가
    - D-T: think claim이 document로 support/conflict되는가
    - query-Q: query가 unresolved requirement를 향하는가
 -> <vg_hint> feedback을 다음 turn prompt에 주입
 -> final answer
```

실행:

```bash
sbatch graphqa/hs/run_online_feedback.sh
```

기본 output root는 `graphqa/hs/outputs`입니다. 다른 위치에 저장하고 싶으면
`OUT_BASE`를 넘기면 됩니다.

```bash
OUT_BASE=/home/hyeseojeon/data/graph/graphqa/hs/outputs \
DATASETS="2wikimultihopqa hotpotqa musique" \
LIMIT=100 \
sbatch graphqa/hs/run_online_feedback.sh
```

기존 formatter와 label 기반 formatter를 비교할 수 있습니다.

```bash
FEEDBACK_ENGINE=legacy \
FEEDBACK_STYLE=repair_brief \
OUTPUT_SUFFIX=legacy_repair \
sbatch graphqa/hs/run_online_feedback.sh
```

```bash
FEEDBACK_ENGINE=labels \
FEEDBACK_STYLE=repair_brief \
OUTPUT_SUFFIX=labels_repair \
sbatch graphqa/hs/run_online_feedback.sh
```

`FEEDBACK_ENGINE=legacy`는 기존처럼 `alignment_rows`에서 바로 prompt 문장을 만듭니다.
`FEEDBACK_ENGINE=labels`는 먼저 `verification_labels`를 만들고, 그 label을 prompt로
변환합니다. stage 2 실험은 우선 `labels`를 중심으로 비교합니다.

## vLLM Latency 실행

Q/D/T graph extractor를 vLLM으로 띄운 뒤 `online_feedback`을 실행할 수 있습니다.
기본 graph model은 모두 Qwen2.5-0.5B 계열입니다.

먼저 graph vLLM 서버를 띄웁니다.

```bash
sbatch graphqa/hs/run_graph_vllm_server.sh
```

서버가 뜬 뒤 online feedback을 vLLM backend로 실행합니다.

```bash
GRAPH_BACKEND=vllm \
VLLM_BASE_URL=http://127.0.0.1:8006/v1 \
FEEDBACK_ENGINE=labels \
FEEDBACK_STYLE=repair_brief \
FEEDBACK_POSITION=inside_info \
OUTPUT_SUFFIX=labels_repair_vllm \
sbatch graphqa/hs/run_online_feedback.sh
```

Q/D/T 서버를 나눠 띄우는 경우에는 아래 URL을 따로 줄 수 있습니다.

```text
VLLM_QUESTION_BASE_URL
VLLM_DOCUMENT_BASE_URL
VLLM_THINK_BASE_URL
```

CSV와 case JSON에는 latency가 같이 저장됩니다.

```text
question_graph_sec
mean_document_graph_sec
mean_think_graph_sec
mean_alignment_sec
mean_verification_label_sec
mean_feedback_format_sec
mean_observer_total_sec
sample_wall_sec
```

Slurm log 마지막에도 dataset별 latency summary가 출력됩니다.

## Trajectory Judge

EM/F1만으로는 reasoning path가 좋아졌는지 보기 어렵기 때문에, 저장된 case JSON을
GPT judge로 평가할 수 있습니다. judge는 SearchR1을 다시 돌리지 않고, 이미 저장된
turn별 think/query/document/`vg_hint`를 읽어서 아래 항목을 판정합니다.

```text
answer_correct
final_faithful
unsupported_claims
conflicts
steering.effect
steering.bad_claim_repeated
main_failure_mode
verdict
```

실행 전 입력이 어떻게 만들어지는지 확인하려면 dry-run을 씁니다.

```bash
CASES=/home/hyeseojeon/data/graph/graphqa/hs/outputs/online_feedback/2wikimultihopqa/online_feedback_2wikimultihopqa_cases_steer10_bm25_labels_explicit_req_inside_n04.json \
DRY_RUN=1 \
sbatch graphqa/hs/run_trajectory_judge.sh
```

실제 judge 실행은 `model_library/openai_client.py`의 SKIML/LiteLLM 설정을 사용합니다.

```bash
CASES=/home/hyeseojeon/data/graph/graphqa/hs/outputs/online_feedback/2wikimultihopqa/online_feedback_2wikimultihopqa_cases_steer10_bm25_labels_explicit_req_inside_n04.json \
SKIML_API_KEY=... \
JUDGE_MODEL=openai/gpt-4.1-mini-2025-04-14 \
MAX_CASES=10 \
OUTPUT_PREFIX=explicit_req_judge_10 \
sbatch graphqa/hs/run_trajectory_judge.sh
```

기본 base URL은 `model_library/openai_client.py`에 있는 `SKIML_API_BASE`입니다.
다른 LiteLLM/SKIML endpoint를 쓰려면 `SKIML_API_BASE`를 지정합니다.

```bash
SKIML_API_BASE=https://147.47.200.198:7861 \
JUDGE_MODEL=your-judge-model \
CASES=... \
sbatch graphqa/hs/run_trajectory_judge.sh
```

출력은 `graphqa/hs/outputs/online_feedback` 아래에 저장됩니다.

```text
<OUTPUT_PREFIX>.json
<OUTPUT_PREFIX>_summary.json
<OUTPUT_PREFIX>_inputs.json
```

## 출력 구조

baseline 실행 결과는 기본적으로 아래에 저장됩니다.

```text
graphqa/hs/outputs/
  analyze_verigraph/
    <dataset>/online_eval_<dataset>.csv
    <dataset>/online_eval_<dataset>_cases.jsonl
    <dataset>/online_eval_<dataset>_summary.json
  fallback_verigraph/
    <dataset>/online_corrector_<dataset>.csv
    <dataset>/online_corrector_<dataset>_cases.jsonl
    <dataset>/online_corrector_<dataset>_summary.json
    online_corrector_all.csv
    online_corrector_all_summary.json
  baseline_summary.json
```

`baseline_summary.json`은 [sum_baseline.py](scripts/sum_baseline.py)가 두 CSV를 읽어 만든
요약 파일입니다.

요약에 들어가는 세 항목:

```text
vanilla_searchr1
analyze_verigraph
fallback_verigraph
```

online feedback 실행 결과도 같은 output root 아래에 저장됩니다.

```text
graphqa/hs/outputs/
  online_feedback/
    <dataset>/online_feedback_<dataset>_<job_id>.csv
    <dataset>/online_feedback_<dataset>_cases_<job_id>.json
    <dataset>/online_feedback_<dataset>_summary_<job_id>.json
    online_feedback_all_<job_id>.csv
    online_feedback_all_summary_<job_id>.json
```

Slurm 실행에서는 `<job_id>`가 자동으로 `SLURM_JOB_ID`가 됩니다. 로컬 실행에서는
`manual_YYYYMMDD_HHMMSS`가 붙습니다. 직접 지정하려면 `OUTPUT_SUFFIX`를 넘기면 됩니다.

기본 feedback style은 `natural_mismatch`입니다. 모델의 reasoning 경로를 강제로
바꾸지 않고, unsupported fact를 그대로 carry forward하지 않도록 evidence 상태를
알려줍니다.

## Q-D-T Analyzer

실행이 끝난 case JSON은 edge 단위 CSV로 펼쳐서 검수할 수 있습니다.

```bash
python graphqa/hs/scripts/analyze_qdt_trajectory.py \
  --cases graphqa/hs/outputs/online_feedback/2wikimultihopqa/online_feedback_2wikimultihopqa_cases_<suffix>.json \
  --out-csv graphqa/hs/outputs/qdt_edges_<suffix>.csv \
  --summary-json graphqa/hs/outputs/qdt_edges_summary_<suffix>.json \
  --threshold 0.50
```

생성되는 edge:

```text
Q-D
Q-T
D-T
query-Q
trajectory
```

따라서 최종 비교용 구조는:

```text
graphqa/hs/outputs/
  analyze_verigraph/
  fallback_verigraph/
  online_feedback/
  baseline_summary.json
```

## 왜 먼저 이걸 하나?

앞으로 만들 새 방법은 대략 아래 방향입니다.

```text
single-pass online VeriGraph feedback:
  SearchR1이 매 turn think/search/docs를 만들 때마다
  Q-D/Q-T를 즉시 비교하고
  다음 turn 또는 same-turn retry에 feedback을 반영
```

이 새 방법을 평가하려면 먼저 기존 기준선이 필요합니다.

```text
vanilla_searchr1:
  VeriGraph 없이 SearchR1만 썼을 때

analyze_verigraph:
  SearchR1 결과를 사후 분석만 했을 때

fallback_verigraph:
  기존 fallback corrector를 썼을 때
```

이 세 값을 고정해두면, 새 `single-pass` online feedback이 정확도, turn 수, latency,
wrong-to-correct, correct-to-wrong 측면에서 실제로 개선되는지 비교할 수 있습니다.
