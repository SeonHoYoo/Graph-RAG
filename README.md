## Retriever 실행 방법

기본적으로 Search-R1의 bm25 retriever 환경을 사용합니다.

`/data3/seonhoyoo/multihopqa/Search-R1` 경로에 이미 Search-R1 repo가 클론되어 있고, retriever 환경이 설치되어 있습니다.

해당 경로에서 `sbatch retrieval_launch_bm25.sh` 를 실행하시면 됩니다.

### 주의사항

graphcheck 스크립트를 실행하는 노드와 동일한 노드에 retrieval_launch_bm25.sh 을 제출하셔야 합니다. 

sh 파일의 `#SBATCH --nodelist=**` 이 부분에서 노드 이름을 맞춰주세요.

### Retriever 환경 세팅 방법 (참고용)

아래 내용은 이미 제가 진행해 두어서 따로 실행하실 필요는 없습니다만, 참고로 적어둡니다.

1. Search-R1 repo clone 및 retriever environment 설치
    ```bash
    git clone https://github.com/PeterGriffinJin/Search-R1.git
    cd Search-R1

    # 출처: https://github.com/PeterGriffinJin/Search-R1
    conda create -n retriever python=3.10
    conda activate retriever

    # we recommend installing torch with conda for faiss-gpu
    conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install transformers datasets pyserini

    ## install the gpu version faiss to guarantee efficient RL rollout
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0

    ## API function
    pip install uvicorn fastapi

    ```

2. Index 다운로드
    ```bash
    # 출처: https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md
    save_path=downloads
    huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
    ```


2. Retriever launch
    ```bash
    conda activate retriever

    save_path=downloads
    index_file=$save_path/bm25
    corpus_file=$save_path/wiki-18.jsonl
    retriever_name=bm25

    python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
    ```


## GraphCheck 실행 방법(실험 환경 확인용)

Retriver 를 실행한 다음, graphcheck_v1/v2/v3.sh 파일을 sbatch로 제출하시면 각 버전의 GraphCheck가 실행됩니다.

```bash
sbatch graphcheck_v1.sh
```

GraphCheck 실행 결과는 `results` 폴더 내에 저장되며,

아래 명령어를 추가로 실행하면 결과가 저장된 경로에 EM/F1 score 가 담긴 .out 파일이 생성됩니다.
```
python utils/agg_eval.py --input_path <result json 파일 경로>
```

## Online Corrector (`run_online_corrector.sh`) 실행 방법

Vanilla SearchR1 에 **선택적 VeriGraph reasoning corrector** 를 결합한 파이프라인입니다.

샘플별 동작 흐름:
1. Vanilla SearchR1 을 먼저 실행 (Veri-Graph 미적용)
2. vanilla 의 검색 횟수가 임계값 이하이고 `<answer>` 를 만들었으면 그대로 채택 (저비용 경로)
3. 아니면 thinking budget 을 늘려 SearchR1 을 재실행하며, 매 턴 직후 질문 그래프와 think+docs 의 head/relation/tail 코사인 정렬을 콜백으로 주입 (정답은 노출하지 않고 추론만 유도)
4. corrector 도 `<answer>` 가 없으면 abstain 처리

### 1. 무엇을 어디에 놓아야 하나

- **Retriever (필수, 먼저 실행)**: 위 *Retriever 실행 방법* 대로 BM25 retriever 를 같은 노드에 띄웁니다. 기본 접속 주소는 `http://127.0.0.1:8000/retrieve` 입니다.
- **입력 데이터**: 이미 repo 에 포함되어 있어 따로 받을 필요가 없습니다. 스크립트는 `datasets/<dataset>/claims/train_sampled.json` 을 읽습니다. (`dataset ∈ {2wikimultihopqa, hotpotqa, musique}`)
- **모델**: 전부 HuggingFace Hub 에서 **자동 다운로드** 되므로 수동 배치가 필요 없습니다. (SearchR1 PPO 모델, `Llama-3.2-1B-Instruct` 어댑터들, `all-MiniLM-L6-v2` 인코더) 다운로드 캐시는 `HF_HOME`(기본 `~/.cache/huggingface`) 에 저장되며, 인터넷 및 HF 접근이 필요합니다.
- **conda 환경**: `graphcheck` env 를 스크립트가 자동으로 activate 합니다.

### 2. 다른 머신에서 돌릴 때 수정해야 할 경로

`graphqa/run_online_corrector.sh` 상단의 절대경로를 본인 환경에 맞게 바꿔주세요.

- `PROJECT_ROOT` — 이 repo 를 clone 한 위치
- `PY` — `graphcheck` conda env 의 `python3` 경로
- `#SBATCH --nodelist=` / 로그 경로 — retriever 와 **같은 노드** 로 맞춤

### 3. 실행

```bash
# SLURM 제출
sbatch graphqa/run_online_corrector.sh

# 또는 직접 실행
bash graphqa/run_online_corrector.sh
```

### 4. 주요 옵션 (환경변수로 override)

```bash
CORRECTOR_DATASETS="2wikimultihopqa hotpotqa musique" \
CORRECTOR_DATASET_LIMITS="500 500 1000" \
CORRECTOR_VANILLA_TRIGGER_THRESHOLD=4 \
CORRECTOR_COSINE_THRESHOLD=0.50 \
bash graphqa/run_online_corrector.sh
```

- `CORRECTOR_DATASETS` — 평가할 데이터셋 (기본 `2wikimultihopqa`)
- `CORRECTOR_DATASET_LIMITS` — 데이터셋별 샘플 수 (기본 `500`)
- `CORRECTOR_INPUT_FILENAME` — 입력 파일명 (기본 `train_sampled.json`)
- `CORRECTOR_RETRIEVER_URL` — retriever 주소 (기본 `http://127.0.0.1:8000/retrieve`)
- `CORRECTOR_OUT_DIR` — 출력 위치 (기본 `graphqa/outputs/online_corrector`)

### 5. 출력

`graphqa/outputs/online_corrector/` 아래에 생성됩니다.

- `<dataset>/online_corrector_<dataset>.csv` — per-sample 결과
- `<dataset>/online_corrector_<dataset>_summary.json` — 집계 지표
- `<dataset>/online_corrector_<dataset>_cases.jsonl` — 전체 trajectory + VeriGraph feedback
- `online_corrector_all.csv` / `online_corrector_all_summary.json` — 데이터셋 통합

> 참고: `graphqa/outputs/`, `infilling/output/`, `graph_data/`, `logs/` 등 **대용량 생성물은 `.gitignore` 로 제외**되어 GitHub 에는 올라가지 않습니다. 스크립트를 실행하면 로컬에 다시 생성됩니다.

## 실험 대상 데이터셋 관련 정보
- `datasets` 폴더 내에 각 데이터셋별 json 파일이 있습니다.
- 각 데이터셋의 `train_sampled.json` 이 저희가 실험으로 사용할 샘플 데이터셋입니다. (train set에서 일부 샘플링)
